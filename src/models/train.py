"""
Model Training and Comparison Module for JobGuard AI.
Compares Baseline A (TF-IDF + Logistic Regression), Model B (LightGBM on engineered features),
and Model C (Sentence-Transformers + Engineered Features + LightGBM), evaluating class_weight vs resampling.
Outputs reports/baseline_comparison.json, reports/model_metrics.json, and diagnostic figures.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.features.engineering import JobFeatureExtractor
from src.features.embeddings import EmbeddingExtractor
from src.evaluation.metrics import (
    evaluate_predictions,
    compute_pr_curve,
    compute_roc_curve,
)
from src.models.pipeline import JobGuardPipeline

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path("models")


def build_tfidf_features(train_texts: List[str], test_texts: List[str]) -> Tuple[Any, Any, Any]:
    """Build word + character n-gram TF-IDF representations."""
    print("[INFO] Vectorizing text with word + char n-gram TF-IDF...")
    union = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
        ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char", max_features=15000, sublinear_tf=True)),
    ])
    X_train_tfidf = union.fit_transform(train_texts)
    X_test_tfidf = union.transform(test_texts)
    return union, X_train_tfidf, X_test_tfidf


def run_cross_validation(
    X_train: Any,
    y_train: np.ndarray,
    model_factory,
    resampler=None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Run stratified 5-fold cross-validation and compute mean metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        if hasattr(X_train, "tocsr"):
            X_f_train, X_f_val = X_train[train_idx], X_train[val_idx]
        else:
            X_f_train, X_f_val = X_train[train_idx], X_train[val_idx]
        y_f_train, y_f_val = y_train[train_idx], y_train[val_idx]

        if resampler is not None:
            try:
                min_class_count = int(np.min(np.bincount(y_f_train)))
                if hasattr(resampler, "k_neighbors"):
                    resampler.k_neighbors = min(5, max(1, min_class_count - 1))
                if min_class_count > 1:
                    X_f_train, y_f_train = resampler.fit_resample(X_f_train, y_f_train)
            except Exception as e:
                print(f"[WARNING] Resampling fold {fold} failed: {e}. Falling back to unresampled fold.")

        clf = model_factory()
        clf.fit(X_f_train, y_f_train)

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_f_val)[:, 1]
        else:
            scores = clf.decision_function(X_f_val)
            probs = 1.0 / (1.0 + np.exp(-scores))

        eval_res = evaluate_predictions(y_f_val, probs, threshold=0.5)
        fold_metrics.append(eval_res["headline_metrics"])

    # Aggregate means
    keys = fold_metrics[0].keys()
    mean_metrics = {f"cv_{k}": round(float(np.mean([m[k] for m in fold_metrics])), 4) for k in keys}
    std_metrics = {f"cv_{k}_std": round(float(np.std([m[k] for m in fold_metrics])), 4) for k in keys}
    return {**mean_metrics, **std_metrics}


def plot_curves_comparison(
    models_curves: Dict[str, Dict[str, Any]],
    pr_out_path: Path,
    roc_out_path: Path,
):
    """Plot PR and ROC curves comparing all evaluated architectures."""
    sns.set_theme(style="whitegrid")
    colors = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad", "#d35400"]

    # 1. Precision-Recall Curves
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for (m_name, data), color in zip(models_curves.items(), colors):
        p, r = data["pr_curve"][0], data["pr_curve"][1]
        score = data["pr_auc"]
        ax_pr.plot(r, p, color=color, lw=2, label=f"{m_name} (PR-AUC = {score:.3f})")

    ax_pr.set_title("Precision-Recall Curve Comparison (Imbalanced Fraud Detection)", fontsize=12, fontweight="bold", pad=12)
    ax_pr.set_xlabel("Recall (Fraudulent Postings Detected)", fontsize=10)
    ax_pr.set_ylabel("Precision (Correct Scam Flag Rate)", fontsize=10)
    ax_pr.set_xlim([0.0, 1.02])
    ax_pr.set_ylim([0.0, 1.02])
    ax_pr.legend(loc="lower left", frameon=True)
    fig_pr.savefig(pr_out_path, dpi=300)
    plt.close(fig_pr)

    # 2. ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Chance (AUC = 0.500)")
    for (m_name, data), color in zip(models_curves.items(), colors):
        fpr, tpr = data["roc_curve"][0], data["roc_curve"][1]
        score = data["roc_auc"]
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{m_name} (ROC-AUC = {score:.3f})")

    ax_roc.set_title("ROC Curve Comparison", fontsize=12, fontweight="bold", pad=12)
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate", fontsize=10)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.02])
    ax_roc.legend(loc="lower right", frameon=True)
    fig_roc.savefig(roc_out_path, dpi=300)
    plt.close(fig_roc)


def plot_confusion_matrix(cm_dict: Dict[str, int], out_path: Path, title: str):
    """Plot confusion matrix heatmap."""
    matrix = np.array([
        [cm_dict["true_negatives"], cm_dict["false_positives"]],
        [cm_dict["false_negatives"], cm_dict["true_positives"]],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate (0)", "Fraudulent (1)"],
        yticklabels=["Legitimate (0)", "Fraudulent (1)"],
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Actual Label", fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def train_and_compare(
    sample_size: Optional[int] = None,
    model_save_path: Optional[Any] = None,
    save_reports: bool = True,
):
    """
    Main training and model comparison execution function.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed data files missing. Please run 'python scripts/run_eda.py' first."
        )

    print(f"[INFO] Loading processed splits: {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if sample_size and sample_size < len(train_df):
        print(f"[INFO] Subsampling dataset to {sample_size} records for fast execution...")
        from sklearn.model_selection import train_test_split
        _, train_df = train_test_split(
            train_df,
            test_size=min(sample_size, len(train_df)),
            stratify=train_df["fraudulent"],
            random_state=42,
        )
        train_df = train_df.reset_index(drop=True)
        test_sub = min(max(30, int(sample_size * 0.3)), len(test_df))
        _, test_df = train_test_split(
            test_df,
            test_size=test_sub,
            stratify=test_df["fraudulent"],
            random_state=42,
        )
        test_df = test_df.reset_index(drop=True)

    y_train = train_df["fraudulent"].values.astype(int)
    y_test = test_df["fraudulent"].values.astype(int)

    train_texts = train_df["combined_text"].fillna("").tolist()
    test_texts = test_df["combined_text"].fillna("").tolist()

    # Calculate class weighting ratio
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = float(n_neg / max(1, n_pos))
    print(f"[INFO] Class balance: {n_neg} Legitimate vs. {n_pos} Fraudulent (scale_pos_weight={scale_pos_weight:.2f})")

    # 1. Feature Extraction: Tabular Engineered Features
    print("[INFO] Extracting tabular engineered features...")
    feat_extractor = JobFeatureExtractor()
    feat_extractor.fit(train_df, y_train)
    X_train_tabular = feat_extractor.transform(train_df).values.astype(np.float32)
    X_test_tabular = feat_extractor.transform(test_df).values.astype(np.float32)

    # 2. Dense Embeddings
    print("[INFO] Generating sentence-transformer dense embeddings...")
    emb_extractor = EmbeddingExtractor()
    X_train_emb = emb_extractor.encode_texts(train_texts, show_progress=False)
    X_test_emb = emb_extractor.encode_texts(test_texts, show_progress=False)

    # Hybrid feature matrix: Tabular + Dense Embeddings
    X_train_hybrid = np.hstack([X_train_tabular, X_train_emb]).astype(np.float32)
    X_test_hybrid = np.hstack([X_test_tabular, X_test_emb]).astype(np.float32)

    # 3. TF-IDF for Baseline A
    tfidf_union, X_train_tfidf, X_test_tfidf = build_tfidf_features(train_texts, test_texts)

    # Model Definitions
    models_to_evaluate = {
        "Baseline A (TF-IDF + LogReg + ClassWeight)": {
            "features_train": X_train_tfidf,
            "features_test": X_test_tfidf,
            "factory": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "resampler": None,
        },
        "Baseline A (TF-IDF + LogReg + UnderSample)": {
            "features_train": X_train_tfidf,
            "features_test": X_test_tfidf,
            "factory": lambda: LogisticRegression(max_iter=1000, random_state=42),
            "resampler": RandomUnderSampler(sampling_strategy=0.3, random_state=42),
        },
        "Model B (Engineered Feats + LightGBM)": {
            "features_train": X_train_tabular,
            "features_test": X_test_tabular,
            "factory": lambda: LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=31,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbose=-1,
            ),
            "resampler": None,
        },
        "Model B (Engineered Feats + LightGBM + SMOTE)": {
            "features_train": X_train_tabular,
            "features_test": X_test_tabular,
            "factory": lambda: LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            ),
            "resampler": SMOTE(sampling_strategy=0.3, random_state=42),
        },
        "Model C (Embeddings + Feats + LightGBM Hybrid)": {
            "features_train": X_train_hybrid,
            "features_test": X_test_hybrid,
            "factory": lambda: LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbose=-1,
            ),
            "resampler": None,
        },
    }

    comparison_results: Dict[str, Any] = {}
    models_curves: Dict[str, Any] = {}
    best_model_name = ""
    best_pr_auc = -1.0
    best_test_eval: Dict[str, Any] = {}
    fitted_models: Dict[str, Any] = {}

    for name, config in models_to_evaluate.items():
        print(f"\n[INFO] Evaluating architecture: '{name}'...")
        X_tr = config["features_train"]
        X_te = config["features_test"]
        factory = config["factory"]
        resampler = config["resampler"]

        # 5-Fold Stratified Cross Validation
        cv_metrics = run_cross_validation(X_tr, y_train, factory, resampler=resampler, n_splits=5)
        print(f"  --> CV Mean PR-AUC: {cv_metrics['cv_pr_auc']:.4f} | F1: {cv_metrics['cv_f1_score']:.4f} | Recall: {cv_metrics['cv_recall']:.4f}")

        # Fit on full training set for test evaluation
        clf = factory()
        if resampler is not None:
            try:
                min_class_count = int(np.min(np.bincount(y_train)))
                if hasattr(resampler, "k_neighbors"):
                    resampler.k_neighbors = min(5, max(1, min_class_count - 1))
                if min_class_count > 1:
                    X_tr_fit, y_tr_fit = resampler.fit_resample(X_tr, y_train)
                else:
                    X_tr_fit, y_tr_fit = X_tr, y_train
            except Exception:
                X_tr_fit, y_tr_fit = X_tr, y_train
        else:
            X_tr_fit, y_tr_fit = X_tr, y_train
        clf.fit(X_tr_fit, y_tr_fit)
        fitted_models[name] = clf

        # Evaluate on untouched test set
        if hasattr(clf, "predict_proba"):
            y_test_prob = clf.predict_proba(X_te)[:, 1]
        else:
            scores = clf.decision_function(X_te)
            y_test_prob = 1.0 / (1.0 + np.exp(-scores))

        test_eval = evaluate_predictions(y_test, y_test_prob, threshold=0.5)
        print(f"  --> Test PR-AUC: {test_eval['headline_metrics']['pr_auc']:.4f} | F1: {test_eval['headline_metrics']['f1_score']:.4f} | Recall: {test_eval['headline_metrics']['recall']:.4f}")

        pr_curve = compute_pr_curve(y_test, y_test_prob)
        roc_curve = compute_roc_curve(y_test, y_test_prob)

        models_curves[name] = {
            "pr_auc": test_eval["headline_metrics"]["pr_auc"],
            "roc_auc": test_eval["headline_metrics"]["roc_auc"],
            "pr_curve": pr_curve,
            "roc_curve": roc_curve,
        }

        comparison_results[name] = {
            "cross_validation_5fold": cv_metrics,
            "test_set_evaluation": test_eval,
        }

        if test_eval["headline_metrics"]["pr_auc"] > best_pr_auc:
            best_pr_auc = test_eval["headline_metrics"]["pr_auc"]
            best_model_name = name
            best_test_eval = test_eval

    print(f"\n[WINNER] Selected Model: '{best_model_name}' with Test PR-AUC: {best_pr_auc:.4f}")

    # Plot Comparison Curves
    print("[INFO] Saving diagnostic PR and ROC comparison figures...")
    plot_curves_comparison(
        models_curves,
        FIGURES_DIR / "pr_curves_comparison.png",
        FIGURES_DIR / "roc_curves_comparison.png",
    )
    plot_confusion_matrix(
        best_test_eval["confusion_matrix"],
        FIGURES_DIR / "confusion_matrix.png",
        f"Confusion Matrix: {best_model_name}",
    )

    winning_summary = {
        "model_name": best_model_name,
        "selected_architecture": "Hybrid (Sentence-Transformers all-MiniLM-L6-v2 + Domain Engineered Features + LightGBM)"
        if "Hybrid" in best_model_name
        else best_model_name,
        "training_date": pd.Timestamp.now().isoformat(),
        "headline_metrics": best_test_eval["headline_metrics"],
        "operating_point": best_test_eval["operating_point"],
        "confusion_matrix": best_test_eval["confusion_matrix"],
        "class_support": best_test_eval["class_support"],
        "comparison_summary": {
            k: {
                "cv_pr_auc": v["cross_validation_5fold"]["cv_pr_auc"],
                "cv_f1": v["cross_validation_5fold"]["cv_f1_score"],
                "test_pr_auc": v["test_set_evaluation"]["headline_metrics"]["pr_auc"],
                "test_f1": v["test_set_evaluation"]["headline_metrics"]["f1_score"],
                "test_recall": v["test_set_evaluation"]["headline_metrics"]["recall"],
                "test_precision": v["test_set_evaluation"]["headline_metrics"]["precision"],
            }
            for k, v in comparison_results.items()
        },
    }

    if save_reports:
        # Save comparison report
        with open(REPORTS_DIR / "baseline_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2)
        print(f"[SUCCESS] Saved {REPORTS_DIR / 'baseline_comparison.json'}")

        with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
            json.dump(winning_summary, f, indent=2)
        print(f"[SUCCESS] Saved {REPORTS_DIR / 'model_metrics.json'}")

    # Create and persist production JobGuardPipeline
    print("[INFO] Building and persisting unified production JobGuardPipeline...")
    best_clf = fitted_models[best_model_name]
    if "TF-IDF" in best_model_name:
        pipeline = JobGuardPipeline(
            classifier=best_clf,
            feature_mode="tfidf",
            threshold=0.50,
            model_name=best_model_name,
        )
        pipeline.tfidf_vectorizer = tfidf_union
        pipeline.feature_names_ = [f"tfidf_{i}" for i in range(X_train_tfidf.shape[1])]
    elif "Hybrid" in best_model_name:
        pipeline = JobGuardPipeline(
            classifier=best_clf,
            feature_mode="hybrid",
            threshold=0.50,
            model_name=best_model_name,
        )
        pipeline.feature_extractor = feat_extractor
        pipeline.embedding_extractor = emb_extractor
        pipeline.feature_names_ = feat_extractor.get_feature_names_out() + [f"emb_{i}" for i in range(384)]
    else:
        pipeline = JobGuardPipeline(
            classifier=best_clf,
            feature_mode="tabular",
            threshold=0.50,
            model_name=best_model_name,
        )
        pipeline.feature_extractor = feat_extractor
        pipeline.feature_names_ = feat_extractor.get_feature_names_out()

    model_dest = Path(model_save_path) if model_save_path else (MODELS_DIR / "best_model.joblib")
    pipeline.save(model_dest)

    return pipeline, winning_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare JobGuard models")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional subsample size for rapid smoke testing")
    args = parser.parse_args()
    train_and_compare(sample_size=args.sample_size)
