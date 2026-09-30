"""
SHAP Explainability Module for JobGuard AI.
Computes TreeExplainer SHAP values for LightGBM models,
generates global feature importance reports, and creates summary figures.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def compute_shap_explanations(
    pipeline: Any,
    test_df: pd.DataFrame,
    max_samples: int = 500,
    output_dir: Path = REPORTS_DIR,
) -> Dict[str, Any]:
    """
    Compute SHAP values for the pipeline's underlying tree classifier.
    Saves reports/shap_global_importance.json and reports/figures/shap_summary.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Subsample test set for faster SHAP computation if needed
    if len(test_df) > max_samples:
        sample_df = test_df.sample(max_samples, random_state=42).copy()
    else:
        sample_df = test_df.copy()

    X_mat, feat_names = pipeline.transform_features(sample_df)
    clf = pipeline.classifier

    # Check if tree-based or linear
    is_tree = hasattr(clf, "tree_") or hasattr(clf, "booster_") or "LightGBM" in str(type(clf))

    if is_tree:
        print(f"[INFO] Computing SHAP values on {len(sample_df)} instances using TreeExplainer...")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_mat)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals_pos = shap_values[1]
        else:
            shap_vals_pos = shap_values
        mean_abs_shap = np.mean(np.abs(shap_vals_pos), axis=0)
    else:
        print("[INFO] Computing feature importance weights from Linear Model coefficients...")
        if hasattr(pipeline, "tfidf_vectorizer") and pipeline.tfidf_vectorizer is not None:
            feat_names = list(pipeline.tfidf_vectorizer.get_feature_names_out())
        coefs = clf.coef_[0]
        mean_abs_shap = np.abs(coefs)

    top_indices = np.argsort(mean_abs_shap)[::-1]

    global_rankings = []
    for rank, idx in enumerate(top_indices[:30], 1):
        name = feat_names[idx]
        score = float(mean_abs_shap[idx])
        global_rankings.append({
            "rank": rank,
            "feature": name,
            "mean_abs_shap": round(score, 5),
            "is_embedding": name.startswith("emb_"),
        })

    # Plot SHAP summary (top 15 features)
    fig, ax = plt.subplots(figsize=(9, 6))
    top_15_idx = top_indices[:15][::-1]
    y_pos = np.arange(len(top_15_idx))
    scores_15 = [mean_abs_shap[i] for i in top_15_idx]
    names_15 = [feat_names[i] for i in top_15_idx]

    colors = ["#2b5c8f" if not n.startswith("emb_") else "#e67e22" for n in names_15]
    ax.barh(y_pos, scores_15, color=colors, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_15, fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Model Fraud Probability)", fontsize=10)
    ax.set_title("Global Feature Importance (SHAP TreeExplainer)", fontsize=12, fontweight="bold", pad=12)

    plot_path = FIGURES_DIR / "shap_summary.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    shap_summary = {
        "model_analyzed": getattr(pipeline, "model_name", "JobGuard-Pipeline"),
        "samples_evaluated": len(sample_df),
        "total_features": len(feat_names),
        "top_features": global_rankings,
    }

    json_path = output_dir / "shap_global_importance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(shap_summary, f, indent=2)

    print(f"[SUCCESS] Saved SHAP analysis to '{json_path}' and '{plot_path}'.")
    return shap_summary
