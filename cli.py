"""
JobGuard AI - Command Line Interface (CLI)
Provides commands for model training, evaluation, threshold calibration, and posting prediction.

Usage:
    python cli.py train [--sample-size N]
    python cli.py evaluate [--threshold T]
    python cli.py predict --file data/samples/sample_scam.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.models.pipeline import JobGuardPipeline
from src.models.train import train_and_compare
from src.evaluation.thresholds import analyze_thresholds
from src.evaluation.error_analysis import perform_error_analysis
from src.explain.shap_explainer import compute_shap_explanations
from src.explain.reasons import generate_plain_english_reasons

MODEL_PATH = Path("models/best_model.joblib")
TEST_DATA_PATH = Path("data/processed/test.csv")


def cmd_train(args):
    """Train and compare models, persisting the winning architecture."""
    print("=" * 60)
    print("JobGuard AI: Model Training & Architecture Selection")
    print("=" * 60)
    train_and_compare(sample_size=args.sample_size)


def cmd_evaluate(args):
    """Run evaluation, threshold analysis, error analysis, and SHAP explainability."""
    print("=" * 60)
    print("JobGuard AI: Model Evaluation & Explainability Pipeline")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"[FATAL] Model artifact '{MODEL_PATH}' not found. Please run 'python cli.py train' first.")
        sys.exit(1)

    if not TEST_DATA_PATH.exists():
        print(f"[FATAL] Test data '{TEST_DATA_PATH}' not found. Please run 'python scripts/run_eda.py' first.")
        sys.exit(1)

    print(f"[INFO] Loading production pipeline from '{MODEL_PATH}'...")
    pipeline = JobGuardPipeline.load(MODEL_PATH)

    print(f"[INFO] Loading test dataset from '{TEST_DATA_PATH}'...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    y_test = test_df["fraudulent"].values.astype(int)

    # Predictions
    print("[INFO] Computing test predictions and probabilities...")
    probs = pipeline.predict_proba(test_df)[:, 1]

    # 1. Threshold analysis
    print("[INFO] Running precision-recall threshold trade-off analysis...")
    thresh_summary = analyze_thresholds(y_test, probs)
    opt_point = thresh_summary["recommended_operating_points"]["balanced_f1_optimal"]
    print(f"  --> Optimal F1 Operating Point: Threshold={opt_point['threshold']} | F1={opt_point['f1_score']} | Recall={opt_point['recall']} | Precision={opt_point['precision']}")

    # 2. Error analysis
    print("[INFO] Running qualitative and statistical error analysis...")
    err_summary = perform_error_analysis(pipeline, test_df, threshold=float(opt_point["threshold"]))
    print(f"  --> False Positives: {err_summary['evaluation_summary']['false_positives']} | False Negatives: {err_summary['evaluation_summary']['false_negatives']}")

    # 3. SHAP global explainability
    print("[INFO] Computing SHAP TreeExplainer feature importances...")
    compute_shap_explanations(pipeline, test_df, max_samples=400)

    print("\n[SUCCESS] Full evaluation, threshold analysis, error audit, and SHAP explainability generated in reports/.")


def cmd_predict(args):
    """Classify a single job posting from JSON file or interactive input."""
    if not MODEL_PATH.exists():
        print(f"[FATAL] Model artifact '{MODEL_PATH}' not found. Please run 'python cli.py train' first.")
        sys.exit(1)

    pipeline = JobGuardPipeline.load(MODEL_PATH)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[ERROR] Input file not found: {file_path}")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            posting = json.load(f)
    else:
        print("\n--- Enter Job Details Interactively ---")
        title = input("Job Title: ").strip()
        company_profile = input("Company Profile (or empty): ").strip()
        description = input("Job Description: ").strip()
        requirements = input("Requirements (or empty): ").strip()
        benefits = input("Benefits (or empty): ").strip()
        posting = {
            "title": title,
            "company_profile": company_profile,
            "description": description,
            "requirements": requirements,
            "benefits": benefits,
            "has_company_logo": 1 if company_profile else 0,
            "telecommuting": 0,
        }

    probs = pipeline.predict_proba([posting])
    fraud_prob = float(probs[0, 1])

    if fraud_prob < 0.30:
        risk_level = "LOW RISK"
        color_prefix = "\033[92m"  # Green
    elif fraud_prob < 0.60:
        risk_level = "MODERATE RISK"
        color_prefix = "\033[93m"  # Yellow
    else:
        risk_level = "HIGH RISK (SCAM ALERT)"
        color_prefix = "\033[91m"  # Red
    color_reset = "\033[0m"

    is_fraud = fraud_prob >= pipeline.threshold
    reasons = generate_plain_english_reasons(posting, fraud_prob, pipeline.threshold)

    print("\n" + "=" * 65)
    print("JOBGUARD AI: PREDICTION RESULT")
    print("=" * 65)
    print(f"Posting Title:       {posting.get('title')}")
    print(f"Risk Assessment:     {color_prefix}{risk_level}{color_reset}")
    print(f"Fraud Probability:   {fraud_prob:.2%}")
    print(f"Decision Threshold:  {pipeline.threshold:.2f}")
    print(f"Binary Flag:         {'FRAUDULENT' if is_fraud else 'LEGITIMATE'}")
    print("\nKey Contributing Reasons:")
    for idx, r in enumerate(reasons, 1):
        print(f"  {idx}. {r}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="JobGuard AI CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train and select best model")
    train_parser.add_argument("--sample-size", type=int, default=None, help="Subsample rows for fast training")
    train_parser.set_defaults(func=cmd_train)

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run full evaluation and explainability")
    eval_parser.set_defaults(func=cmd_evaluate)

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Predict risk for a job posting")
    predict_parser.add_argument("--file", type=str, help="Path to JSON file containing posting attributes")
    predict_parser.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
