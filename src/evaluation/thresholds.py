"""
Threshold Analysis Module for JobGuard AI.
Analyzes operating trade-offs between precision and recall across decision thresholds,
identifies optimal F1, high-precision, and high-recall operating points,
and outputs reports/threshold_analysis.json and reports/figures/threshold_tradeoffs.png.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def analyze_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path = REPORTS_DIR,
) -> Dict[str, Any]:
    """
    Evaluate binary classification performance across candidate decision thresholds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    threshold_range = np.linspace(0.05, 0.95, 19)
    records = []

    best_f1 = -1.0
    best_f1_thresh = 0.50

    for thresh in threshold_range:
        y_pred = (y_prob >= thresh).astype(int)
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

        records.append({
            "threshold": round(float(thresh), 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positive_rate": round(fpr, 4),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = float(thresh)

    df_thresh = pd.DataFrame(records)

    # Identify key operating points
    # 1. Balanced F1 optimal
    f1_optimal = df_thresh.loc[df_thresh["f1_score"].idxmax()].to_dict()

    # 2. High precision operating point (conservative, minimal false alarms)
    high_prec_candidates = df_thresh[df_thresh["precision"] >= 0.85]
    if not high_prec_candidates.empty:
        high_prec_point = high_prec_candidates.sort_values(by="recall", ascending=False).iloc[0].to_dict()
    else:
        high_prec_point = df_thresh.loc[df_thresh["precision"].idxmax()].to_dict()

    # 3. High recall operating point (sensitive audit, minimal missed scams)
    high_rec_candidates = df_thresh[df_thresh["recall"] >= 0.85]
    if not high_rec_candidates.empty:
        high_rec_point = high_rec_candidates.sort_values(by="precision", ascending=False).iloc[0].to_dict()
    else:
        high_rec_point = df_thresh.loc[df_thresh["recall"].idxmax()].to_dict()

    # Plot Trade-offs
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(df_thresh["threshold"], df_thresh["precision"], "b-o", label="Precision (Reliability)", lw=2, ms=4)
    ax.plot(df_thresh["threshold"], df_thresh["recall"], "r-s", label="Recall (Scam Capture Rate)", lw=2, ms=4)
    ax.plot(df_thresh["threshold"], df_thresh["f1_score"], "g-^", label="F1 Score", lw=2, ms=4)

    # Highlight optimal operating point
    ax.axvline(best_f1_thresh, color="gray", linestyle="--", alpha=0.7, label=f"Optimal F1 ({best_f1_thresh:.2f})")

    ax.set_title("Precision, Recall, and F1 Trade-offs vs. Decision Threshold", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Probability Threshold", fontsize=10)
    ax.set_ylabel("Metric Value", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=True, loc="center left")

    tradeoff_plot = FIGURES_DIR / "threshold_tradeoffs.png"
    fig.savefig(tradeoff_plot, dpi=300)
    plt.close(fig)

    threshold_summary = {
        "recommended_operating_points": {
            "balanced_f1_optimal": f1_optimal,
            "high_precision_conservative": high_prec_point,
            "high_recall_audit": high_rec_point,
        },
        "threshold_schedule": records,
    }

    out_json = output_dir / "threshold_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(threshold_summary, f, indent=2)

    print(f"[SUCCESS] Saved threshold analysis to '{out_json}' and '{tradeoff_plot}'.")
    return threshold_summary
