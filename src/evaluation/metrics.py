"""
Comprehensive evaluation metrics module for JobGuard AI.
Focuses on PR-AUC, F1, Recall, Precision, ROC-AUC, and Confusion Matrix.
Does NOT treat accuracy as the headline metric for imbalanced data.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics at a specific probability threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Core fraud metrics
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cm_dict = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    # False positive rate (FPR) and specificity
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "headline_metrics": {
            "pr_auc": round(pr_auc, 4),
            "f1_score": round(f1, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "roc_auc": round(roc_auc, 4),
        },
        "operating_point": {
            "threshold": round(threshold, 4),
            "false_positive_rate": round(fpr, 4),
            "specificity": round(specificity, 4),
            "brier_score": round(brier, 4),
        },
        "confusion_matrix": cm_dict,
        "class_support": {
            "total_samples": len(y_true),
            "positive_samples": int(np.sum(y_true)),
            "negative_samples": int(len(y_true) - np.sum(y_true)),
            "fraud_rate": round(float(np.mean(y_true)), 4),
        },
    }


def compute_pr_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve coordinates."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds


def compute_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve coordinates."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds
