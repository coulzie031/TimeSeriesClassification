from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)


def classification_metrics(
    y_true: np.ndarray, probs: np.ndarray, class_weights: np.ndarray | None = None
) -> dict[str, float]:
    """
    Computes robust metrics for imbalanced transient classification (e.g., LSST).
    """
    pred = probs.argmax(axis=1)

    sample_weight = None
    if class_weights is not None:
        sample_weight = class_weights[y_true]

    # 1. Brier Score (Multi-class Mean Squared Error of Probabilities)
    # Excellent for evaluating probability calibration
    y_true_one_hot = np.zeros_like(probs)
    y_true_one_hot[np.arange(y_true.size), y_true] = 1

    if sample_weight is not None:
        brier_score = np.average(
            np.sum((probs - y_true_one_hot) ** 2, axis=1), weights=sample_weight
        )
    else:
        brier_score = np.mean(np.sum((probs - y_true_one_hot) ** 2, axis=1))

    # 2. Log-Loss (Cross-Entropy) - PLAsTiCC challenge standard
    # Heavily penalizes confident incorrect predictions (eps prevents log(0))
    logloss = float(log_loss(y_true, probs, sample_weight=sample_weight))

    # 3. Macro PR-AUC (Average Precision)
    # Better than AUROC for highly imbalanced data as it focuses on minority class P/R
    pr_auc = float(average_precision_score(y_true_one_hot, probs, average="macro"))

    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier_score": float(brier_score),
        "log_loss": logloss,
        "macro_pr_auc": pr_auc,
    }
