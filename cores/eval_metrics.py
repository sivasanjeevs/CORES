"""AUROC and FPR95 using scikit-learn."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc_fpr95(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float]:
    """
    y_true: 0 = ID, 1 = OOD. scores: higher = more OOD.
    FPR95: false positive rate at 95% true positive rate (OOD recall).
    """
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    auroc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    hit = np.where(tpr >= 0.95)[0]
    if hit.size == 0:
        fpr95 = float("nan")
    else:
        fpr95 = float(fpr[int(hit[0])])
    return auroc, fpr95
