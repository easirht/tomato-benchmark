"""Classification metrics (10-class, plus the derived healthy-vs-diseased binary task)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, healthy_idx: int, num_classes: int) -> dict:
    probs = softmax(logits.astype(np.float64))
    pred = probs.argmax(1)
    labels = list(range(num_classes))
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=labels, zero_division=0)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", labels=labels, zero_division=0)),
        "per_class_f1": [float(x) for x in f1_score(y_true, pred, average=None, labels=labels, zero_division=0)],
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
    }
    try:                                    # needs every class present in y_true
        out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                                                        labels=labels))
    except ValueError:
        out["roc_auc_ovr_macro"] = None
    yb, pb = (y_true != healthy_idx).astype(int), (pred != healthy_idx).astype(int)   # 1 = diseased
    out["binary_accuracy"] = float(accuracy_score(yb, pb))
    out["binary_f1_diseased"] = float(f1_score(yb, pb, zero_division=0))
    out["binary_f1_healthy"] = float(f1_score(1 - yb, 1 - pb, zero_division=0))
    try:
        out["binary_auc"] = float(roc_auc_score(yb, 1.0 - probs[:, healthy_idx]))
    except ValueError:
        out["binary_auc"] = None
    return out
