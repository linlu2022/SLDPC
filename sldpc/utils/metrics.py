"""Classification metrics used in the SLDPC paper.

The paper reports three metrics in every table: Accuracy, Macro-F1, and
AUC (one-vs-rest for multiclass). This module provides thin wrappers
around :mod:`sklearn.metrics` so that downstream training code can emit
a uniform ``{"ACC": ..., "F1": ..., "AUC": ...}`` dict without worrying
about multiclass quirks.

Numerical conventions
---------------------
- All metrics are returned on the ``[0, 100]`` scale to match the paper
  tables (a value of ``85.81`` means ``85.81%``, not ``0.8581``).
- AUC uses one-vs-rest with macro averaging for multiclass problems
  (``C > 2``) and the standard binary form for two-class subtyping.
- Macro-F1 averages per-class F1 with equal weight, matching Eq. 16.
- All three metrics gracefully degrade to ``NaN`` when a class is absent
  from a particular fold, rather than raising.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


__all__ = ["compute_classification_metrics"]


ArrayLike = Union[np.ndarray, "torch.Tensor"]   # noqa: F821


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Detach and convert to CPU NumPy, accepting Tensors or arrays."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def compute_classification_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    num_classes: int,
) -> Dict[str, float]:
    """Return ``{"ACC", "F1", "AUC"}`` as percentages.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Ground-truth class indices in ``[0, num_classes)``.
    y_prob : array-like of shape (N, num_classes)
        Softmax probabilities or logits. Argmax is taken along dim 1
        for ACC/F1; raw values are used for AUC.
    num_classes : int
        Total number of classes.

    Returns
    -------
    dict[str, float]
        Keys ``"ACC"``, ``"F1"``, ``"AUC"``. Values are on the
        ``[0, 100]`` scale.
    """
    y_true = _to_numpy(y_true).astype(np.int64).ravel()
    y_prob = _to_numpy(y_prob)

    # Hard predictions for ACC / F1.
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(
        y_true, y_pred, average="macro", labels=list(range(num_classes)),
        zero_division=0,
    ) * 100.0

    # AUC handles binary vs. multiclass differently.
    try:
        if num_classes == 2:
            # roc_auc_score expects the positive-class score for binary.
            auc = roc_auc_score(y_true, y_prob[:, 1]) * 100.0
        else:
            auc = roc_auc_score(
                y_true, y_prob,
                labels=list(range(num_classes)),
                multi_class="ovr",
                average="macro",
            ) * 100.0
    except ValueError:
        # Raised when a class is missing from ``y_true`` in this fold.
        auc = float("nan")

    return {"ACC": acc, "F1": f1, "AUC": auc}
