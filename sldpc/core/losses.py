"""Contrastive losses for Slide-Level Dual-Prompt Collaboration.

This module provides the symmetric InfoNCE loss that implements Equations
11-13 of the SLDPC paper, together with a convenience wrapper designed to
consume the extended batches produced by the dynamic hard-negative sampler.

The loss is structured as two layers:

1. :func:`symmetric_info_nce`  — low-level primitive operating on aligned
   ``(N, D) x (N, D)`` feature pairs. This mirrors the pure mathematical
   form of Eq. 11.

2. :func:`hard_negative_contrastive_loss` — high-level convenience wrapper
   consuming the ``(feat_ext, label_ext, txt_bank)`` tuple produced by
   :func:`sldpc.core.negative_sampler.dynamic_hard_negative_sampler`. It
   indexes ``txt_bank[label_ext]`` to form the aligned text tensor and then
   delegates to :func:`symmetric_info_nce`.

Both layers share the same underlying math and are therefore numerically
equivalent.

References
----------
    Eqs. 11-13 of the paper.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F

from .fusion import l2norm


__all__ = [
    "symmetric_info_nce",
    "info_nce_image_to_text",
    "info_nce_text_to_image",
    "hard_negative_contrastive_loss",
    "classification_ce_loss",
]


# -----------------------------------------------------------------------------
# Numerical guard
# -----------------------------------------------------------------------------

# Clamp logits before softmax to prevent overflow in fp16 and extreme
# temperature regimes.
_LOGITS_CLAMP_MIN = -100.0
_LOGITS_CLAMP_MAX = 100.0


def _resolve_tau(tau: Union[float, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    """Convert ``tau`` to a scalar tensor on the same device/dtype as ``ref``."""
    if isinstance(tau, torch.Tensor):
        if tau.numel() != 1:
            raise ValueError(
                f"tau Tensor must be a scalar; got shape {tuple(tau.shape)}."
            )
        return tau.to(dtype=ref.dtype, device=ref.device)
    tau_float = float(tau)
    if tau_float <= 0:
        raise ValueError(f"tau must be positive; got {tau_float}.")
    return torch.tensor(tau_float, device=ref.device, dtype=ref.dtype)


# -----------------------------------------------------------------------------
# Low-level primitive: Symmetric InfoNCE
# -----------------------------------------------------------------------------

def symmetric_info_nce(
    img_feat: torch.Tensor,
    txt_feat: torch.Tensor,
    tau: Union[float, torch.Tensor] = 0.07,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the symmetric InfoNCE loss (Eq. 11).

    The loss is the average of image-to-text and text-to-image cross entropy
    over the similarity matrix ``img_feat @ txt_feat.T / tau``::

        L = 0.5 * ( CE(logits, labels) + CE(logits.T, labels) )

    Both branches use the same ``labels``, which default to the identity
    permutation ``[0, 1, ..., N-1]``. This makes the ``i``-th image the
    positive for the ``i``-th text and vice versa.

    Parameters
    ----------
    img_feat : torch.Tensor
        Image (slide) features of shape ``(N, D)``. Typically passed with
        ``.detach()`` during Stage-2 to prevent gradients from flowing into
        the visual encoder / feature bank.
    txt_feat : torch.Tensor
        Text (prompt-derived) features of shape ``(N, D)``. Must be aligned
        positionally with ``img_feat`` — i.e. ``txt_feat[i]`` is the
        positive match for ``img_feat[i]``.
    tau : float or torch.Tensor, optional
        Temperature coefficient. Must be > 0. Default 0.07 (matches the
        paper's experimental setup).
    labels : torch.Tensor, optional
        Shape ``(N,)``. Overrides the default identity positive assignment.
        Useful for advanced scenarios; leave as ``None`` in normal usage.

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Raises
    ------
    ValueError
        If input shapes mismatch or ``tau <= 0``.
    """
    if img_feat.shape != txt_feat.shape:
        raise ValueError(
            f"img_feat and txt_feat must have identical shape; "
            f"got {tuple(img_feat.shape)} vs {tuple(txt_feat.shape)}."
        )

    N = img_feat.shape[0]
    tau_val = _resolve_tau(tau, img_feat)

    # L2-normalize and compute logits.
    logits = l2norm(img_feat) @ l2norm(txt_feat).t() / tau_val    # (N, N)
    logits = logits.clamp(min=_LOGITS_CLAMP_MIN, max=_LOGITS_CLAMP_MAX)

    # Default positive-pair assignment: identity permutation.
    if labels is None:
        labels = torch.arange(N, device=img_feat.device, dtype=torch.long)
    else:
        if labels.shape != (N,):
            raise ValueError(
                f"labels must have shape ({N},); got {tuple(labels.shape)}."
            )
        labels = labels.to(device=img_feat.device, dtype=torch.long)

    loss_i2t = F.cross_entropy(logits, labels, reduction="mean")
    loss_t2i = F.cross_entropy(logits.t(), labels, reduction="mean")
    return 0.5 * (loss_i2t + loss_t2i)


def info_nce_image_to_text(
    img_feat: torch.Tensor,
    txt_feat: torch.Tensor,
    tau: Union[float, torch.Tensor] = 0.07,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-direction InfoNCE (image-to-text).

    This is the ``CE(logits, labels)`` branch of symmetric InfoNCE and is
    used by directionality ablations (paper Table 10 / appendix).
    """
    if img_feat.shape != txt_feat.shape:
        raise ValueError(
            f"img_feat and txt_feat must have identical shape; "
            f"got {tuple(img_feat.shape)} vs {tuple(txt_feat.shape)}."
        )

    N = img_feat.shape[0]
    tau_val = _resolve_tau(tau, img_feat)
    logits = l2norm(img_feat) @ l2norm(txt_feat).t() / tau_val
    logits = logits.clamp(min=_LOGITS_CLAMP_MIN, max=_LOGITS_CLAMP_MAX)

    if labels is None:
        labels = torch.arange(N, device=img_feat.device, dtype=torch.long)
    else:
        if labels.shape != (N,):
            raise ValueError(
                f"labels must have shape ({N},); got {tuple(labels.shape)}."
            )
        labels = labels.to(device=img_feat.device, dtype=torch.long)

    return F.cross_entropy(logits, labels, reduction="mean")


def info_nce_text_to_image(
    img_feat: torch.Tensor,
    txt_feat: torch.Tensor,
    tau: Union[float, torch.Tensor] = 0.07,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-direction InfoNCE (text-to-image)."""
    if img_feat.shape != txt_feat.shape:
        raise ValueError(
            f"img_feat and txt_feat must have identical shape; "
            f"got {tuple(img_feat.shape)} vs {tuple(txt_feat.shape)}."
        )

    N = img_feat.shape[0]
    tau_val = _resolve_tau(tau, img_feat)
    logits = l2norm(img_feat) @ l2norm(txt_feat).t() / tau_val
    logits = logits.clamp(min=_LOGITS_CLAMP_MIN, max=_LOGITS_CLAMP_MAX)

    if labels is None:
        labels = torch.arange(N, device=img_feat.device, dtype=torch.long)
    else:
        if labels.shape != (N,):
            raise ValueError(
                f"labels must have shape ({N},); got {tuple(labels.shape)}."
            )
        labels = labels.to(device=img_feat.device, dtype=torch.long)

    return F.cross_entropy(logits.t(), labels, reduction="mean")


# -----------------------------------------------------------------------------
# High-level: consume extended batch from hard-negative sampler
# -----------------------------------------------------------------------------

def hard_negative_contrastive_loss(
    feat_ext: torch.Tensor,
    label_ext: torch.Tensor,
    txt_bank: torch.Tensor,
    tau: Union[float, torch.Tensor] = 0.07,
) -> torch.Tensor:
    """SICL loss computed over the extended batch from the HNS sampler.

    The DHNO sampler returns ``(feat_ext, label_ext)`` where each consecutive
    block of ``k`` rows in ``feat_ext`` contains one positive followed by
    ``k - 1`` hard-negative images, and ``label_ext[i]`` is the ground-truth
    class for ``feat_ext[i]``. This function pairs each feature with the
    text embedding of its own class via ``txt_bank[label_ext]``, yielding
    an aligned ``(N, D) x (N, D)`` tuple that can be fed to
    :func:`symmetric_info_nce`.

    Parameters
    ----------
    feat_ext : torch.Tensor
        Extended image features of shape ``(B*k, D)``.
    label_ext : torch.Tensor
        Class labels of shape ``(B*k,)`` with values in ``[0, C)``.
    txt_bank : torch.Tensor
        Per-class text features of shape ``(C, D)`` produced by the current
        learnable prompt P'.
    tau : float or torch.Tensor, optional
        Temperature, default 0.07.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if feat_ext.shape[0] != label_ext.shape[0]:
        raise ValueError(
            f"feat_ext and label_ext batch dims must match; "
            f"got {feat_ext.shape[0]} vs {label_ext.shape[0]}."
        )
    if label_ext.max().item() >= txt_bank.shape[0]:
        raise ValueError(
            f"label_ext contains an index >= txt_bank.shape[0] "
            f"({label_ext.max().item()} >= {txt_bank.shape[0]})."
        )

    # Pair each feature with the text embedding of its ground-truth class.
    txt_feat_ext = txt_bank[label_ext]    # (B*k, D)
    return symmetric_info_nce(feat_ext, txt_feat_ext, tau=tau)


# -----------------------------------------------------------------------------
# Stage-1 helper: classification cross-entropy
# -----------------------------------------------------------------------------

def classification_ce_loss(
    img_feat: torch.Tensor,
    txt_bank: torch.Tensor,
    labels: torch.Tensor,
    tau: Union[float, torch.Tensor] = 0.07,
    apply_tau: bool = True,
) -> torch.Tensor:
    """Stage-1 classification cross-entropy loss (Eq. 7).

    Computes ``CE(logits, labels)`` with L2-normalized features. When
    ``apply_tau=True`` (paper form, Eq. 7), ``logits = cos / tau``. When
    ``apply_tau=False`` (legacy CoOp form used throughout the original
    SLDPC-TITAN codebase), ``logits = cos`` and softmax operates directly
    on cosine similarities in ``[-1, 1]``. The two regimes produce very
    different gradient scales; the legacy regime is the default in the
    public release and reproduces the paper's main-table numbers, while
    ``apply_tau=True`` matches the paper's §4.1 hyperparameter description
    and is exposed for transparency.

    Parameters
    ----------
    img_feat : torch.Tensor
        Slide features of shape ``(N, D)``.
    txt_bank : torch.Tensor
        Per-class text features of shape ``(C, D)``.
    labels : torch.Tensor
        Ground-truth class indices of shape ``(N,)``.
    tau : float or torch.Tensor, optional
        Temperature, default 0.07. Ignored when ``apply_tau=False``.
    apply_tau : bool, optional
        If ``True`` (default), scale cosine similarities by ``1/tau``
        before softmax (paper Eq. 7). If ``False``, use cosine
        similarities as raw logits (legacy CoOp behavior, the actual
        implementation used by the original SLDPC-TITAN codebase to
        obtain the numbers in the paper's main tables).

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if img_feat.shape[1] != txt_bank.shape[1]:
        raise ValueError(
            f"img_feat and txt_bank feature dims must match; "
            f"got {img_feat.shape[1]} vs {txt_bank.shape[1]}."
        )
    if labels.shape != (img_feat.shape[0],):
        raise ValueError(
            f"labels must have shape ({img_feat.shape[0]},); "
            f"got {tuple(labels.shape)}."
        )

    logits = l2norm(img_feat) @ l2norm(txt_bank).t()             # (N, C)
    if apply_tau:
        tau_val = _resolve_tau(tau, img_feat)
        logits = logits / tau_val
    logits = logits.clamp(min=_LOGITS_CLAMP_MIN, max=_LOGITS_CLAMP_MAX)

    labels = labels.to(device=img_feat.device, dtype=torch.long)
    return F.cross_entropy(logits, labels, reduction="mean")
