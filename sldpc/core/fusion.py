"""Weight Fusion Mechanism (WFM) for Slide-Level Dual-Prompt Collaboration.

This module implements Equation 9 from the SLDPC paper:

    P_tilde = omega * P_prime + (1 - omega) * P

where:
    - P is the base prompt (Stage-1 trained, frozen during inference).
    - P_prime is the task-specific prompt (Stage-2 trained).
    - omega is the fusion weight in [0, 1], an inference-time hyperparameter.

The fusion mechanism is post-hoc and does not participate in gradient-based
optimization. The base branch is always detached to prevent gradient leakage
from the frozen prompt to the optimizer.

References
----------
    Eq. 9 in the main paper:  P_tilde = omega * P' + (1 - omega) * P
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F


__all__ = ["l2norm", "mix_ctx"]


# -----------------------------------------------------------------------------
# Normalization helper
# -----------------------------------------------------------------------------

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize ``x`` along ``dim`` in a numerically safe way.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary shape.
    dim : int, optional
        Dimension along which to normalize. Default is the last dimension.
    eps : float, optional
        Small constant added to the denominator to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as ``x`` with unit L2 norm along ``dim``.
    """
    return F.normalize(x, p=2.0, dim=dim, eps=eps)


# -----------------------------------------------------------------------------
# Weight Fusion Mechanism
# -----------------------------------------------------------------------------

def mix_ctx(
    ctx_frozen: torch.Tensor,
    ctx_learnable: torch.Tensor,
    omega: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Linearly fuse the frozen base prompt P with the learnable prompt P'.

    Implements Equation 9 of the paper:

        ctx_mix = omega * ctx_learnable + (1 - omega) * ctx_frozen

    The ``ctx_frozen`` tensor is detached internally so that no gradient
    flows back into the base prompt during Stage-2 training. Only the
    learnable branch retains gradients.

    Parameters
    ----------
    ctx_frozen : torch.Tensor
        Frozen base prompt P with shape ``(..., n_ctx, dim)``.
    ctx_learnable : torch.Tensor
        Learnable task-specific prompt P' with shape identical to
        ``ctx_frozen``.
    omega : float or torch.Tensor
        Fusion weight in [0, 1]. If a Tensor is passed it must be a scalar
        (``numel() == 1``); this allows future extensions where omega is
        itself a learnable parameter.

    Returns
    -------
    torch.Tensor
        The fused prompt of the same shape as the inputs.

    Raises
    ------
    ValueError
        If shapes mismatch, if ``omega`` is a non-scalar tensor, or if a
        Python ``omega`` lies outside [0, 1].
    """
    # ------------------------- shape validation --------------------------
    if ctx_frozen.shape != ctx_learnable.shape:
        raise ValueError(
            "ctx_frozen and ctx_learnable must have the same shape; "
            f"got {tuple(ctx_frozen.shape)} vs {tuple(ctx_learnable.shape)}."
        )

    # ------------------------- omega handling ----------------------------
    if isinstance(omega, torch.Tensor):
        if omega.numel() != 1:
            raise ValueError(
                "omega Tensor must be a scalar (numel == 1); "
                f"got shape {tuple(omega.shape)}."
            )
        omega_val = omega.to(dtype=ctx_frozen.dtype, device=ctx_frozen.device)
    else:
        omega_float = float(omega)
        if not (0.0 <= omega_float <= 1.0):
            raise ValueError(
                f"omega must be in [0, 1]; got {omega_float}."
            )
        omega_val = torch.tensor(
            omega_float, device=ctx_frozen.device, dtype=ctx_frozen.dtype
        )

    # ------------------------- fusion ------------------------------------
    # Detach the frozen branch so no gradient flows into the base prompt.
    ctx_frozen_det = ctx_frozen.detach()
    return omega_val * ctx_learnable + (1.0 - omega_val) * ctx_frozen_det
