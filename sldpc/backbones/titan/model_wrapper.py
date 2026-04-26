"""Top-level prompted-TITAN model.

Couples a loaded TITAN backbone with a :class:`TitanPromptLearner` and
exposes a uniform ``forward(slide_feat, mode=...)`` that returns logits.
The TITAN backbone is frozen at init time so that only the prompt
learner's context parameters receive gradients.

Input expectations
------------------

``slide_feat`` is a ``(B, D_slide)`` tensor of slide-level features
produced offline by TITAN's ``slide_encoder`` (D_slide = 768 for
CONCH-v1.5-based TITAN). These features are projected through
``titan.vision_encoder.proj`` and L2-normalized inside this module;
callers should **not** pre-project.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prompt_learner import TitanPromptLearner


__all__ = ["PromptedTitan"]


class PromptedTitan(nn.Module):
    """Frozen TITAN + learnable prompt → similarity logits.

    Parameters
    ----------
    titan : nn.Module
        Loaded TITAN model. All parameters are frozen in-place.
    prompt_learner : TitanPromptLearner
        The only module that contributes trainable parameters.
    """

    def __init__(
        self,
        titan: nn.Module,
        prompt_learner: TitanPromptLearner,
    ) -> None:
        super().__init__()

        # Hold a non-owning reference to TITAN so that its (large) set
        # of frozen parameters is not iterated by optimizers that walk
        # ``self.parameters()``. We still move it to the correct device
        # and freeze all params explicitly.
        for p in titan.parameters():
            p.requires_grad = False
        object.__setattr__(self, "titan", titan)

        self.prompt_learner = prompt_learner

    # ------------------------------------------------------------------
    # Slide-feature projection
    # ------------------------------------------------------------------

    def project_slide(self, slide_feat: torch.Tensor) -> torch.Tensor:
        """Project pre-extracted slide features into the shared L2-unit sphere.

        Parameters
        ----------
        slide_feat : torch.Tensor
            Raw slide-level features of shape ``(B, D_slide)`` as
            produced by ``titan.slide_encoder``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, D_proj)``, L2-normalized.
        """
        proj = self.titan.vision_encoder.proj     # parameter matrix (D_slide, D_proj)
        dtype = proj.dtype
        device = proj.device
        slide_feat = slide_feat.to(device=device, dtype=dtype)
        return F.normalize(slide_feat @ proj, dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        slide_feat: torch.Tensor,
        mode: str = "task",
        omega: Optional[float] = None,
        return_text_features: bool = False,
    ) -> torch.Tensor:
        """Compute similarity logits for one mini-batch.

        Parameters
        ----------
        slide_feat : torch.Tensor
            ``(B, D_slide)`` raw slide features.
        mode : {"train", "base", "task", "fused"}
            Which prompt branch to use — delegated to
            :meth:`PromptLearnerBase.forward`.
        omega : float, optional
            Overrides the default fusion weight when ``mode="fused"``.
        return_text_features : bool, optional
            If ``True``, additionally return the ``(C, D_proj)`` text
            features produced by the chosen prompt branch.

        Returns
        -------
        torch.Tensor
            ``(B, C)`` similarity logits (cosine similarity between
            projected slide and text features, before temperature
            scaling).
        """
        slide_norm = self.project_slide(slide_feat)             # (B, D_proj)
        text_feat = self.prompt_learner(mode=mode, omega=omega)  # (C, D_proj)

        # Both tensors are unit-norm; the following is cosine sim.
        logits = slide_norm @ text_feat.t()

        if return_text_features:
            return logits, text_feat
        return logits
