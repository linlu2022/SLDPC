"""Stage-2 trainer: Dynamic Hard-Negative Sampling + Symmetric InfoNCE.

Stage-2 trains the task-specific prompt ``P'`` with the symmetric
InfoNCE loss (Eq. 11) over an extended batch that augments each
positive with ``K-1`` hard negatives sampled from a pre-computed
feature bank. The frozen base prompt ``P`` (established at the end of
Stage-1) is used to score candidate hard-negative classes and to
provide fallback text features for validation in ``mode="fused"``.

Prerequisites
-------------

Before constructing this trainer the caller must have:

1. Trained Stage-1 and restored the best checkpoint into the
   prompt learner (``Stage1Trainer.load_best_into_model()``).
2. Called ``prompt_learner.clone_learnable_to_frozen()`` so that
   ``ctx_frozen`` holds ``P`` and ``ctx_learnable`` is a clone of it
   ready to become ``P'`` (CPI cloning, paper Eq. 8).
3. Built the slide-level feature bank ``(N, D_proj)`` from the
   training set, along with a ``class_to_indices`` lookup — the
   helper :func:`build_feature_bank` does both.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core.fusion import l2norm
from ..core.losses import (
    classification_ce_loss,
    info_nce_image_to_text,
    info_nce_text_to_image,
    symmetric_info_nce,
)
from ..core.negative_sampler import (
    build_class_index,
    dynamic_hard_negative_sampler,
    random_negative_sampler,
)
from .base_trainer import BaseTrainer, TrainingConfig


__all__ = ["Stage2Trainer", "build_feature_bank"]


# -----------------------------------------------------------------------------
# Feature-bank construction
# -----------------------------------------------------------------------------

@torch.no_grad()
def build_feature_bank(
    model: nn.Module,
    feature_dataset,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
    """Pre-compute the projected & normalized slide feature bank.

    Uses :meth:`PromptedTitan.project_slide` so that the bank lives
    in the same L2-unit shared embedding space as the text features.

    Parameters
    ----------
    model : nn.Module
        Wrapped backbone exposing ``project_slide``.
    feature_dataset : SlideFeatureDataset
        The training set over which we sample hard negatives.
    device : torch.device

    Returns
    -------
    feat_bank : torch.Tensor
        ``(N, D_proj)``, L2-normalized.
    label_bank : torch.Tensor
        ``(N,)`` long tensor of ground-truth class indices.
    class_to_indices : dict[int, torch.Tensor]
        ``{class_id: sample_indices_in_feat_bank}``.
    """
    raw = feature_dataset.stack_features().to(device)    # (N, D_slide)
    feat_bank = model.project_slide(raw)                  # (N, D_proj)
    label_bank = feature_dataset.labels.to(device)        # (N,)
    class_to_indices = build_class_index(label_bank, device=device)
    return feat_bank, label_bank, class_to_indices


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class Stage2Trainer(BaseTrainer):
    """DHNO + symmetric InfoNCE training of ``P'``.

    Additional arguments beyond :class:`BaseTrainer`:

    Parameters
    ----------
    feature_bank : torch.Tensor
        ``(N, D_proj)`` L2-normalized slide features for hard-negative sampling.
    label_bank : torch.Tensor
        ``(N,)`` labels aligned with ``feature_bank``.
    class_to_indices : dict[int, Tensor]
        Output of :func:`build_class_index`.
    rng_seed : int, optional
        Seed for the per-step ``torch.Generator``. Default 0.
    eval_mode : {"fused", "task", "base"}
        Which prompt branch to use at validation time. The paper's
        main numbers come from ``"fused"`` with the trained ``omega``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        cfg: TrainingConfig,
        device: torch.device,
        feature_bank: torch.Tensor,
        label_bank: torch.Tensor,
        class_to_indices: Dict[int, torch.Tensor],
        rng_seed: int = 0,
        eval_mode: str = "fused",
    ) -> None:
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            cfg=cfg,
            device=device,
        )
        self.feature_bank = feature_bank.to(device)
        self.label_bank = label_bank.to(device)
        self.class_to_indices = {int(k): v.to(device) for k, v in class_to_indices.items()}

        # Eligible negative classes — in standard few-shot eval this is all-True.
        self.base_class_mask = torch.ones(
            num_classes, dtype=torch.bool, device=device
        )

        # Deterministic RNG for Stage-2 sampling.
        try:
            self.rng = torch.Generator(device=device).manual_seed(rng_seed)
        except RuntimeError:
            # Some CUDA / PyTorch combinations disallow device-specific generators;
            # fall back to CPU-side generator and let the sampler's random fallback
            # handle per-call draws.
            self.rng = None

        self.eval_mode = eval_mode

        # Cache the frozen text features produced by P. These do not
        # change during Stage-2 because ``ctx_frozen.requires_grad =
        # False``; caching avoids a forward through the text encoder
        # for every batch.
        with torch.no_grad():
            self._text_feat_frozen = self.model.prompt_learner(mode="base")  # (C, D_proj)

    # ------------------------------------------------------------------
    # Per-batch training
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any], epoch: int) -> torch.Tensor:
        feats = batch["feat"].to(self.device)
        labels = batch["label"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        labels = labels.to(self.device)

        # 1. Project mini-batch slides and fetch current P' text features.
        slide_norm = self.model.project_slide(feats)                       # (B, D_proj)
        text_feat_train = self.model.prompt_learner(mode="train")          # (C, D_proj)

        dhno_mode = str(self.cfg.dhno_mode).lower()
        sampling_mode = str(self.cfg.negative_sampling).lower()
        loss_mode = str(self.cfg.stage2_loss).lower()

        if dhno_mode not in {"full", "sampling_only", "none"}:
            raise ValueError(
                f"Unsupported dhno_mode={self.cfg.dhno_mode!r}. "
                "Expected one of: full, sampling_only, none."
            )
        if sampling_mode not in {"hns", "random"}:
            raise ValueError(
                f"Unsupported negative_sampling={self.cfg.negative_sampling!r}. "
                "Expected one of: hns, random."
            )
        if loss_mode not in {"symmetric", "i2t", "t2i", "ce"}:
            raise ValueError(
                f"Unsupported stage2_loss={self.cfg.stage2_loss!r}. "
                "Expected one of: symmetric, i2t, t2i, ce."
            )

        # 2. Build training pairs for stage-2.
        if dhno_mode == "none":
            feat_ext = slide_norm
            label_ext = labels
        else:
            sampler = (
                dynamic_hard_negative_sampler
                if sampling_mode == "hns"
                else random_negative_sampler
            )
            if sampler is dynamic_hard_negative_sampler:
                feat_ext, label_ext = sampler(
                    img_feat_batch=slide_norm,
                    label_batch=labels,
                    text_feat_frozen=self._text_feat_frozen,
                    feature_bank=self.feature_bank,
                    class_to_indices=self.class_to_indices,
                    base_class_mask=self.base_class_mask,
                    k=self.cfg.topk,
                    rng=self.rng,
                )
            else:
                feat_ext, label_ext = sampler(
                    img_feat_batch=slide_norm,
                    label_batch=labels,
                    feature_bank=self.feature_bank,
                    class_to_indices=self.class_to_indices,
                    base_class_mask=self.base_class_mask,
                    k=self.cfg.topk,
                    rng=self.rng,
                )

        # 3. Loss family selection for ablations.
        if loss_mode == "ce":
            # CE ablation: matches old code's dhno_mode='none' and
            # 'sampling_only' paths which use raw cosine logits (no tau)
            # in cross-entropy. Symmetric/directional InfoNCE variants
            # below still divide by tau (paper Eq. 11, matching the old
            # info_nce_loss implementation).
            return classification_ce_loss(
                img_feat=feat_ext,
                txt_bank=text_feat_train,
                labels=label_ext,
                tau=self.cfg.tau,
                apply_tau=self.cfg.stage2_ce_apply_tau,
            )

        txt_feat_ext = text_feat_train[label_ext]
        feat_ext = feat_ext.detach()
        if loss_mode == "symmetric":
            return symmetric_info_nce(
                img_feat=feat_ext,
                txt_feat=txt_feat_ext,
                tau=self.cfg.tau,
            )
        if loss_mode == "i2t":
            return info_nce_image_to_text(
                img_feat=feat_ext,
                txt_feat=txt_feat_ext,
                tau=self.cfg.tau,
            )
        return info_nce_text_to_image(
            img_feat=feat_ext,
            txt_feat=txt_feat_ext,
            tau=self.cfg.tau,
        )

    # ------------------------------------------------------------------
    # Validation: use the fused prompt by default
    # ------------------------------------------------------------------

    def _logits_for_eval(self, feats: torch.Tensor) -> torch.Tensor:
        return self.model(feats, mode=self.eval_mode)
