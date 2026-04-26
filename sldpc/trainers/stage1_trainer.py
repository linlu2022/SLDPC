"""Stage-1 trainer: Continuous Prompt Initialization via cross-entropy.

Stage-1 trains the base prompt ``P`` by minimizing a cross-entropy
objective on slide-level cosine similarities. Two numerical regimes
are supported via :attr:`TrainingConfig.stage1_apply_tau`:

* **Legacy (default, ``stage1_apply_tau=False``)** — the regime that
  produced the paper's reported numbers. Logits are the raw cosine
  similarities ``z @ t.T`` (no temperature scaling), and softmax operates
  directly on values in ``[-1, 1]``::

      L_CE = CE( softmax(z @ t.T), labels )

* **Paper-text (``stage1_apply_tau=True``)** — strict matching of paper
  Eq. 7. Logits are scaled by ``1/tau`` before softmax::

      L_CE = CE( softmax(z @ t.T / tau), labels )

In both cases ``z`` is the L2-normalized, vision-projected slide feature
and ``t`` is the L2-normalized text feature produced by the current
``ctx_learnable``. Only ``ctx_learnable`` receives gradients; the
underlying backbone is frozen.

At the end of Stage-1 the caller should:

1. Invoke :meth:`Stage1Trainer.load_best_into_model` to restore the
   best-epoch ``ctx_learnable`` values (not the final epoch's).
2. Call :meth:`PromptLearnerBase.clone_learnable_to_frozen` on the
   prompt learner to copy ``ctx_learnable`` into ``ctx_frozen``
   (CPI cloning, Eq. 8). After this, Stage-2 can begin with P' = P.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..core.losses import classification_ce_loss
from .base_trainer import BaseTrainer


__all__ = ["Stage1Trainer"]


class Stage1Trainer(BaseTrainer):
    """Cross-entropy training of the base prompt."""

    def train_step(self, batch: Dict[str, Any], epoch: int) -> torch.Tensor:
        feats = batch["feat"].to(self.device)
        labels = batch["label"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        labels = labels.to(self.device)

        # Project slides into the shared space, fetch text features
        # from the current learnable prompt. We call project_slide
        # directly rather than self.model(feat, mode="task") because
        # we need the projected features separately for the CE loss.
        slide_norm = self.model.project_slide(feats)                    # (B, D_proj)
        text_feat = self.model.prompt_learner(mode="train")             # (C, D_proj)

        # ``apply_tau`` toggles between legacy (cosine logits) and paper
        # (cosine / tau) forms; default False preserves the original
        # SLDPC-TITAN numerics.
        loss = classification_ce_loss(
            img_feat=slide_norm,
            txt_bank=text_feat,
            labels=labels,
            tau=self.cfg.tau,
            apply_tau=self.cfg.stage1_apply_tau,
        )
        return loss

    def _logits_for_eval(self, feats: torch.Tensor) -> torch.Tensor:
        """Stage-1 validation uses the ``"task"`` branch (there is no frozen P yet)."""
        return self.model(feats, mode="task")
