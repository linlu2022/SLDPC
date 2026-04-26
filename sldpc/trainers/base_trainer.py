"""Common infrastructure for SLDPC trainers.

Both Stage-1 (CPI via cross-entropy) and Stage-2 (DHNO + symmetric
InfoNCE) share the same surrounding scaffolding: checkpoint I/O,
per-epoch validation, early stopping, TensorBoard logging, optimizer
construction. This module centralizes that scaffolding so that the
actual stage trainers only need to implement the per-batch ``train_step``.
"""

from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


__all__ = ["TrainingConfig", "BaseTrainer"]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Minimal hyperparameter bundle shared by Stage-1 and Stage-2.

    Defaults reflect the **legacy SLDPC-TITAN behavior** (the actual
    implementation that produced the paper's reported numbers), not the
    literal text of paper §4.1. Specifically:

    - ``lr=1e-3``: matches ``train.py`` and ``trainer_dpc.py`` CLI defaults.
      Paper §4.1 says ``1e-4``, but the old code's CLI defaulted to and
      typically ran with ``1e-3``.
    - ``weight_decay=0.0``: old code did not pass ``weight_decay`` to AdamW,
      so the effective wd was PyTorch's AdamW default (``0.01``). Setting
      ``0.0`` here gives a cleaner baseline; set ``0.01`` for bit-level
      legacy parity.
    - ``patience=100``: old code's early-stop patience was effectively
      disabled (100 epochs with 50 total epochs = never triggers).
    - ``stage1_apply_tau=False``: old Stage-1 CE uses raw cosine
      similarity as logits, not cosine / tau.
    - ``stage2_ce_apply_tau=False``: old Stage-2 CE ablations
      (``dhno_mode="none"``/``"sampling_only"``) also use raw cosine.
    - ``eval_apply_tau_in_softmax=False``: old eval uses softmax(cos)
      for probabilities (AUC inputs), not softmax(cos / tau).

    Symmetric InfoNCE always divides by tau — this is paper Eq. 11 and
    matches the old ``info_nce_loss`` implementation.
    """

    # Optimization
    lr: float = 1e-3                       # aligned with old CLI default
    weight_decay: float = 0.0              # see class docstring
    epochs: int = 50
    batch_size: int = 4
    patience: int = 100                    # aligned with old es_patience

    # Loss temperature (paper fixes at 0.07)
    tau: float = 0.07

    # Metric used for best-model selection & early stopping
    monitor_metric: str = "F1"             # one of "ACC", "F1", "AUC"
    monitor_mode: str = "max"              # "max" or "min"

    # I/O
    output_dir: str = "./runs/sldpc_stage"
    save_every_epoch: bool = False         # if True, keep per-epoch ckpt
    log_every_n_batches: int = 10

    # Stage-2 specific (ignored by Stage-1)
    topk: int = 8                          # number of Top-K hard negatives
    dhno_mode: str = "full"                # "full" | "sampling_only" | "none"
    negative_sampling: str = "hns"         # "hns" | "random"
    stage2_loss: str = "symmetric"         # "symmetric" | "i2t" | "t2i" | "ce"

    # ---- Behavior toggles: opt-in to modern tau-scaled numerics ----
    # Stage-1 CE: False=legacy (logits=cos), True=paper (logits=cos/tau).
    stage1_apply_tau: bool = False
    # Stage-2 CE ablations: False=legacy, True=paper Eq. 7 form.
    stage2_ce_apply_tau: bool = False
    # Softmax on eval (only affects AUC's probability inputs; ACC/F1
    # use argmax and are unaffected regardless).
    eval_apply_tau_in_softmax: bool = False

    # ---- Optional LR scheduler (old code used ReduceLROnPlateau) ----
    use_lr_scheduler: bool = False
    scheduler_factor: float = 0.1
    scheduler_patience: int = 8

    # Extra free-form fields for experiment reproducibility
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Trainer base
# -----------------------------------------------------------------------------

class BaseTrainer(ABC):
    """Abstract scaffold for the two SLDPC training stages.

    Subclasses implement :meth:`train_step` (what happens per batch)
    and optionally override :meth:`build_optimizer` (default: AdamW on
    ``model.prompt_learner.ctx_learnable`` only).

    Parameters
    ----------
    model : nn.Module
        The wrapped backbone (e.g. :class:`PromptedTitan`). Must
        expose a ``prompt_learner`` submodule with a
        ``ctx_learnable`` parameter.
    train_loader : DataLoader
        Yields dicts with ``"feat"``, ``"label"``, ``"slide_id"``.
    val_loader : DataLoader
        Same interface as ``train_loader``.
    num_classes : int
        Number of target classes.
    cfg : TrainingConfig
    device : torch.device
    compute_metrics : callable, optional
        ``(y_true, y_prob, num_classes) -> dict``. Defaults to
        :func:`sldpc.utils.metrics.compute_classification_metrics`.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        cfg: TrainingConfig,
        device: torch.device,
        compute_metrics: Optional[Callable[..., Dict[str, float]]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = device

        if compute_metrics is None:
            from ..utils.metrics import compute_classification_metrics
            compute_metrics = compute_classification_metrics
        self.compute_metrics = compute_metrics

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = self.build_optimizer()
        self.scheduler = self._maybe_build_scheduler()
        self.writer = self._maybe_build_writer()

        # Early-stopping state.
        self._best_score = -math.inf if cfg.monitor_mode == "max" else math.inf
        self._best_metrics: Dict[str, float] = {}
        self._epochs_since_improvement = 0
        self._best_ckpt_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Optimizer / logging
    # ------------------------------------------------------------------

    def build_optimizer(self) -> Optimizer:
        """Default: AdamW on the single ``ctx_learnable`` parameter.

        Subclasses can override to e.g. include additional parameters.
        """
        pl = self.model.prompt_learner
        params = [pl.ctx_learnable]
        return AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def _maybe_build_writer(self):
        """Construct a TensorBoard writer if tensorboard is importable."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning("tensorboard not available; skipping TB logging.")
            return None
        return SummaryWriter(log_dir=str(self.output_dir / "runs"))

    def _maybe_build_scheduler(self):
        """Optionally build a ReduceLROnPlateau scheduler.

        Enabled by ``cfg.use_lr_scheduler = True``. Mirrors the old
        SLDPC-TITAN setup (``mode='max', factor=0.1, patience=8``).
        """
        if not getattr(self.cfg, "use_lr_scheduler", False):
            return None
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        mode = "max" if self.cfg.monitor_mode == "max" else "min"
        return ReduceLROnPlateau(
            self.optimizer,
            mode=mode,
            factor=float(getattr(self.cfg, "scheduler_factor", 0.1)),
            patience=int(getattr(self.cfg, "scheduler_patience", 8)),
        )

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, float]:
        """Run training for ``cfg.epochs`` with early stopping.

        Returns
        -------
        dict
            The best-epoch validation metrics, matching whatever keys
            :attr:`compute_metrics` produces.
        """
        logger.info("Starting %s for %d epochs.", type(self).__name__, self.cfg.epochs)
        try:
            from sldpc.utils.run_logging import print_config_block
            print_config_block(logger, f"{type(self).__name__} TrainingConfig", self.cfg.to_dict())
        except Exception:
            # Fall back to the old one-line dump if run_logging is missing.
            logger.info("Config: %s", self.cfg.to_dict())

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            metrics = self._validate(epoch)

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                for k, v in metrics.items():
                    self.writer.add_scalar(f"Metrics/val/{k}", v, epoch)

            self._log_epoch(epoch, train_loss, metrics)

            # Best-model tracking.
            score = metrics.get(self.cfg.monitor_metric, float("nan"))
            improved = self._is_improvement(score)
            if improved:
                self._best_score = score
                self._best_metrics = dict(metrics)
                self._epochs_since_improvement = 0
                self._best_ckpt_path = self._save_checkpoint(epoch, tag="best")
            else:
                self._epochs_since_improvement += 1

            # LR scheduler step (if enabled). ReduceLROnPlateau consumes
            # the current validation score.
            if self.scheduler is not None and not math.isnan(score):
                self.scheduler.step(score)

            if self.cfg.save_every_epoch:
                self._save_checkpoint(epoch, tag=f"epoch{epoch:03d}")

            if self._epochs_since_improvement >= self.cfg.patience:
                logger.info(
                    "Early stop at epoch %d (no improvement in %d epochs).",
                    epoch, self.cfg.patience,
                )
                break

        self._write_summary()
        if self.writer is not None:
            self.writer.close()
        return self._best_metrics

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch, epoch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            if batch_idx % self.cfg.log_every_n_batches == 0:
                logger.debug(
                    "[ep %d | batch %d/%d] loss=%.4f",
                    epoch, batch_idx, len(self.train_loader), loss.item(),
                )

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        all_probs: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        tau = float(self.cfg.tau)
        for batch in self.val_loader:
            feats = batch["feat"].to(self.device)
            labels = batch["label"] if isinstance(batch["label"], torch.Tensor) \
                else torch.as_tensor(batch["label"])
            logits = self._logits_for_eval(feats)            # (B, C), raw cosine
            if self.cfg.eval_apply_tau_in_softmax:
                probs = torch.softmax(logits / tau, dim=-1)
            else:
                # Legacy behavior: softmax over raw cosine similarities.
                # ACC/F1 are unaffected (argmax is scale-invariant); AUC
                # ranking is also unaffected because scaling by a positive
                # constant preserves order.
                probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

        y_prob = torch.cat(all_probs, dim=0)
        y_true = torch.cat(all_labels, dim=0)
        return self.compute_metrics(y_true, y_prob, num_classes=self.num_classes)

    def _logits_for_eval(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute validation logits. Default uses the prompt-learner
        forward in ``"task"`` mode (Stage-1) or ``"fused"`` (Stage-2).
        Subclasses override as needed.
        """
        return self.model(feats, mode="task")

    # ------------------------------------------------------------------
    # Best-model I/O
    # ------------------------------------------------------------------

    def _is_improvement(self, score: float) -> bool:
        if math.isnan(score):
            return False
        if self.cfg.monitor_mode == "max":
            return score > self._best_score
        return score < self._best_score

    def _save_checkpoint(self, epoch: int, tag: str) -> Path:
        path = self.output_dir / f"ckpt_{tag}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "prompt_learner": self.model.prompt_learner.state_dict(),
            "cfg": self.cfg.to_dict(),
            "best_metrics": self._best_metrics,
        }
        # On Windows, torch.save raises ``RuntimeError: Parent directory
        # ... does not exist.`` when the path contains non-ASCII
        # characters (PyTorch issues #98918 / #103949). Passing an open
        # file handle bypasses PyTorch's internal path parsing and
        # works on every platform.
        with open(path, "wb") as f:
            torch.save(payload, f)
        return path

    def load_best_into_model(self) -> None:
        """Restore the best-checkpoint weights into ``model.prompt_learner``.

        Call this between Stage-1 and Stage-2, so Stage-2 starts from
        the best Stage-1 context rather than the final epoch's.
        """
        if self._best_ckpt_path is None or not self._best_ckpt_path.exists():
            logger.warning("No best checkpoint recorded; skipping restore.")
            return
        # File-handle form mirrors _save_checkpoint to avoid the same
        # Windows multi-byte-path issue on the load side.
        with open(self._best_ckpt_path, "rb") as f:
            payload = torch.load(f, map_location=self.device, weights_only=False)
        self.model.prompt_learner.load_state_dict(payload["prompt_learner"])

    def _write_summary(self) -> None:
        summary_path = self.output_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "best_metrics": self._best_metrics,
            "best_score": self._best_score,
            "best_checkpoint": str(self._best_ckpt_path) if self._best_ckpt_path else None,
            "cfg": self.cfg.to_dict(),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary written to %s.", summary_path)

    def _log_epoch(self, epoch: int, train_loss: float, metrics: Dict[str, float]) -> None:
        metric_str = " | ".join(f"{k}={v:.2f}" for k, v in metrics.items())
        logger.info("[ep %d] train_loss=%.4f | %s", epoch, train_loss, metric_str)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def train_step(self, batch: Dict[str, Any], epoch: int) -> torch.Tensor:
        """Compute the per-batch loss (scalar tensor, ``.backward()`` ready)."""
        raise NotImplementedError