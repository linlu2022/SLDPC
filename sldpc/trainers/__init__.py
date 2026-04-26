"""Trainer package.

Modules are intentionally loaded lazily so that CLI `--help` and dry-run
paths can work even when heavyweight ML dependencies are not installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaseTrainer",
    "TrainingConfig",
    "Stage1Trainer",
    "Stage2Trainer",
    "build_feature_bank",
]


def __getattr__(name: str) -> Any:
    if name in {"BaseTrainer", "TrainingConfig"}:
        mod = import_module("sldpc.trainers.base_trainer")
        return getattr(mod, name)
    if name == "Stage1Trainer":
        mod = import_module("sldpc.trainers.stage1_trainer")
        return getattr(mod, name)
    if name in {"Stage2Trainer", "build_feature_bank"}:
        mod = import_module("sldpc.trainers.stage2_trainer")
        return getattr(mod, name)
    raise AttributeError(name)
