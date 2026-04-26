"""SLDPC — Slide-Level Dual-Prompt Collaboration.

Official implementation of:

    "SLDPC: Slide-Level Dual-Prompt Collaboration for Few-Shot Whole
     Slide Image Classification"

This top-level package re-exports the main public API. The typical
usage is:

    from sldpc.backbones import get_backbone
    from sldpc.backbones.titan import PromptedTitan
    from sldpc.core import (
        dynamic_hard_negative_sampler,
        hard_negative_contrastive_loss,
        classification_ce_loss,
        mix_ctx,
    )
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
