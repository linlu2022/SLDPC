"""Backbone adapters for SLDPC slide-level foundation models.

Currently only the TITAN backbone is supported in this release.

Use :func:`sldpc.backbones.get_backbone` as the preferred entry
point for loading a backbone + its prompt learner.
"""

from .registry import get_backbone

__all__ = ["get_backbone"]
