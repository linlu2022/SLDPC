"""Abstract prompt learner defining the SLDPC interface.

This module defines :class:`PromptLearnerBase`, an abstract PyTorch module
that encapsulates the two prompt tensors (``P`` — frozen base, ``P'`` —
learnable task-specific), their fusion via WFM, and the two-stage
train/infer protocol.

Concrete subclasses — one per backbone — implement the single abstract
method :meth:`_encode_to_text_features`, which turns a ``(C, n_ctx, D)``
context tensor into the backbone's final ``(C, D_proj)`` class-text
embeddings. Everything else (WFM, CPI, shape bookkeeping, gradient
masking) lives in the base class and is shared.

Stage-1 vs Stage-2 protocol
---------------------------

- **Stage-1 (CPI)**: only ``ctx_learnable`` is optimized; ``ctx_frozen``
  is ignored. Call :meth:`forward` with ``mode="train"``.
- **Between stages**: the caller invokes :meth:`clone_learnable_to_frozen`
  to freeze the trained ``ctx_learnable`` as the base prompt, then
  re-initializes ``ctx_learnable`` from the same values (CPI cloning).
- **Stage-2**: ``ctx_frozen`` is held constant (no grad), while
  ``ctx_learnable`` is fine-tuned with DHNO + SICL. Queries under the
  frozen P use ``mode="base"``; the task-specific branch uses
  ``mode="task"``.
- **Inference**: :meth:`forward` with ``mode="fused"`` returns the WFM
  output ``omega * P' + (1 - omega) * P``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import torch
import torch.nn as nn

from .fusion import mix_ctx


__all__ = ["PromptLearnerBase"]


PromptMode = Literal["train", "base", "task", "fused"]


class PromptLearnerBase(nn.Module, ABC):
    """Backbone-agnostic prompt learner for SLDPC.

    The class owns two context tensors of shape ``(n_ctx, dim)`` or
    ``(n_cls, n_ctx, dim)`` depending on whether class-specific context
    (CSC) is enabled. Both live on the same device/dtype.

    Parameters
    ----------
    classnames : list[str]
        Ordered list of class names, used downstream by subclasses to
        build tokenized prompts.
    n_ctx : int, optional
        Number of learnable context tokens. Default 8 (matches paper).
    ctx_dim : int, optional
        Width of the text-encoder embedding space. Subclasses usually
        infer this from the backbone and pass it in.
    csc : bool, optional
        If ``True``, use a separate context per class (class-specific
        context). If ``False`` (default, matches paper Sec. 4.1), share
        one context across all classes (class-unified context).
    omega : float, optional
        Default fusion weight for ``mode="fused"``. Caller can override
        at forward time. Default 0.8 (paper's optimum).
    ctx_init : str, optional
        If given, initialize ``ctx_learnable`` from these tokens instead
        of random normal. Subclasses decide how to interpret this
        (typically tokenizer encoding).

    Attributes
    ----------
    ctx_frozen : nn.Parameter
        The base prompt ``P``. After Stage-1 cloning it has
        ``requires_grad_(False)``.
    ctx_learnable : nn.Parameter
        The task-specific prompt ``P'``. Always has gradient.
    """

    def __init__(
        self,
        classnames: List[str],
        n_ctx: int = 8,
        ctx_dim: int = 768,
        csc: bool = False,
        omega: float = 0.8,
        ctx_init: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.classnames = list(classnames)
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.csc = csc
        self.omega = omega
        self.ctx_init = ctx_init

        # Shape: (n_cls, n_ctx, ctx_dim) if CSC, else (n_ctx, ctx_dim).
        if csc:
            ctx_shape = (self.n_cls, n_ctx, ctx_dim)
        else:
            ctx_shape = (n_ctx, ctx_dim)

        # ctx_learnable starts random-initialized; subclass __init__ may
        # overwrite .data using a tokenized ctx_init prior to training.
        ctx_learnable = torch.empty(ctx_shape)
        nn.init.normal_(ctx_learnable, std=0.02)

        # ctx_frozen starts as a copy of ctx_learnable; it is replaced by
        # the Stage-1-trained prompt via clone_learnable_to_frozen().
        ctx_frozen = ctx_learnable.clone()

        self.ctx_learnable = nn.Parameter(ctx_learnable)                # requires_grad=True
        self.ctx_frozen = nn.Parameter(ctx_frozen, requires_grad=False)

    # -------------------------------------------------------------------
    # Two-stage protocol
    # -------------------------------------------------------------------

    @torch.no_grad()
    def clone_learnable_to_frozen(self) -> None:
        """Copy ``ctx_learnable`` into ``ctx_frozen`` (CPI cloning).

        Call this once at the end of Stage-1 training, before beginning
        Stage-2. After this point, ``ctx_frozen`` is treated as P and
        ``ctx_learnable`` continues to be fine-tuned as P'.
        """
        self.ctx_frozen.data.copy_(self.ctx_learnable.data)
        self.ctx_frozen.requires_grad_(False)

    @torch.no_grad()
    def reinit_learnable_from_frozen(self) -> None:
        """Reinitialize ``ctx_learnable`` from ``ctx_frozen``.

        Equivalent to starting Stage-2 with ``P' := P`` as described in
        paper Eq. 8. In most pipelines this is called together with (or
        immediately after) :meth:`clone_learnable_to_frozen`.
        """
        self.ctx_learnable.data.copy_(self.ctx_frozen.data)

    # -------------------------------------------------------------------
    # Core forward dispatch
    # -------------------------------------------------------------------

    def forward(
        self,
        mode: PromptMode = "train",
        omega: Optional[float] = None,
    ) -> torch.Tensor:
        """Produce ``(n_cls, D_proj)`` class-text features for the given mode.

        Parameters
        ----------
        mode : {"train", "base", "task", "fused"}, optional
            - ``"train"`` — Stage-1 training: use ``ctx_learnable`` with
              gradient. Aliased to ``"task"`` because the two are
              identical in this context; the separate name exists for
              clarity in call sites.
            - ``"base"`` — use frozen ``ctx_frozen`` (no gradient).
            - ``"task"`` — use ``ctx_learnable`` (with gradient).
            - ``"fused"`` — WFM: ``omega * P' + (1 - omega) * P``.
        omega : float, optional
            Overrides the default fusion weight when ``mode="fused"``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_cls, D_proj)``. ``D_proj`` is the final projection
            dimension of the backbone's text/vision shared space and is
            defined by the subclass.
        """
        if mode in ("train", "task"):
            ctx = self.ctx_learnable
        elif mode == "base":
            ctx = self.ctx_frozen
        elif mode == "fused":
            w = self.omega if omega is None else omega
            ctx = mix_ctx(self.ctx_frozen, self.ctx_learnable, omega=w)
        else:
            raise ValueError(
                f"Unknown prompt mode {mode!r}; expected one of "
                "'train', 'base', 'task', 'fused'."
            )

        return self._encode_to_text_features(ctx)

    # -------------------------------------------------------------------
    # Backbone-specific hook
    # -------------------------------------------------------------------

    @abstractmethod
    def _encode_to_text_features(self, ctx: torch.Tensor) -> torch.Tensor:
        """Backbone-specific encoding from context vectors to text features.

        Concrete subclasses implement this to bridge the abstract context
        tensor and the backbone's text encoder. Exact I/O:

        - Input ``ctx``: shape ``(n_ctx, dim)`` if ``csc=False``, else
          ``(n_cls, n_ctx, dim)``.
        - Output: shape ``(n_cls, D_proj)`` — L2-unnormalized class-text
          features in the backbone's shared embedding space.

        Typical subclass responsibilities:

        1. Assemble the tokenized prompts (e.g. ``[SOS] [CTX] [CLASS] [EOS]``).
        2. Replace the placeholder context embeddings with ``ctx``.
        3. Run the backbone's text encoder.
        4. Project to the shared vision-language space.

        Whether the returned features are L2-normalized is a
        backbone-dependent convention. Downstream losses in
        :mod:`sldpc.core.losses` apply L2 normalization internally, so
        either convention is safe — but **consistency** across the
        four forward modes (``train`` / ``base`` / ``task`` /
        ``fused``) is essential.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------
    # Convenience / introspection
    # -------------------------------------------------------------------

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters in the prompt learner.

        Useful for reporting parameter efficiency (cf. paper Table 1,
        which cites "8.2K trainable parameters").
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"n_cls={self.n_cls}, n_ctx={self.n_ctx}, ctx_dim={self.ctx_dim}, "
            f"csc={self.csc}, omega={self.omega}"
        )
