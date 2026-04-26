"""TITAN-specific prompt learner.

Concrete subclass of :class:`sldpc.core.prompt_learner_base.PromptLearnerBase`
that implements the CLIP-style ``[BOS] [CTX_1..CTX_M] [CLASS_TOKENS] [EOS]``
prompt assembly used by TITAN. Prompt tokenization is performed once at
``__init__`` time and stored as non-persistent buffers so that the module
is cheap to instantiate and serializes only the learnable parameters.

The fusion / two-stage protocol (``mode="train"|"base"|"task"|"fused"``)
is inherited from the base class. The only backbone-specific behaviour
implemented here is :meth:`_encode_to_text_features`, which feeds the
context-injected prompt sequence through TITAN's text encoder via
:func:`sldpc.backbones.titan.text_encoding.encode_text`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from ...core.prompt_learner_base import PromptLearnerBase
from .text_encoding import encode_text, resolve_eos_token_id


__all__ = ["TitanPromptLearner"]


def _get_token_attr(tokenizer: Any, name: str, default: Optional[int] = None) -> Optional[int]:
    """Read a tokenizer attribute from either wrapper or wrapped tokenizer."""
    val = getattr(tokenizer, name, None)
    if val is not None:
        return int(val)
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        val = getattr(inner, name, None)
        if val is not None:
            return int(val)
    return default


def _tokenize_input_ids(tokenizer: Any, texts: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Tokenize text(s) and always return ``(input_ids, attention_mask?)``.

    Supports:
    - HuggingFace tokenizers (dict output, supports ``padding``).
    - TITAN's Conch_Tokenizer wrapper (tensor output).
    """
    try:
        out = tokenizer(texts, padding=True, return_tensors="pt")
        if isinstance(out, dict) and "input_ids" in out:
            return out["input_ids"], out.get("attention_mask")
    except TypeError:
        pass

    out = tokenizer(texts)
    if isinstance(out, dict) and "input_ids" in out:
        return out["input_ids"], out.get("attention_mask")
    if isinstance(out, torch.Tensor):
        pad_id = _get_token_attr(tokenizer, "pad_token_id", default=0)
        attn = (out != int(pad_id)).long()
        return out, attn
    raise TypeError(f"Unsupported tokenizer output type: {type(out)!r}")


def _safe_token_id(tokenizer: Any, *names: str, fallback: int = 0) -> int:
    """Return the first defined token-id attribute among ``names``."""
    for n in names:
        tid = _get_token_attr(tokenizer, n, default=None)
        if tid is not None:
            return int(tid)
    return fallback


class TitanPromptLearner(PromptLearnerBase):
    """CLIP-style learnable context prompt for TITAN.

    Parameters
    ----------
    classnames : list[str]
        Ordered class names.
    titan : nn.Module
        A loaded TITAN model (see :func:`encode_text` for the required
        attribute set). Used at init time to access the tokenizer,
        token embedding, and to read ``ctx_dim`` / ``dtype``.
    n_ctx : int, optional
        Number of learnable context tokens. Default 8 (paper).
    ctx_init : str, optional
        Initializer phrase (e.g. ``"a_photo_of"``). Underscores are
        replaced by spaces; the resulting sequence is tokenized and
        its non-special token embeddings become the initial
        ``ctx_learnable`` values. If ``None`` (default), use random
        normal init with std 0.02.
    csc : bool, optional
        Class-specific context. Default ``False`` (paper uses CUC).
    class_token_position : {"end", "middle", "front"}, optional
        Where to insert the context tokens relative to the class name.
        Default ``"end"``.
    omega : float, optional
        Default WFM fusion weight. Default 0.8 (paper optimum).
    """

    def __init__(
        self,
        classnames: List[str],
        titan: nn.Module,
        n_ctx: int = 8,
        ctx_init: Optional[str] = None,
        csc: bool = False,
        class_token_position: str = "end",
        omega: float = 0.8,
    ) -> None:
        text_encoder = titan.text_encoder
        tokenizer = text_encoder.tokenizer
        dtype = text_encoder.ln_final.weight.dtype
        ctx_dim = int(text_encoder.ln_final.weight.shape[0])
        device = next(text_encoder.parameters()).device

        # Initialize base with random normal; we will overwrite below
        # if ctx_init is provided.
        super().__init__(
            classnames=classnames,
            n_ctx=n_ctx,
            ctx_dim=ctx_dim,
            csc=csc,
            omega=omega,
            ctx_init=ctx_init,
        )

        self.class_token_position = class_token_position
        if class_token_position == "middle":
            if self.n_ctx % 2 != 0:
                raise ValueError(
                    "class_token_position='middle' requires an even n_ctx."
                )

        # -------- (Re)initialize ctx from a seed phrase, if requested -----
        if ctx_init is not None:
            init_text = ctx_init.replace("_", " ")
            tok_init, _ = _tokenize_input_ids(tokenizer, [init_text])
            tok_init = tok_init.to(device)
            pad_id = _get_token_attr(tokenizer, "pad_token_id", default=0)
            actual_n_ctx = int(
                (tok_init != pad_id).sum().item() - 2    # minus BOS + EOS
            )
            if actual_n_ctx <= 0:
                raise ValueError(
                    f"ctx_init {init_text!r} tokenizes to <= 2 tokens after removing BOS/EOS."
                )
            # Override n_ctx if the seed phrase length differs.
            self.n_ctx = actual_n_ctx
            with torch.no_grad():
                emb = text_encoder.token_embedding(tok_init).type(dtype)    # (1, L, D)
                # Skip BOS; take next actual_n_ctx tokens.
                seed_vec = emb[0, 1 : 1 + actual_n_ctx, :].clone()
                # Re-shape ctx parameters to match the new n_ctx.
                if csc:
                    new_shape = (self.n_cls, actual_n_ctx, ctx_dim)
                    seed_tiled = seed_vec.unsqueeze(0).expand(self.n_cls, -1, -1).contiguous()
                else:
                    new_shape = (actual_n_ctx, ctx_dim)
                    seed_tiled = seed_vec.contiguous()
                self.ctx_learnable = nn.Parameter(seed_tiled.clone().to(dtype=dtype))
                self.ctx_frozen = nn.Parameter(
                    seed_tiled.clone().to(dtype=dtype), requires_grad=False
                )
                del new_shape  # not needed; shape is inherent to the tensor
        else:
            # Cast random-init parameters to the backbone's dtype.
            with torch.no_grad():
                self.ctx_learnable.data = self.ctx_learnable.data.to(device=device, dtype=dtype)
                self.ctx_frozen.data = self.ctx_frozen.data.to(device=device, dtype=dtype)

        # ------------------- Tokenize classnames --------------------------
        classnames_clean = [name.replace("_", " ") for name in classnames]
        token_ids_cls, attention_mask = _tokenize_input_ids(tokenizer, classnames_clean)
        token_ids_cls = token_ids_cls.to(device)
        if attention_mask is None:
            pad_id = _get_token_attr(tokenizer, "pad_token_id", default=0)
            attention_mask = (token_ids_cls != int(pad_id)).long()
        attention_mask = attention_mask.to(device)
        # Actual class-name token count excludes BOS + EOS.
        name_lens = (attention_mask.sum(dim=-1) - 2).tolist()

        max_name_len = int(max(name_lens))
        total_len = 1 + self.n_ctx + max_name_len + 1
        model_max = int(_get_token_attr(tokenizer, "model_max_length", default=10**6))
        if total_len > model_max:
            raise ValueError(
                f"Prompt length {total_len} exceeds tokenizer.model_max_length "
                f"{model_max}. Reduce n_ctx or use shorter classnames."
            )

        # Build placeholder input_ids [BOS | PAD*n_ctx | class_tokens | EOS | PAD*].
        pad_id = _get_token_attr(tokenizer, "pad_token_id", default=0)
        input_ids = torch.full(
            (self.n_cls, total_len), pad_id, device=device, dtype=torch.long
        )
        bos_id = _safe_token_id(
            tokenizer, "cls_token_id", "bos_token_id", "pad_token_id", fallback=0
        )
        eos_id = resolve_eos_token_id(tokenizer)
        input_ids[:, 0] = bos_id
        for i, name_id_seq in enumerate(token_ids_cls):
            valid = name_lens[i]
            input_ids[i, 1 + self.n_ctx : 1 + self.n_ctx + valid] = name_id_seq[1 : 1 + valid]
        input_ids[:, 1 + self.n_ctx + max_name_len] = eos_id

        with torch.no_grad():
            embedding = text_encoder.token_embedding(input_ids).type(dtype)     # (C, L, D)

        # Static buffers: prefix = [BOS], suffix = [CLASS_TOKENS + EOS + padding].
        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix",
            embedding[:, 1 + self.n_ctx : 1 + self.n_ctx + max_name_len + 1, :],
            persistent=False,
        )
        self.register_buffer("tokenized_prompts", input_ids, persistent=False)
        self.name_lens = list(name_lens)

        # Hold a non-owning reference to TITAN for the text encoding call.
        # We explicitly do NOT register ``titan`` as a submodule so that
        # optimizers iterating over ``self.parameters()`` see only
        # ``ctx_learnable``.
        object.__setattr__(self, "_titan", titan)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _broadcast_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        """Expand shared context to per-class context if CUC."""
        if ctx.dim() == 2:    # (n_ctx, D)
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return ctx

    def _assemble_prompts(self, ctx: torch.Tensor) -> torch.Tensor:
        """Inject ``ctx`` into ``[BOS | CTX | CLASS | EOS]`` at the chosen position."""
        ctx = self._broadcast_ctx(ctx)
        prefix = self.token_prefix    # (C, 1, D)
        suffix = self.token_suffix    # (C, name_len_max + 1, D)

        if self.class_token_position == "end":
            return torch.cat([prefix, ctx, suffix], dim=1)

        if self.class_token_position == "middle":
            half = self.n_ctx // 2
            parts = []
            for i in range(self.n_cls):
                pfx = prefix[i : i + 1]
                cls_tok = suffix[i : i + 1, : self.name_lens[i], :]
                suf_tok = suffix[i : i + 1, self.name_lens[i] :, :]
                parts.append(
                    torch.cat(
                        [pfx, ctx[i : i + 1, :half, :], cls_tok, ctx[i : i + 1, half:, :], suf_tok],
                        dim=1,
                    )
                )
            return torch.cat(parts, dim=0)

        if self.class_token_position == "front":
            parts = []
            for i in range(self.n_cls):
                pfx = prefix[i : i + 1]
                cls_tok = suffix[i : i + 1, : self.name_lens[i], :]
                suf_tok = suffix[i : i + 1, self.name_lens[i] :, :]
                parts.append(torch.cat([pfx, cls_tok, ctx[i : i + 1], suf_tok], dim=1))
            return torch.cat(parts, dim=0)

        raise ValueError(f"Unknown class_token_position: {self.class_token_position!r}.")

    # ------------------------------------------------------------------
    # Base-class hook
    # ------------------------------------------------------------------

    def _encode_to_text_features(self, ctx: torch.Tensor) -> torch.Tensor:
        """Assemble prompts, run through TITAN text encoder, return (C, D_proj)."""
        prompt_emb = self._assemble_prompts(ctx)
        device = prompt_emb.device
        return encode_text(
            titan=self._titan,
            prompt_emb=prompt_emb,
            token_ids=self.tokenized_prompts,
            device=device,
        )
