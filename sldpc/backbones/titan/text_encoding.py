"""TITAN text encoding — context embeddings → normalized text features.

This module provides a single function :func:`encode_text` that takes
a ``(C, L, D)`` assembled-prompt embedding tensor and returns the
``(C, D_proj)`` normalized class-text features used for similarity
scoring with slide features.

The implementation mirrors TITAN's CLIP-style text pathway:

1. Add positional embedding (first ``L`` positions).
2. Pass through ``text_encoder.transformer``.
3. ``ln_final``.
4. Pick the ``<EOT>`` token position per row (looked up via the
   tokenizer's ``eos_token_id`` / ``sep_token_id``).
5. Linear projection to the shared vision-language space.
6. L2 normalization.

Historical pitfalls handled here
--------------------------------

- **Positional embedding must be added manually** when prompts are
  assembled from ``token_embedding`` outputs. Skipping this silently
  destroys performance because the transformer sees identical position
  signals for every token.
- **Batch-first, not sequence-first.** TITAN / HuggingFace transformers
  take ``(B, L, D)``; do **not** ``permute(1, 0, 2)``.
- **EOT lookup must go through the tokenizer's ``eos_token_id``**
  rather than ``argmax(token_ids)``, which can latch onto pad tokens
  in non-CLIP vocabularies.
- **``text_projection`` may be either an ``nn.Linear`` or a raw
  ``nn.Parameter`` matrix** depending on the TITAN release — handle
  both call patterns.
- **dtype pinning**: ``text_encoder.ln_final.weight.dtype`` is the
  authoritative dtype for text-side math; avoid forcing fp32 when the
  backbone was loaded in fp16.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["encode_text", "resolve_eos_token_id"]


def resolve_eos_token_id(tokenizer: Any) -> int:
    """Return the tokenizer's end-of-text token id with a safe fallback.

    Order of preference: ``eos_token_id`` → ``sep_token_id`` → ``0``.
    """
    inner = getattr(tokenizer, "tokenizer", None)
    for attr in ("eos_token_id", "sep_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            return int(tid)
        if inner is not None:
            tid = getattr(inner, attr, None)
            if tid is not None:
                return int(tid)
    return 0


def encode_text(
    titan: nn.Module,
    prompt_emb: torch.Tensor,
    token_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Encode assembled prompt embeddings into normalized text features.

    Parameters
    ----------
    titan : nn.Module
        A loaded TITAN model (the object returned by
        ``AutoModel.from_pretrained("MahmoodLab/TITAN", ...)``). Must
        expose ``text_encoder`` with the attributes
        ``positional_embedding``, ``transformer``, ``ln_final``,
        ``text_projection``, and ``tokenizer``.
    prompt_emb : torch.Tensor
        Assembled token embeddings of shape ``(C, L, D)``, typically
        the output of :meth:`TitanPromptLearner._assemble_prompts`.
    token_ids : torch.Tensor
        Tokenized prompt ids of shape ``(C, L)``; used only to locate
        the ``<EOT>`` position per row. Produced by the prompt
        learner's ``tokenized_prompts`` buffer.
    device : torch.device
        Device on which the forward pass runs.

    Returns
    -------
    torch.Tensor
        Shape ``(C, D_proj)``. L2-normalized class-text features in
        the shared vision-language space.
    """
    dtype = titan.text_encoder.ln_final.weight.dtype
    prompt_emb = prompt_emb.to(device=device, dtype=dtype)

    # 1. Positional embedding.
    L = prompt_emb.size(1)
    pos = titan.text_encoder.positional_embedding[:L].to(device=device, dtype=dtype)
    x = prompt_emb + pos.unsqueeze(0)     # (C, L, D)

    # 2. Transformer + LayerNorm.
    x = titan.text_encoder.transformer(x)
    x = titan.text_encoder.ln_final(x)    # (C, L, D)

    # 3. Locate <EOT> per row via the tokenizer.
    eos_id = resolve_eos_token_id(titan.text_encoder.tokenizer)
    idx = (token_ids.to(device) == eos_id).float().argmax(dim=-1)    # (C,)
    txt_feat = x[torch.arange(x.size(0), device=device), idx]        # (C, D)

    # 4. Projection: nn.Linear or raw parameter matrix.
    proj = titan.text_encoder.text_projection
    if isinstance(proj, nn.Linear):
        txt_feat = proj(txt_feat)
    else:
        txt_feat = txt_feat @ proj

    # 5. L2 normalize.
    return F.normalize(txt_feat, dim=-1)
