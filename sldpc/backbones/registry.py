"""Backbone factory.

Single entry point for loading a slide-level foundation model plus
its SLDPC-compatible prompt learner. Callers should prefer
:func:`get_backbone` over constructing ``PromptedTitan`` directly — it
centralizes model loading (HuggingFace Hub, local cache, etc.) and
keeps the dependency on ``transformers`` isolated to this module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


__all__ = ["get_backbone"]


def _load_titan(
    hf_model_id: str = "MahmoodLab/TITAN",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **hf_kwargs: Any,
):
    """Load a TITAN model from HuggingFace Hub.

    Users must have accepted the model's license agreement on
    https://huggingface.co/MahmoodLab/TITAN and be authenticated via
    ``huggingface-cli login`` before this call. The weights are NOT
    redistributed with this repository — see the License section of
    ``README.md``.
    """
    try:
        from transformers import AutoModel
    except ImportError as e:
        raise ImportError(
            "transformers is required to load TITAN. "
            "Install via `pip install transformers`."
        ) from e

    model = AutoModel.from_pretrained(
        hf_model_id,
        trust_remote_code=True,
        **hf_kwargs,
    )
    if dtype is not None:
        model = model.to(dtype=dtype)
    if device is not None:
        model = model.to(device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_backbone(
    name: str,
    classnames: List[str],
    device: Optional[torch.device] = None,
    prompt_kwargs: Optional[Dict[str, Any]] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple["torch.nn.Module", "torch.nn.Module"]:
    """Load a backbone + matching prompt learner.

    Parameters
    ----------
    name : {"titan"}
        Backbone identifier. Only ``"titan"`` is supported in this
        release.
    classnames : list[str]
        Ordered list of class names for the downstream task.
    device : torch.device, optional
        Target device. Defaults to ``"cuda"`` if available else CPU.
    prompt_kwargs : dict, optional
        Passed through to the prompt-learner constructor.
        For TITAN the accepted keys are ``n_ctx``, ``ctx_init``,
        ``csc``, ``class_token_position``, ``omega``.
    load_kwargs : dict, optional
        Passed through to the backbone loader. For TITAN the accepted
        keys are ``hf_model_id``, ``dtype``, plus any extras forwarded
        to ``AutoModel.from_pretrained``.

    Returns
    -------
    (model, prompt_learner)
        ``model`` is the underlying backbone module (frozen). The
        caller typically wraps it in the backbone-specific model
        wrapper (e.g. :class:`PromptedTitan`) along with the
        ``prompt_learner``.
    """
    prompt_kwargs = dict(prompt_kwargs or {})
    load_kwargs = dict(load_kwargs or {})
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name_lc = name.lower()

    if name_lc == "titan":
        from .titan import TitanPromptLearner

        model = _load_titan(device=device, **load_kwargs)
        prompt_learner = TitanPromptLearner(
            classnames=classnames,
            titan=model,
            **prompt_kwargs,
        ).to(device)
        return model, prompt_learner

    raise ValueError(f"Unknown backbone name: {name!r}. Supported: 'titan'.")
