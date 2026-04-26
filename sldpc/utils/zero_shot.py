"""Zero-shot baseline helpers.

Wraps TITAN's official ``zero_shot_classifier`` API so that the
training pipeline can compute and log a zero-shot baseline alongside
each Stage-1/Stage-2 run, with no checkpoint involvement and no
prompt training.

Two callable surfaces:

- :func:`build_zero_shot_weights` — pure function: feed a TITAN model
  and a list-of-list of class synonyms, get back the column-normalized
  ``(D_text, C)`` zero-shot weight matrix.
- :func:`evaluate_zero_shot` — given the weights, the loaded TITAN
  model, and a feature DataLoader, run the eval loop and return the
  same metrics dict produced for trained runs.

Both functions use :func:`compute_classification_metrics` so the
numbers are directly comparable to Stage-1/Stage-2 outputs in
``final_report.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# The 23-template ensemble used by TITAN's official zero-shot reference.
# Source: ``titan/utils.py`` in the MahmoodLab/TITAN code release.
# Each template has a ``CLASSNAME`` placeholder substituted at runtime.
TEMPLATES: List[str] = [
    "CLASSNAME.",
    "an image of CLASSNAME.",
    "the image shows CLASSNAME.",
    "the image displays CLASSNAME.",
    "the image exhibits CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "I observe CLASSNAME.",
    "the pathology image shows CLASSNAME.",
    "a pathology image shows CLASSNAME.",
    "the pathology slide shows CLASSNAME.",
    "shows CLASSNAME.",
    "contains CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "CLASSNAME is observed.",
    "the pathology image reveals CLASSNAME.",
    "a microscopic image of showing CLASSNAME.",
    "histology shows CLASSNAME.",
    "CLASSNAME can be seen.",
    "the tissue shows CLASSNAME.",
    "CLASSNAME is identified.",
]


__all__ = [
    "TEMPLATES",
    "load_class_prompts",
    "build_zero_shot_weights",
    "evaluate_zero_shot",
]


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_class_prompts(yaml_path: Path, class_names: Sequence[str]) -> List[List[str]]:
    """Load the legacy-format prompts yaml.

    The yaml is expected to have a top-level ``prompts`` mapping, each
    value being a non-empty list of synonyms::

        prompts:
          CLASS_A: ["synonym 1", "synonym 2"]
          CLASS_B: ["..."]

    Returns a list-of-lists in the order specified by ``class_names``.
    """
    import yaml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Prompts yaml not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    prompts = cfg.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError(
            f"{yaml_path}: top-level 'prompts' field is missing or not a dict."
        )

    missing = [c for c in class_names if c not in prompts]
    if missing:
        raise ValueError(
            f"{yaml_path}: prompts missing for classes: {missing}. "
            f"Available keys: {sorted(prompts.keys())}"
        )

    out: List[List[str]] = []
    for c in class_names:
        synonyms = prompts[c]
        if not isinstance(synonyms, list) or not synonyms:
            raise ValueError(
                f"{yaml_path}: prompts[{c!r}] must be a non-empty list of strings."
            )
        out.append([str(s) for s in synonyms])
    return out


# ---------------------------------------------------------------------------
# Zero-shot weight construction
# ---------------------------------------------------------------------------

def build_zero_shot_weights(
    titan: "Any",
    class_prompts_list: List[List[str]],
    device: "Any",
):
    """Build the ``(D_text, C)`` zero-shot classifier weight matrix.

    Mirrors the legacy behavior in ``CoOp/train.py``::

        baseline_weights = titan.zero_shot_classifier(
            class_prompts_list, TEMPLATES, device=device,
        )
        baseline_weights = baseline_weights.to(device, dtype=proj_dtype)
        baseline_weights = F.normalize(baseline_weights, dim=0)

    The dim=0 normalization at the end is per-column (per-class),
    bit-for-bit matching the legacy implementation.

    Returns
    -------
    torch.Tensor
        Shape ``(D_text, C)``. Each column is unit-norm.
    """
    import torch
    import torch.nn.functional as F

    if not hasattr(titan, "zero_shot_classifier"):
        raise RuntimeError(
            "Loaded TITAN model does not expose ``zero_shot_classifier``. "
            "Make sure transformers loads the official MahmoodLab/TITAN "
            "remote code via ``trust_remote_code=True``."
        )

    weights = titan.zero_shot_classifier(
        class_prompts_list, TEMPLATES, device=device,
    )
    proj = titan.vision_encoder.proj
    target_dtype = getattr(proj, "weight", proj).dtype
    weights = weights.to(device=device, dtype=target_dtype)
    weights = F.normalize(weights, dim=0)
    return weights


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def evaluate_zero_shot(
    titan: "Any",
    weights: "Any",            # (D_text, C)
    test_loader: "Any",
    device: "Any",
    num_classes: int,
) -> Dict[str, float]:
    """Run zero-shot eval over ``test_loader``, return ACC/F1/AUC.

    Each batch is processed with the same math as legacy
    ``eval_baseline()``::

        emb_proj = feats @ titan.vision_encoder.proj
        emb_norm = F.normalize(emb_proj, dim=-1)
        logits   = emb_norm @ weights
        probs    = softmax(logits, dim=-1)

    Note the absence of a temperature: the legacy zero-shot baseline
    uses raw cosine logits, matching paper Table 2 numbers.
    """
    import torch
    import torch.nn.functional as F

    from sldpc.utils.metrics import compute_classification_metrics

    proj = titan.vision_encoder.proj
    dtype = getattr(proj, "weight", proj).dtype

    all_probs: List["torch.Tensor"] = []
    all_labels: List["torch.Tensor"] = []

    with torch.no_grad():
        for batch in test_loader:
            feats = batch["feat"].to(device=device, dtype=dtype)
            labels = batch["label"]
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels)

            emb_proj = feats @ proj
            emb_norm = F.normalize(emb_proj, dim=-1)
            logits = emb_norm @ weights
            probs = F.softmax(logits, dim=-1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    y_prob = torch.cat(all_probs, dim=0)
    y_true = torch.cat(all_labels, dim=0)
    return compute_classification_metrics(y_true, y_prob, num_classes=num_classes)
