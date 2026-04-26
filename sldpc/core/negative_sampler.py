"""Dynamic Hard-Negative Sampler (DHNO) for Slide-Level Dual-Prompt Collaboration.

This module implements Section 3.4 of the SLDPC paper: given a mini-batch of
slide features, use the frozen base prompt ``P`` as a query to retrieve the
Top-K most similar classes (excluding the ground truth and intra-batch
duplicates) as hard negatives, then sample one slide per hard-negative class
from a pre-computed feature bank.

The resulting extended batch has shape ``(B*k, D)``: each block of ``k``
consecutive rows is ``[positive | neg_1 | neg_2 | ... | neg_{k-1}]``, and
the paired label vector records the ground-truth class for each row.

Interface
---------
Two functions are exported:

- :func:`build_class_index` — construct a ``{class_id: tensor_of_indices}``
  lookup from various data sources (label tensor, Dataset with a ``samples``
  attribute, or a ``[(path, label), ...]`` list).
- :func:`dynamic_hard_negative_sampler` — the sampling algorithm itself.

Numerical determinism
---------------------
Reproducibility is controlled by ``torch.manual_seed(...)`` in the caller.
The implementation uses :class:`torch.Generator` on the sampler's device
when one is provided, otherwise it falls back to Python's ``random``
module (seeded via the global ``random.seed``). The Python fallback was
introduced originally to avoid a known CUDA/CPU ``torch.Generator``
incompatibility in certain PyTorch versions; it is retained here as the
default because it is more portable.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import torch
from .fusion import l2norm

if TYPE_CHECKING:
    from torch.utils.data import Dataset


__all__ = [
    "build_class_index",
    "dynamic_hard_negative_sampler",
    "random_negative_sampler",
]


# -----------------------------------------------------------------------------
# Class index builder
# -----------------------------------------------------------------------------

def build_class_index(
    data_source: Union[torch.Tensor, List, "Dataset"],
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """Build a ``{class_id: sample_indices}`` lookup.

    Parameters
    ----------
    data_source : one of
        - ``torch.Tensor``: a 1-D label tensor of shape ``(N,)``.
        - A dataset-like object exposing a ``samples`` attribute that yields
          ``(feature_or_path, label)`` pairs.
        - A plain ``list`` of ``(feature_or_path, label)`` tuples.
    device : torch.device, optional
        If given, the returned index tensors are moved to this device.

    Returns
    -------
    dict[int, torch.Tensor]
        Mapping from class id to a 1-D LongTensor of sample indices.

    Raises
    ------
    TypeError
        If ``data_source`` is none of the supported types.
    """
    # -------- Normalize to a 1-D label tensor ---------------------------
    if isinstance(data_source, torch.Tensor):
        label_bank = data_source
    elif hasattr(data_source, "samples"):
        labels = [label for _, label in data_source.samples]
        label_bank = torch.tensor(labels, dtype=torch.long)
    elif isinstance(data_source, list):
        labels = [label for _, label in data_source]
        label_bank = torch.tensor(labels, dtype=torch.long)
    else:
        raise TypeError(
            f"Unsupported data_source type: {type(data_source)}. "
            "Expected Tensor, Dataset with .samples, or list of (x, label) tuples."
        )

    if device is not None:
        label_bank = label_bank.to(device)

    # -------- Build the inverted index ----------------------------------
    classes = torch.unique(label_bank)
    result: Dict[int, torch.Tensor] = {}
    for c in classes:
        idx = (label_bank == c).nonzero(as_tuple=False).squeeze(1)
        if device is not None:
            idx = idx.to(device)
        result[int(c.item())] = idx
    return result


# -----------------------------------------------------------------------------
# Core sampler
# -----------------------------------------------------------------------------

def dynamic_hard_negative_sampler(
    img_feat_batch: torch.Tensor,
    label_batch: torch.Tensor,
    text_feat_frozen: torch.Tensor,
    feature_bank: torch.Tensor,
    class_to_indices: Dict[int, torch.Tensor],
    base_class_mask: torch.Tensor,
    k: int = 8,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample hard negatives and build the extended batch (Sec. 3.4).

    For every positive sample ``i`` in the mini-batch, the sampler picks
    ``k-1`` hard-negative classes (the Top-K classes most similar to the
    sample's feature under the frozen prompt, excluding the ground-truth
    class and any class already present in the mini-batch), then draws one
    image per hard-negative class from ``feature_bank``. The result is a
    flat ``(B*k, D)`` extended batch paired with a ``(B*k,)`` label vector.

    Parameters
    ----------
    img_feat_batch : torch.Tensor
        Slide features of shape ``(B, D)`` — the positives.
    label_batch : torch.Tensor
        Ground-truth class indices of shape ``(B,)``.
    text_feat_frozen : torch.Tensor
        Class text features from the frozen base prompt ``P``, shape ``(C, D)``.
    feature_bank : torch.Tensor
        Slide-level features for all training samples, shape ``(N, D)``.
    class_to_indices : dict[int, torch.Tensor]
        Output of :func:`build_class_index` — maps class id to sample ids
        in ``feature_bank``.
    base_class_mask : torch.Tensor
        Boolean mask of shape ``(C,)`` selecting which classes are eligible
        as negatives. In standard few-shot classification this is all-True;
        in Base->New evaluation it masks out novel classes.
    k : int, optional
        Number of rows per positive in the output (1 positive + ``k-1``
        hard negatives). Default 8.
    rng : torch.Generator, optional
        Torch random generator. If ``None``, Python's ``random`` module is
        used (seeded via ``random.seed`` in the caller). Passing an explicit
        generator is only useful when you need device-specific torch RNG.

    Returns
    -------
    feat_ext : torch.Tensor
        Shape ``(B*k, D)``. Row ``i*k + j`` is the ``j``-th feature for the
        ``i``-th positive; ``j == 0`` is the positive itself.
    label_ext : torch.Tensor
        Shape ``(B*k,)``. Ground-truth class id for each row.
    """
    device = img_feat_batch.device
    B = img_feat_batch.size(0)

    # ------------------- 1. Normalize & select candidate classes -------
    img_norm = l2norm(img_feat_batch)                                   # (B, D)
    txt_base = l2norm(text_feat_frozen)[base_class_mask]                # (C_b, D)
    base_ids = torch.arange(
        text_feat_frozen.size(0), device=device
    )[base_class_mask]                                                  # (C_b,)
    C_base = txt_base.size(0)
    k_eff = min(k, C_base)

    # ------------------- 2. Top-K similarity lookup --------------------
    sim = img_norm @ txt_base.t()                                       # (B, C_b)
    _, topk_local = sim.topk(k_eff, dim=-1)                             # (B, k_eff)
    topk_cls_ids = base_ids[topk_local]                                 # (B, k_eff)

    # ------------------- 3. Assemble [GT | hard-neg ...] per row -------
    # Per positive i we build a length-k list:
    #   out[0] = ground truth
    #   out[1..k-1] = Top-K classes filtered to exclude
    #       (a) the positive's own GT, and
    #       (b) any class that is the GT of another sample in this batch
    #           (avoids trivially mixing another positive as a "negative").
    # If the filtered Top-K is not enough to reach k entries, we fall
    # back to uniform random sampling from the pool of remaining classes.
    unique_cls = torch.empty((B, k), dtype=torch.long, device=device)
    batch_gt_set = set(label_batch.tolist())

    for i in range(B):
        self_gt = int(label_batch[i])
        taken = {self_gt}
        out = [self_gt]
        forbid = batch_gt_set - {self_gt}

        for cls in topk_cls_ids[i].tolist():
            if cls not in taken and cls not in forbid:
                taken.add(cls)
                out.append(cls)
            if len(out) == k:
                break

        # Random fill if Top-K was exhausted before k was reached.
        while len(out) < k:
            pool = [c for c in base_ids.tolist() if c not in taken]
            if not pool:
                # All base classes already used — fall back to sampling from
                # base_ids (will duplicate, but keeps shape consistent).
                pool = base_ids.tolist()

            if rng is not None:
                idx = torch.randint(
                    0, len(pool), (1,), generator=rng, device=device
                ).item()
                rand_cls = pool[idx]
            else:
                rand_cls = random.choice(pool)

            taken.add(rand_cls)
            out.append(rand_cls)

        unique_cls[i] = torch.tensor(out, device=device, dtype=torch.long)

    return _assemble_extended_batch(
        img_feat_batch=img_feat_batch,
        feature_bank=feature_bank,
        class_to_indices=class_to_indices,
        unique_cls=unique_cls,
        rng=rng,
    )


def random_negative_sampler(
    img_feat_batch: torch.Tensor,
    label_batch: torch.Tensor,
    feature_bank: torch.Tensor,
    class_to_indices: Dict[int, torch.Tensor],
    base_class_mask: torch.Tensor,
    k: int = 8,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random-negative variant with the same output contract as DHNO.

    Used by directionality / component ablations where hard-negative class
    mining is replaced by uniform class sampling.
    """
    device = img_feat_batch.device
    B = img_feat_batch.size(0)

    base_ids = torch.arange(
        int(base_class_mask.numel()), device=device
    )[base_class_mask]

    unique_cls = torch.empty((B, k), dtype=torch.long, device=device)
    batch_gt_set = set(label_batch.tolist())

    for i in range(B):
        self_gt = int(label_batch[i])
        taken = {self_gt}
        out = [self_gt]
        forbid = batch_gt_set - {self_gt}

        while len(out) < k:
            pool = [c for c in base_ids.tolist() if c not in taken and c not in forbid]
            if not pool:
                pool = [c for c in base_ids.tolist() if c not in taken]
            if not pool:
                pool = base_ids.tolist()

            if rng is not None:
                idx = torch.randint(0, len(pool), (1,), generator=rng, device=device).item()
                rand_cls = pool[idx]
            else:
                rand_cls = random.choice(pool)
            taken.add(rand_cls)
            out.append(rand_cls)

        unique_cls[i] = torch.tensor(out, device=device, dtype=torch.long)

    return _assemble_extended_batch(
        img_feat_batch=img_feat_batch,
        feature_bank=feature_bank,
        class_to_indices=class_to_indices,
        unique_cls=unique_cls,
        rng=rng,
    )


def _assemble_extended_batch(
    img_feat_batch: torch.Tensor,
    feature_bank: torch.Tensor,
    class_to_indices: Dict[int, torch.Tensor],
    unique_cls: torch.Tensor,
    rng: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared helper: class lists -> sampled extended features."""
    device = img_feat_batch.device
    B, k = unique_cls.shape

    neg_cls_flat = unique_cls[:, 1:].reshape(-1)                        # (B*(k-1),)
    selected_indices: List[int] = []
    for c in neg_cls_flat.tolist():
        pool_tensor = class_to_indices[int(c)]
        if rng is not None:
            idx = torch.randint(
                0, pool_tensor.numel(), (1,), generator=rng, device=device
            ).item()
        else:
            idx = random.randint(0, pool_tensor.numel() - 1)
        selected_indices.append(int(pool_tensor[idx].item()))

    selected_idx = torch.tensor(selected_indices, dtype=torch.long, device=device)
    img_neg_flat = feature_bank.index_select(0, selected_idx)           # (B*(k-1), D)
    img_neg = img_neg_flat.view(B, k - 1, -1)                           # (B, k-1, D)
    img_pos = img_feat_batch.unsqueeze(1)                               # (B, 1, D)

    feat_ext = torch.cat([img_pos, img_neg], dim=1).view(B * k, -1)    # (B*k, D)
    label_ext = unique_cls.view(-1)                                     # (B*k,)
    return feat_ext, label_ext
