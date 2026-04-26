"""K-shot slide-level feature dataset for SLDPC.

Provides a single dataset class that loads pre-extracted slide-level
embeddings under two storage layouts: a single ``.pkl`` containing all
embeddings, or a directory of per-slide ``.pth`` files. The loading
strategy is selected by ``source_type``; the downstream interface is
identical.

The two-step split logic used in the paper (8:2 stratified outer split,
then per-class k-shot sampling from the 80% training pool) uses RNG
isolation so that the outer split depends only on ``seed`` and is
independent of ``K``.

Returned item format
--------------------
Every ``__getitem__`` call returns a dict::

    {
        "feat":     torch.FloatTensor of shape (D,),
        "label":    int,
        "slide_id": str,
    }

This dict-shaped return allows downstream code to treat features
uniformly without branching on tensor layout.
"""

from __future__ import annotations

import csv
import logging
import pickle
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


__all__ = [
    "SlideFeatureDataset",
    "create_data_split",
]


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class SlideFeatureDataset(Dataset):
    """K-shot slide-level feature dataset.

    Parameters
    ----------
    source_type : {"pkl", "per_slide_pth"}
        Storage layout of the pre-extracted features.

        - ``"pkl"``: a single pickle file containing
          ``{"embeddings": np.ndarray (N, D), "filenames": list[str]}``.
        - ``"per_slide_pth"``: a directory of ``.pth`` files, one per
          slide, organized by ``<feature_dir>/<class_name>/*.pth``.
    features_path : str or Path
        Path to the pkl file (for ``"pkl"``) or to the root feature
        directory (for ``"per_slide_pth"``).
    csv_path : str or Path
        CSV with columns ``slide_id, label`` (no header) listing the
        slides that belong to this split. Typically produced by
        :func:`create_data_split`.
    class_to_idx : dict[str, int]
        Mapping from class name to integer label.
    class_names : list[str], optional
        Only required for ``"per_slide_pth"`` — used to locate per-slide
        ``.pth`` files when the slide's class is ambiguous from the path
        alone. Defaults to the keys of ``class_to_idx`` in insertion
        order.
    """

    def __init__(
        self,
        source_type: str,
        features_path: Union[str, Path],
        csv_path: Union[str, Path],
        class_to_idx: Dict[str, int],
        class_names: Optional[List[str]] = None,
    ) -> None:
        if source_type not in ("pkl", "per_slide_pth"):
            raise ValueError(
                f"source_type must be 'pkl' or 'per_slide_pth'; got {source_type!r}."
            )

        self.source_type = source_type
        self.features_path = Path(features_path)
        self.csv_path = Path(csv_path)
        self.class_to_idx = dict(class_to_idx)
        self.class_names = class_names or list(class_to_idx.keys())

        # Records: list of (slide_id, label_idx, feature_tensor).
        # For pkl-sourced datasets we resolve features eagerly at init
        # time into a single (N, D) tensor and store the row index as
        # the "feature pointer". For per-slide-pth datasets we store the
        # resolved absolute path and load lazily.
        self._records: List[Tuple[str, int, Any]] = self._build_records()

        if not self._records:
            raise RuntimeError(
                f"No usable samples found. features_path={features_path}, "
                f"csv_path={csv_path}. Check that slide_ids in the CSV "
                "match those in the feature store."
            )

        self.dim = self._infer_feature_dim()

    # ------------------------------------------------------------------
    # Record building
    # ------------------------------------------------------------------

    def _build_records(self) -> List[Tuple[str, int, Any]]:
        """Build ``(slide_id, label_idx, ptr)`` tuples from the CSV."""
        slide_entries = self._read_csv()

        if self.source_type == "pkl":
            # Load the single pickle once; store row indices as pointers.
            with open(self.features_path, "rb") as f:
                data = pickle.load(f)
            embeddings = data["embeddings"]
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.from_numpy(np.asarray(embeddings)).float()
            else:
                embeddings = embeddings.float()
            self._embeddings = embeddings          # (N, D)
            id2row = {str(sid): i for i, sid in enumerate(data["filenames"])}

            records = []
            missing = 0
            for sid, cls_name in slide_entries:
                if sid not in id2row:
                    missing += 1
                    continue
                lbl = self.class_to_idx[cls_name]
                records.append((sid, lbl, id2row[sid]))
            if missing:
                logger.warning(
                    "%d slide_ids from %s not found in the pkl feature bank; skipped.",
                    missing, self.csv_path,
                )
            return records

        # source_type == "per_slide_pth"
        # Build an index of all .pth files under features_path, then
        # resolve each CSV entry against it.
        self._embeddings = None
        all_pth = list(self.features_path.rglob("*.pth"))
        stem2path = {p.stem: p for p in all_pth}

        records = []
        missing = 0
        for sid, cls_name in slide_entries:
            if sid not in stem2path:
                missing += 1
                continue
            lbl = self.class_to_idx[cls_name]
            records.append((sid, lbl, stem2path[sid]))
        if missing:
            logger.warning(
                "%d slide_ids from %s not found under %s; skipped.",
                missing, self.csv_path, self.features_path,
            )
        return records

    def _read_csv(self) -> List[Tuple[str, str]]:
        """Read ``slide_id,class_name`` pairs from the split CSV."""
        entries: List[Tuple[str, str]] = []
        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                sid = str(row[0]).strip()
                cls_name = str(row[1]).strip()
                if cls_name not in self.class_to_idx:
                    logger.debug("Skipping unknown class %r for slide %s.", cls_name, sid)
                    continue
                entries.append((sid, cls_name))
        return entries

    def _infer_feature_dim(self) -> int:
        if self.source_type == "pkl":
            return int(self._embeddings.shape[1])
        # per_slide_pth: peek at the first sample
        _, _, ptr = self._records[0]
        feat = self._load_pth_feature(ptr)
        return int(feat.shape[-1])

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pth_feature(pth_path: Path) -> torch.Tensor:
        """Load a single per-slide feature tensor from a .pth file.

        Accepts three common storage layouts:

        1. A raw 1-D ``torch.Tensor`` of shape ``(D,)``.
        2. A dict with a ``"slide_embedding"`` key.
        3. A dict with a ``"embedding"`` key.

        Raises ``KeyError`` if none of the above is found.
        """
        # File-handle form to avoid PyTorch's Windows multi-byte-path
        # issue (#98918 / #103949).
        with open(pth_path, "rb") as fh:
            obj = torch.load(fh, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.Tensor):
            return obj.float().view(-1)
        if isinstance(obj, dict):
            for key in ("slide_embedding", "embedding", "feature"):
                if key in obj:
                    t = obj[key]
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    return t.float().view(-1)
        raise KeyError(
            f"Could not find a feature tensor in {pth_path}. Expected a bare "
            "Tensor or a dict containing one of ('slide_embedding', 'embedding', 'feature')."
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, label, ptr = self._records[idx]
        if self.source_type == "pkl":
            feat = self._embeddings[ptr]
        else:
            feat = self._load_pth_feature(ptr)
        return {"feat": feat.float(), "label": int(label), "slide_id": sid}

    # ------------------------------------------------------------------
    # Convenience: all labels at once (used by HNS feature bank builder)
    # ------------------------------------------------------------------

    @property
    def labels(self) -> torch.Tensor:
        """Return all labels as a 1-D LongTensor, in sample order."""
        return torch.tensor([lbl for _, lbl, _ in self._records], dtype=torch.long)

    def stack_features(self) -> torch.Tensor:
        """Return all features as a single ``(N, D)`` FloatTensor.

        This is the feature bank consumed by
        :func:`sldpc.core.negative_sampler.dynamic_hard_negative_sampler`.
        """
        if self.source_type == "pkl":
            row_ids = torch.tensor(
                [ptr for _, _, ptr in self._records], dtype=torch.long
            )
            return self._embeddings[row_ids].float()

        # per_slide_pth: load each file.
        feats = [self._load_pth_feature(ptr) for _, _, ptr in self._records]
        return torch.stack(feats, dim=0).float()


# -----------------------------------------------------------------------------
# Two-step split generator
# -----------------------------------------------------------------------------

def create_data_split(
    all_slide_entries: List[Tuple[str, str]],
    class_names: List[str],
    few_shot_k: int,
    seed: int,
    split_dir: Union[str, Path],
    train_ratio: float = 0.2,
) -> Dict[str, int]:
    """Two-step stratified split used throughout the paper.

    Step 1: stratified 20/80 outer split seeded by ``seed`` produces
    ``train_pool`` (~20%) and ``test`` (~80%). This split depends only
    on ``seed`` — it is K-independent, so varying K at fixed seed
    leaves the held-out test set unchanged.

    Step 2: from ``train_pool`` draw ``few_shot_k`` samples per class
    to form the K-shot training set. A **separate** RNG instance seeded
    with the same ``seed`` is used for Step 2, so that Step 1 and Step
    2 do not share state.

    Three CSVs are written::

        <split_dir>/train_pool.csv    (all Step-1 training slides)
        <split_dir>/test.csv          (fixed held-out test slides)
        <split_dir>/kshot.csv         (the K-shot subset actually used)

    Each row is ``slide_id,class_name`` with **no header**, matching
    the format read by :class:`SlideFeatureDataset`.

    Parameters
    ----------
    all_slide_entries : list of (slide_id, class_name)
        Universe of usable slides. ``slide_id`` should match the keys
        present in the feature store (``data["filenames"]`` for pkl,
        ``<file>.stem`` for per-slide pth).
    class_names : list[str]
        Ordered list of class names.
    few_shot_k : int
        Per-class training samples for Step 2.
    seed : int
        Random seed. Controls both Step 1 and Step 2 (isolated RNGs).
    split_dir : str or Path
        Output directory; created if absent.
    train_ratio : float, optional
        Fraction of slides assigned to ``train_pool`` in Step 1.
        Default 0.2.

    Returns
    -------
    dict
        Summary statistics: ``{"total", "train_pool", "test", "kshot",
        "few_shot_k", "seed"}``.
    """
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    entries = list(all_slide_entries)
    entries.sort(key=lambda x: x[0])    # deterministic ordering

    # ---------------- Step 1 : stratified 20/80 outer split ----------------
    rng_split = random.Random(seed)
    by_class: Dict[str, List[str]] = defaultdict(list)
    for sid, cls_name in entries:
        if cls_name in class_names:
            by_class[cls_name].append(sid)

    train_pool: List[Tuple[str, str]] = []
    test_set: List[Tuple[str, str]] = []
    for cls_name in sorted(by_class.keys()):
        sids = list(by_class[cls_name])
        rng_split.shuffle(sids)
        n_train = max(1, int(len(sids) * train_ratio))
        for s in sids[:n_train]:
            train_pool.append((s, cls_name))
        for s in sids[n_train:]:
            test_set.append((s, cls_name))

    # ---------------- Step 2 : per-class K-shot sampling ------------------
    # Aligned with legacy TCGAOncoTreeDataset few-shot sampling:
    #   - Group in CSV/insertion order (NOT sorted).
    #   - Use GLOBAL ``random.seed`` (wrapped in getstate/setstate to
    #     avoid polluting the caller's RNG).
    #   - Use ``random.sample(pool, k)`` (NOT ``shuffle + [:k]``).
    # This keeps Step 2 consistent between the auto-split path and the
    # fixed-CSV path in ``_prepare_splits``.
    pool_by_class: "OrderedDict[str, List[str]]" = OrderedDict()
    for sid, cls_name in train_pool:
        pool_by_class.setdefault(cls_name, []).append(sid)

    _saved_rng_state = random.getstate()
    try:
        random.seed(seed)
        kshot: List[Tuple[str, str]] = []
        for cls_name, sids in pool_by_class.items():
            n = min(few_shot_k, len(sids))
            if n < few_shot_k:
                logger.warning(
                    "Class %s has only %d samples in train_pool (<%d). Using all of them.",
                    cls_name, len(sids), few_shot_k,
                )
            chosen = random.sample(sids, n)
            for s in chosen:
                kshot.append((s, cls_name))
    finally:
        random.setstate(_saved_rng_state)

    # ---------------- Persist CSVs ----------------------------------------
    def _save(path: Path, rows: List[Tuple[str, str]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            for sid, cls_name in rows:
                w.writerow([sid, cls_name])

    _save(split_dir / "train_pool.csv", train_pool)
    _save(split_dir / "test.csv", test_set)
    _save(split_dir / "kshot.csv", kshot)

    return {
        "total": len(entries),
        "train_pool": len(train_pool),
        "test": len(test_set),
        "kshot": len(kshot),
        "few_shot_k": few_shot_k,
        "seed": seed,
    }
