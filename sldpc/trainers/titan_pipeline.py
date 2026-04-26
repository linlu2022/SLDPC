#!/usr/bin/env python
"""Unified TITAN training pipeline for paper experiments and ablations.

This module exposes a single CLI entrypoint used by `scripts/train_titan.py`.
It supports:
- main experiments (4 datasets, 5-seed setup)
- fixed-split vs random-split policies
- ablation switches through explicit CLI flags
- legacy-compatible argument aliases such as
  `--no_initial_checkpoint`, `--dhno_mode`, `--dynamic_omega_b`
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import yaml

# Heavyweight imports that we only need for type checking — runtime usage
# is guarded inside functions so that ``--help`` / ``--dry-run`` paths can
# work even before ``pip install -r requirements.txt`` has been run.
if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
    from sldpc.backbones.titan import PromptedTitan


_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    config: str
    features_path: str
    fixed_train_csv: Optional[str]
    fixed_test_csv: Optional[str]


DATASET_PRESETS: Dict[str, DatasetPreset] = {
    "tcga_nsclc": DatasetPreset(
        name="tcga_nsclc",
        config="configs/datasets/tcga_nsclc.yaml",
        features_path="data/datasets/tcga_nsclc/TCGA-NSCLC_feature.pkl",
        fixed_train_csv="data/datasets/tcga_nsclc/TCGA-NSCLC_train.csv",
        fixed_test_csv="data/datasets/tcga_nsclc/TCGA-NSCLC_test.csv",
    ),
    "tcga_rcc": DatasetPreset(
        name="tcga_rcc",
        config="configs/datasets/tcga_rcc.yaml",
        features_path="data/datasets/tcga_rcc/TCGA-RCC_feature.pkl",
        fixed_train_csv="data/datasets/tcga_rcc/TCGA-RCC_train.csv",
        fixed_test_csv="data/datasets/tcga_rcc/TCGA-RCC_test.csv",
    ),
    "ubc_ocean": DatasetPreset(
        name="ubc_ocean",
        config="configs/datasets/ubc_ocean.yaml",
        features_path="data/datasets/ubc_ocean/slide_embedding_UBC-OCEAN.pkl",
        fixed_train_csv="data/datasets/ubc_ocean/UBC-OCEAN_all_train.csv",
        fixed_test_csv="data/datasets/ubc_ocean/UBC-OCEAN_all_test.csv",
    ),
    "tcga_ot": DatasetPreset(
        name="tcga_ot",
        config="configs/datasets/tcga_ot.yaml",
        features_path="data/datasets/tcga_ot/TCGA_TITAN_features.pkl",
        fixed_train_csv="data/datasets/tcga_ot/tcga-ot_train.csv",
        fixed_test_csv="data/datasets/tcga_ot/tcga-ot_test.csv",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SLDPC + TITAN unified runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset", type=str, choices=list(DATASET_PRESETS.keys()), default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--backbone-config", type=str, default="configs/backbones/titan.yaml")
    parser.add_argument("--features-path", "--feat_pkl", dest="features_path", type=str, default=None)
    parser.add_argument("--source-type", type=str, default="pkl", choices=["pkl", "per_slide_pth"])

    parser.add_argument("--hf-model-id", type=str, default="MahmoodLab/TITAN")
    parser.add_argument("--hf-revision", type=str, default=None)
    parser.add_argument("--hf-local-files-only", action="store_true")

    # --- Seed control ---------------------------------------------------
    # Two mutually-exclusive modes:
    #   (A) --seed S1 S2 ...  : run exactly those seeds (highest priority).
    #   (B) --n-seeds N       : generate N reproducible random seeds from
    #                           a fixed meta-seed. (default: N=5)
    # The old --seed-list flag has been removed; use --seed instead.
    parser.add_argument(
        "--seed", "--seeds",
        dest="seed",
        nargs="+",
        type=int,
        default=None,
        help="One or more explicit seeds, e.g. '--seed 9448 52 21'. "
             "Takes priority over --n-seeds when provided.",
    )
    parser.add_argument(
        "--n-seeds", "--num_seeds",
        dest="n_seeds",
        type=int,
        default=5,
        help="Number of random seeds to generate (ignored if --seed is given). "
             "Seeds are drawn deterministically from a fixed meta-seed.",
    )
    parser.add_argument("--few-shot-k", "--few_shot_k", dest="few_shot_k", type=int, default=16)

    parser.add_argument("--fixed-train-csv", "--train_csv_base", dest="fixed_train_csv", type=str, default=None)
    parser.add_argument("--fixed-test-csv", "--val_csv_base", dest="fixed_test_csv", type=str, default=None)
    parser.add_argument("--val-csv-new", "--val_csv_new", dest="val_csv_new", type=str, default=None)
    parser.add_argument(
        "--use-fixed-split",
        action="store_true",
        help="Use dataset preset fixed train/test CSVs when --dataset is provided.",
    )

    parser.add_argument("--output-dir", type=str, default="runs/titan_experiments")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument(
        "--skip-zero-shot",
        action="store_true",
        help="Skip the per-seed TITAN zero-shot baseline evaluation. "
             "By default, every seed runs a zero-shot eval before Stage-1.",
    )
    parser.add_argument(
        "--zero-shot-prompts-yaml",
        dest="zero_shot_prompts_yaml",
        type=str,
        default=None,
        help="Override path to the zero-shot prompts yaml. "
             "Default: data/datasets/zero_shot_prompts/<dataset>.yaml",
    )
    parser.add_argument("--stage2-init", type=str, default=None, choices=["cpi", "random"])
    parser.add_argument(
        "--no-initial-checkpoint",
        "--no_initial_checkpoint",
        dest="no_initial_checkpoint",
        action="store_true",
        help="Compatibility flag: run Stage-2 from random prompt initialization.",
    )

    parser.add_argument(
        "--stage2-dhno-mode",
        "--dhno_mode",
        dest="stage2_dhno_mode",
        type=str,
        default=None,
        choices=["full", "sampling_only", "none"],
    )
    parser.add_argument(
        "--stage2-negative-sampling",
        "--negative_sampling",
        dest="stage2_negative_sampling",
        type=str,
        default=None,
        choices=["hns", "random"],
    )
    parser.add_argument(
        "--stage2-loss",
        dest="stage2_loss",
        type=str,
        default=None,
        choices=["symmetric", "i2t", "t2i", "ce"],
    )
    parser.add_argument(
        "--stage2-eval-mode",
        "--eval_mode",
        dest="stage2_eval_mode",
        type=str,
        default=None,
        choices=["fused", "task", "base", "eval_base", "eval_base_new"],
    )

    parser.add_argument(
        "--dynamic-omega-b",
        "--dynamic_omega_b",
        dest="dynamic_omega_b",
        action="store_true",
        help="Search best fused omega on validation set after Stage-2.",
    )
    parser.add_argument(
        "--omega-search-values",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )

    parser.add_argument(
        "--best-metric",
        "--best_metric",
        dest="best_metric",
        type=str,
        default="F1",
        choices=["ACC", "F1", "AUC"],
    )

    parser.add_argument("--n-ctx", "--n_ctx", dest="n_ctx", type=int, default=None)
    parser.add_argument("--ctx-init", "--ctx_init", dest="ctx_init", type=str, default=None)
    parser.add_argument("--csc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--class-token-position",
        "--class_token_position",
        dest="class_token_position",
        type=str,
        default=None,
        choices=["front", "middle", "end"],
    )
    parser.add_argument("--omega", "--omega_b", dest="omega", type=float, default=None)

    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--workers", "--num_workers", dest="num_workers", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--stage1-lr", type=float, default=None)
    parser.add_argument("--stage2-lr", type=float, default=None)

    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)

    # ---- Behavior toggles (override backbone yaml defaults) ----
    # These flip between legacy (CE-on-cosine, no tau) and paper/modern
    # (CE-on-cosine/tau) numerics. Defaults live in configs/backbones/*.yaml.
    parser.add_argument(
        "--stage1-apply-tau",
        dest="stage1_apply_tau",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Scale Stage-1 CE logits by 1/tau (paper Eq. 7). "
             "Legacy default: false.",
    )
    parser.add_argument(
        "--stage2-ce-apply-tau",
        dest="stage2_ce_apply_tau",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Scale Stage-2 CE-ablation logits by 1/tau. Legacy default: false.",
    )
    parser.add_argument(
        "--eval-apply-tau-in-softmax",
        dest="eval_apply_tau_in_softmax",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use softmax(cos/tau) for eval probabilities (AUC input). "
             "Legacy default: false (softmax(cos)).",
    )
    parser.add_argument(
        "--use-lr-scheduler",
        dest="use_lr_scheduler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable ReduceLROnPlateau(mode='max', factor=0.1, patience=8). "
             "Default: off.",
    )

    # Legacy flags accepted for compatibility.
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--eval-prompt", action="store_true")
    parser.add_argument("--eval-baseline", action="store_true")
    parser.add_argument("--few-shot-indices-dir", "--few_shot_indices_dir", dest="few_shot_indices_dir", type=str, default=None)
    parser.add_argument("--few-shot-seed-dir", "--few_shot_seed_dir", dest="few_shot_seed_dir", type=str, default=None)
    parser.add_argument("--save-few-shot-indices", "--save_few_shot_indices", dest="save_few_shot_indices", action="store_true")

    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (_ROOT / p).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_yaml_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return _load_yaml(path)


def _detect_header(first_row: List[str]) -> bool:
    known = {
        "slide_id", "filename", "file_name", "wsi", "image_id", "id",
        "class_name", "label", "oncotreecode", "oncotree_code",
        "project_id", "tumor_type", "subtype",
    }
    return any(str(x).strip().lower() in known for x in first_row)


def _pick_columns_from_header(
    header: List[str],
    records: List[Dict[str, str]],
    class_names_set: set[str],
    slide_hint: Optional[str] = None,
    label_hint: Optional[str] = None,
) -> Tuple[str, str]:
    h2orig = {h.strip().lower(): h for h in header}

    def _resolve_hint(hint: Optional[str]) -> Optional[str]:
        if not hint:
            return None
        return h2orig.get(hint.strip().lower())

    slide_col = _resolve_hint(slide_hint)
    label_col = _resolve_hint(label_hint)

    slide_candidates = ["slide_id", "filename", "file_name", "wsi", "id", "image_id"]
    label_candidates = ["class_name", "label", "oncotreecode", "oncotree_code", "subtype", "tumor_type"]
    sample = records[: min(200, len(records))]

    if slide_col is None:
        slide_col = next((h2orig[c] for c in slide_candidates if c in h2orig), None)

    if label_col is None:
        for c in label_candidates:
            if c not in h2orig:
                continue
            cand = h2orig[c]
            overlap = sum(1 for r in sample if str(r.get(cand, "")).strip() in class_names_set)
            if overlap > 0:
                label_col = cand
                break

    if label_col is None:
        best_col = None
        best_overlap = -1
        for col in header:
            overlap = sum(1 for r in sample if str(r.get(col, "")).strip() in class_names_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = col
        if best_col is not None and best_overlap > 0:
            label_col = best_col

    if slide_col is None:
        slide_col = header[0]
    if label_col is None:
        label_col = header[1] if len(header) > 1 else header[0]

    return slide_col, label_col


def _read_slide_class_rows(
    csv_path: str,
    class_names: List[str],
    *,
    slide_id_col_hint: Optional[str] = None,
    label_col_hint: Optional[str] = None,
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    class_names_set = set(class_names)

    csv_file = _resolve_path(csv_path)
    with csv_file.open("r", encoding="utf-8", newline="") as f:
        raw_reader = csv.reader(f)
        all_rows = [r for r in raw_reader if r and len(r) >= 2]

    if not all_rows:
        return rows

    first_row = [str(x).strip() for x in all_rows[0]]
    has_header = _detect_header(first_row)

    if has_header:
        header = first_row
        dict_rows: List[Dict[str, str]] = []
        for r in all_rows[1:]:
            item = {header[i]: str(r[i]).strip() for i in range(min(len(header), len(r)))}
            dict_rows.append(item)
        if not dict_rows:
            return rows

        slide_col, label_col = _pick_columns_from_header(
            header,
            dict_rows,
            class_names_set,
            slide_hint=slide_id_col_hint,
            label_hint=label_col_hint,
        )
        for item in dict_rows:
            sid = str(item.get(slide_col, "")).strip()
            cls_name = str(item.get(label_col, "")).strip()
            if not sid or cls_name not in class_names_set:
                continue
            rows.append((sid, cls_name))
    else:
        for r in all_rows:
            sid = str(r[0]).strip()
            cls_name = str(r[1]).strip()
            if not sid or cls_name not in class_names_set:
                continue
            rows.append((sid, cls_name))

    return rows


def _has_nonempty_valid_split_csv(path: Path, class_names_set: set[str]) -> bool:
    if not path.exists():
        return False
    valid_rows = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 2:
                continue
            sid = str(r[0]).strip()
            cls = str(r[1]).strip()
            if sid and cls in class_names_set:
                valid_rows += 1
    return valid_rows > 0


def _derive_universe(
    features_path: Path,
    source_type: str,
    class_names: List[str],
) -> List[Tuple[str, str]]:
    if source_type == "pkl":
        import pickle

        with features_path.open("rb") as f:
            data = pickle.load(f)
        filenames = data["filenames"]
        rows: List[Tuple[str, str]] = []
        upper = {c.upper(): c for c in class_names}
        for fn in filenames:
            match = next((upper[u] for u in upper if u in str(fn).upper()), None)
            if match is not None:
                rows.append((str(fn), match))
        return rows

    rows: List[Tuple[str, str]] = []
    upper = {c.upper(): c for c in class_names}
    pths = [
        p for p in Path(features_path).rglob("*.pth")
        if not any(skip in p.name.lower() for skip in ("stats", "checkpoint", "model", "best"))
    ]
    pths.sort(key=lambda p: p.name)
    for pth in pths:
        rel_parts_upper = [p.upper() for p in pth.relative_to(features_path).parts[:-1]]
        match = None
        for u, orig in upper.items():
            if any(u in part for part in rel_parts_upper):
                match = orig
                break
        if match is not None:
            rows.append((pth.stem, match))
    return rows


def _prepare_splits(
    dataset_cfg: Dict[str, Any],
    features_path: Path,
    source_type: str,
    class_names: List[str],
    few_shot_k: int,
    seed: int,
    split_dir: Path,
    fixed_train_csv: Optional[str] = None,
    fixed_test_csv: Optional[str] = None,
    csv_schema: Optional[Dict[str, Any]] = None,
) -> None:
    from sldpc.data import create_data_split

    split_files = [split_dir / name for name in ("train_pool.csv", "test.csv", "kshot.csv")]
    has_existing_split = all(p.exists() for p in split_files)
    class_names_set = set(class_names)
    csv_schema = dict(csv_schema or {})

    if (fixed_train_csv is None) ^ (fixed_test_csv is None):
        raise ValueError("--fixed-train-csv and --fixed-test-csv must be provided together.")

    if has_existing_split and not (fixed_train_csv and fixed_test_csv):
        split_ok = all(_has_nonempty_valid_split_csv(p, class_names_set) for p in split_files)
        if split_ok:
            logging.info("Split CSVs already exist at %s; reusing.", split_dir)
            return
        logging.warning("Existing split files at %s are empty/invalid; regenerating.", split_dir)

    if has_existing_split and (fixed_train_csv and fixed_test_csv):
        logging.info("Fixed split CSVs provided; regenerating split files at %s.", split_dir)

    if fixed_train_csv and fixed_test_csv:
        import pickle
        import random
        from collections import OrderedDict

        fixed_schema = dict(csv_schema.get("fixed", {}))
        train_pool_rows = _read_slide_class_rows(
            fixed_train_csv,
            class_names,
            slide_id_col_hint=fixed_schema.get("slide_id_col"),
            label_col_hint=fixed_schema.get("label_col"),
        )
        test_rows = _read_slide_class_rows(
            fixed_test_csv,
            class_names,
            slide_id_col_hint=fixed_schema.get("slide_id_col"),
            label_col_hint=fixed_schema.get("label_col"),
        )

        if not train_pool_rows or not test_rows:
            raise RuntimeError("Fixed split CSV parsing yielded empty train or test rows.")

        avail_ids = None
        if source_type == "pkl":
            with features_path.open("rb") as f:
                data = pickle.load(f)
            avail_ids = set(map(str, data["filenames"]))
        elif source_type == "per_slide_pth":
            avail_ids = {
                p.stem for p in Path(features_path).rglob("*.pth")
                if not any(skip in p.name.lower() for skip in ("stats", "checkpoint", "model", "best"))
            }

        if avail_ids is not None:
            train_pool_rows = [(sid, cls) for sid, cls in train_pool_rows if sid in avail_ids]
            test_rows = [(sid, cls) for sid, cls in test_rows if sid in avail_ids]

        # --- K-shot sampling: aligned with legacy TCGAOncoTreeDataset -----
        # Legacy behavior (dataset.py L68-80):
        #   1) Use the GLOBAL `random.seed(seed)` rather than a local
        #      ``random.Random(seed)`` instance.
        #   2) Group slides by class in CSV-first-appearance order
        #      (OrderedDict insertion order).
        #   3) Iterate classes in that insertion order (NOT sorted).
        #   4) Sample with ``random.sample(pool, k)`` (NOT shuffle+take).
        # Items 1-4 must all match or the selected K-shot subset will
        # differ even at identical ``seed``. We wrap the seed call in
        # getstate/setstate so this function does not pollute the
        # module-level RNG (improvement over legacy code).
        by_class: "OrderedDict[str, List[str]]" = OrderedDict()
        for sid, cls_name in train_pool_rows:
            by_class.setdefault(cls_name, []).append(sid)

        _saved_rng_state = random.getstate()
        try:
            random.seed(seed)
            kshot_rows: List[Tuple[str, str]] = []
            for cls_name, sid_list in by_class.items():
                take = min(few_shot_k, len(sid_list))
                if take < few_shot_k:
                    logging.warning(
                        "Class %s has only %d samples in fixed train pool (<%d). Using all.",
                        cls_name,
                        len(sid_list),
                        few_shot_k,
                    )
                chosen = random.sample(sid_list, take)
                kshot_rows.extend((s, cls_name) for s in chosen)
        finally:
            random.setstate(_saved_rng_state)

        def _write_rows(path: Path, rows: List[Tuple[str, str]]) -> None:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

        _write_rows(split_dir / "train_pool.csv", train_pool_rows)
        _write_rows(split_dir / "test.csv", test_rows)
        _write_rows(split_dir / "kshot.csv", kshot_rows)
        return

    all_slides_csv = dataset_cfg.get("all_slides_csv")
    if all_slides_csv:
        all_schema = dict(csv_schema.get("all_slides", {}))
        rows = _read_slide_class_rows(
            all_slides_csv,
            class_names,
            slide_id_col_hint=all_schema.get("slide_id_col"),
            label_col_hint=all_schema.get("label_col"),
        )
        if not rows:
            raise RuntimeError(
                f"No usable rows parsed from all_slides_csv={all_slides_csv}. "
                "Check CSV columns or provide fixed split CSVs."
            )
    else:
        rows = _derive_universe(features_path, source_type, class_names)
        if not rows:
            raise RuntimeError(
                "No slide/class rows could be derived from feature store. "
                "Provide fixed split CSVs or set dataset all_slides_csv."
            )

    create_data_split(
        all_slide_entries=rows,
        class_names=class_names,
        few_shot_k=few_shot_k,
        seed=seed,
        split_dir=split_dir,
    )


def _resolve_dataset_inputs(args: argparse.Namespace) -> None:
    if args.dataset:
        preset = DATASET_PRESETS[args.dataset]
        if not args.config:
            args.config = preset.config
        if not args.features_path:
            args.features_path = preset.features_path
        if args.use_fixed_split:
            if not preset.fixed_train_csv or not preset.fixed_test_csv:
                raise ValueError(f"Dataset {args.dataset} has no fixed split preset.")
            if not args.fixed_train_csv:
                args.fixed_train_csv = preset.fixed_train_csv
            if not args.fixed_test_csv:
                args.fixed_test_csv = preset.fixed_test_csv

    if not args.config:
        raise ValueError("Missing --config (or pass --dataset to use a preset).")
    if not args.features_path:
        raise ValueError("Missing --features-path (or pass --dataset to use a preset).")


def _resolve_stage2_eval_mode(raw_mode: Optional[str]) -> Optional[str]:
    if raw_mode is None:
        return None
    mapping = {
        "eval_base": "fused",
        "eval_base_new": "fused",
    }
    return mapping.get(raw_mode, raw_mode)


# Meta-seed used to generate reproducible seed lists from --n-seeds.
# Derived from the ASCII codes of "SLDPC" so that each checkout gets
# the exact same random seeds without hard-coding magic numbers.
_META_SEED = int.from_bytes(b"SLDPC", "big")      # == 0x534c445043 (~358B)


def _resolve_seeds(args: argparse.Namespace) -> List[int]:
    """Resolve which seeds to run.

    Priority rules (first match wins):
      1. ``--seed S1 S2 ...`` explicitly given  -> run exactly those seeds,
         in the order given. Duplicates are rejected.
      2. ``--n-seeds N`` (default N=5)          -> draw N distinct int32
         seeds deterministically from a fixed meta-seed. Every invocation
         of the script with the same N gets the exact same seed set.
    """
    if args.seed:                                        # mode (A)
        seeds = [int(s) for s in args.seed]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seed values are not allowed: {seeds}")
        return seeds

    # mode (B)
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be > 0")

    import numpy as np
    rng = np.random.default_rng(_META_SEED)
    # Draw a slightly larger pool and deduplicate, in case of collisions.
    pool = rng.integers(1, 2**31 - 1, size=args.n_seeds * 4, dtype=np.int64).tolist()
    uniq: List[int] = []
    seen: set = set()
    for s in pool:
        if s not in seen:
            seen.add(s)
            uniq.append(int(s))
            if len(uniq) == args.n_seeds:
                break
    if len(uniq) < args.n_seeds:
        raise RuntimeError(f"Unable to draw {args.n_seeds} distinct seeds from meta-seed.")
    return uniq


def _build_backbone_cfg(base_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(base_cfg)

    if args.epochs is not None:
        cfg["stage1_epochs"] = int(args.epochs)
        cfg["stage2_epochs"] = int(args.epochs)
    if args.stage1_epochs is not None:
        cfg["stage1_epochs"] = int(args.stage1_epochs)
    if args.stage2_epochs is not None:
        cfg["stage2_epochs"] = int(args.stage2_epochs)

    if args.lr is not None:
        cfg["stage1_lr"] = float(args.lr)
        cfg["stage2_lr"] = float(args.lr)
    if args.stage1_lr is not None:
        cfg["stage1_lr"] = float(args.stage1_lr)
    if args.stage2_lr is not None:
        cfg["stage2_lr"] = float(args.stage2_lr)

    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)

    if args.weight_decay is not None:
        cfg["weight_decay"] = float(args.weight_decay)
    if args.patience is not None:
        cfg["patience"] = int(args.patience)
    if args.tau is not None:
        cfg["tau"] = float(args.tau)

    if args.n_ctx is not None:
        cfg["n_ctx"] = int(args.n_ctx)
    if args.ctx_init is not None:
        cfg["ctx_init"] = str(args.ctx_init)
    if args.csc is not None:
        cfg["csc"] = bool(args.csc)
    if args.class_token_position is not None:
        cfg["class_token_position"] = str(args.class_token_position)
    if args.omega is not None:
        cfg["omega"] = float(args.omega)

    if args.topk is not None:
        cfg["topk"] = int(args.topk)

    if args.stage2_dhno_mode is not None:
        cfg["stage2_dhno_mode"] = str(args.stage2_dhno_mode)
    if args.stage2_negative_sampling is not None:
        cfg["stage2_negative_sampling"] = str(args.stage2_negative_sampling)
    if args.stage2_loss is not None:
        cfg["stage2_loss"] = str(args.stage2_loss)

    eval_mode = _resolve_stage2_eval_mode(args.stage2_eval_mode)
    if eval_mode is not None:
        cfg["stage2_eval_mode"] = str(eval_mode)

    # Behavior toggles (CLI overrides yaml).
    if getattr(args, "stage1_apply_tau", None) is not None:
        cfg["stage1_apply_tau"] = bool(args.stage1_apply_tau)
    if getattr(args, "stage2_ce_apply_tau", None) is not None:
        cfg["stage2_ce_apply_tau"] = bool(args.stage2_ce_apply_tau)
    if getattr(args, "eval_apply_tau_in_softmax", None) is not None:
        cfg["eval_apply_tau_in_softmax"] = bool(args.eval_apply_tau_in_softmax)
    if getattr(args, "use_lr_scheduler", None) is not None:
        cfg["use_lr_scheduler"] = bool(args.use_lr_scheduler)

    cfg["monitor_metric"] = str(args.best_metric)
    return cfg


def _validate_runtime_args(args: argparse.Namespace, backbone_cfg: Dict[str, Any]) -> None:
    if args.skip_stage1 and args.skip_stage2:
        raise ValueError("Nothing to do: both --skip-stage1 and --skip-stage2 are set.")

    if (args.fixed_train_csv is None) ^ (args.fixed_test_csv is None):
        raise ValueError("--fixed-train-csv and --fixed-test-csv must be provided together.")

    if args.few_shot_k <= 0:
        raise ValueError(f"--few-shot-k must be > 0, got {args.few_shot_k}")

    for key in ("stage1_epochs", "stage2_epochs", "batch_size", "patience", "topk"):
        if key in backbone_cfg and int(backbone_cfg[key]) <= 0:
            raise ValueError(f"{key} must be > 0, got {backbone_cfg[key]}")
    for key in ("stage1_lr", "stage2_lr", "weight_decay", "tau"):
        if key in backbone_cfg and float(backbone_cfg[key]) < 0:
            raise ValueError(f"{key} must be >= 0, got {backbone_cfg[key]}")


def _select_device(device_arg: Optional[str]) -> torch.device:
    import torch

    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_stage2_init(args: argparse.Namespace) -> Tuple[bool, str]:
    skip_stage1 = bool(args.skip_stage1)
    stage2_init = args.stage2_init or "cpi"

    if args.no_initial_checkpoint:
        skip_stage1 = True
        if args.stage2_init is None:
            stage2_init = "random"

    return skip_stage1, stage2_init


def _evaluate_with_omega(
    model: Any,
    val_loader: Any,
    num_classes: int,
    tau: float,
    omega: float,
    device: Any,
    apply_tau_in_softmax: bool = False,
) -> Dict[str, float]:
    import torch
    from sldpc.utils.metrics import compute_classification_metrics

    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    model.eval()
    for batch in val_loader:
        feats = batch["feat"].to(device)
        labels = batch["label"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)

        logits = model(feats, mode="fused", omega=float(omega))
        if apply_tau_in_softmax:
            probs = torch.softmax(logits / float(tau), dim=-1)
        else:
            # Legacy behavior (softmax over raw cosine). Does not change
            # AUC rankings or argmax-based ACC/F1; kept consistent with
            # _validate() to avoid surprises in omega sweeps.
            probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    y_prob = torch.cat(all_probs, dim=0)
    y_true = torch.cat(all_labels, dim=0)
    return compute_classification_metrics(y_true, y_prob, num_classes=num_classes)


def _search_best_omega(
    model: PromptedTitan,
    val_loader: DataLoader,
    num_classes: int,
    tau: float,
    metric_name: str,
    omega_values: Sequence[float],
    device: torch.device,
    apply_tau_in_softmax: bool = False,
) -> Dict[str, Any]:
    import torch

    metric_name = metric_name.upper()
    scored: List[Dict[str, Any]] = []

    for om in omega_values:
        with torch.no_grad():
            metrics = _evaluate_with_omega(
                model=model,
                val_loader=val_loader,
                num_classes=num_classes,
                tau=tau,
                omega=float(om),
                device=device,
                apply_tau_in_softmax=apply_tau_in_softmax,
            )
        scored.append({"omega": float(om), "metrics": metrics, "score": float(metrics.get(metric_name, float("nan")))})

    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    best = scored_sorted[0]
    return {
        "metric": metric_name,
        "best_omega": best["omega"],
        "best_metrics": best["metrics"],
        "all": scored,
    }


def _dump_final_report(
    output_dir: Path,
    *,
    seed: int,
    stage1_metrics: Optional[Dict[str, float]],
    stage2_metrics: Optional[Dict[str, float]],
    zero_shot_metrics: Optional[Dict[str, float]],
    dynamic_omega: Optional[Dict[str, Any]],
    args_snapshot: Dict[str, Any],
) -> None:
    report = {
        "seed": seed,
        "zero_shot": zero_shot_metrics,
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "dynamic_omega": dynamic_omega,
        "args": args_snapshot,
    }
    with (output_dir / "final_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _log_split_summary(
    logger: logging.Logger,
    seed: int,
    split_dir: Path,
    requested_k: int,
    *,
    fixed_csv: bool,
) -> None:
    """Read the three split CSVs and log per-class counts + pool sizes."""
    from collections import Counter

    def _read(p: Path):
        rows = []
        if not p.exists():
            return rows
        import csv as _csv
        with p.open("r", encoding="utf-8", newline="") as f:
            rows = [tuple(r) for r in _csv.reader(f) if r]
        return rows

    pool = _read(split_dir / "train_pool.csv")
    test = _read(split_dir / "test.csv")
    kshot = _read(split_dir / "kshot.csv")

    mode_label = "FIXED CSV pool" if fixed_csv else "AUTO 20/80 split"
    logger.info(
        "Split (%s) | seed=%s | pool=%d  test=%d  kshot=%d  (requested K=%d)",
        mode_label, seed, len(pool), len(test), len(kshot), requested_k,
    )
    kshot_by_class = Counter(c for _, c in kshot)
    pool_by_class = Counter(c for _, c in pool)
    for cls in sorted(pool_by_class):
        logger.info(
            "    class %-8s | pool=%4d | kshot=%3d",
            cls, pool_by_class[cls], kshot_by_class.get(cls, 0),
        )


def _log_stage2_loss_path(
    logger: logging.Logger,
    cfg: "TrainingConfig",  # noqa: F821  (forward ref)
) -> None:
    """Explain what loss the Stage-2 batches will actually minimize.

    This disambiguates the interplay of ``dhno_mode`` and ``stage2_loss``
    for anyone reading the log to understand whether tau is in effect.
    """
    mode = cfg.dhno_mode
    loss_family = cfg.stage2_loss
    sampling = cfg.negative_sampling

    if mode == "none":
        loss = f"CE({'cos/tau' if cfg.stage2_ce_apply_tau else 'cos'}, labels)  [tau {'USED' if cfg.stage2_ce_apply_tau else 'NOT used'}]"
        sampler_desc = "random negatives"
    elif mode == "sampling_only":
        loss = f"CE({'cos/tau' if cfg.stage2_ce_apply_tau else 'cos'}, labels)  [tau {'USED' if cfg.stage2_ce_apply_tau else 'NOT used'}]"
        sampler_desc = f"{sampling.upper()} hard negatives (top-{cfg.topk})"
    elif mode == "full":
        family_map = {
            "symmetric": "symmetric InfoNCE (i2t + t2i) / tau  [tau USED]",
            "i2t": "InfoNCE image->text / tau  [tau USED]",
            "t2i": "InfoNCE text->image / tau  [tau USED]",
            "ce": f"CE({'cos/tau' if cfg.stage2_ce_apply_tau else 'cos'}, labels)  [tau {'USED' if cfg.stage2_ce_apply_tau else 'NOT used'}]",
        }
        loss = family_map.get(loss_family, loss_family)
        sampler_desc = f"{sampling.upper()} hard negatives (top-{cfg.topk})"
    else:
        loss = f"(unknown dhno_mode={mode!r})"
        sampler_desc = sampling

    logger.info("Stage-2 sampler  : %s", sampler_desc)
    logger.info("Stage-2 loss path: %s", loss)


# Process-level cache for zero-shot weights. Keyed by (hf_model_id,
# tuple-of-tuples of class prompts). Zero-shot weights only depend on
# the TITAN backbone and the prompts yaml, NOT on the seed or the
# split, so we compute them once and reuse across seeds.
_ZERO_SHOT_CACHE: Dict[Any, Any] = {}


def _resolve_zero_shot_prompts_yaml(args: argparse.Namespace) -> Path:
    """Resolve the prompts yaml path (CLI override > convention)."""
    if args.zero_shot_prompts_yaml:
        return Path(args.zero_shot_prompts_yaml).resolve()
    candidate = (
        _ROOT
        / "data" / "datasets" / "zero_shot_prompts"
        / f"{args.dataset}.yaml"
    )
    return candidate


def _get_or_build_zs_weights(
    titan: Any,
    class_prompts_list: List[List[str]],
    device: Any,
    cache_key: Any,
):
    """Compute zero-shot weights once per process for a given prompts set."""
    cached = _ZERO_SHOT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from sldpc.utils.zero_shot import build_zero_shot_weights
    weights = build_zero_shot_weights(titan, class_prompts_list, device)
    _ZERO_SHOT_CACHE[cache_key] = weights
    return weights


def _run_one_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    output_dir: Path,
    dataset_cfg: Dict[str, Any],
    backbone_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    logger = logging.getLogger(f"titan.seed.{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    final_report = output_dir / "final_report.json"
    if args.skip_existing and final_report.exists():
        logger.info("Skip existing run: %s", final_report)
        return {"seed": seed, "status": "skipped", "run_dir": str(output_dir)}

    if args.dry_run:
        logger.info("[DRY-RUN] seed=%d output_dir=%s", seed, output_dir)
        return {
            "seed": seed,
            "status": "dry_run",
            "run_dir": str(output_dir),
            "config": str(args.config),
            "features_path": str(args.features_path),
            "fixed_train_csv": args.fixed_train_csv,
            "fixed_test_csv": args.fixed_test_csv,
        }

    import torch
    from torch.utils.data import DataLoader

    from sldpc.backbones import get_backbone
    from sldpc.backbones.titan import PromptedTitan
    from sldpc.data import SlideFeatureDataset
    from sldpc.trainers.base_trainer import TrainingConfig
    from sldpc.trainers.stage1_trainer import Stage1Trainer
    from sldpc.trainers.stage2_trainer import Stage2Trainer, build_feature_bank
    from sldpc.utils.seed import set_seed

    if args.mode is not None or args.eval_prompt or args.eval_baseline:
        logger.info("Legacy stage1 mode flags detected; unified two-stage pipeline is used.")

    if args.few_shot_indices_dir or args.few_shot_seed_dir or args.save_few_shot_indices:
        logger.info("Legacy few-shot-index flags detected; split generation is handled by unified splitter.")

    set_seed(seed)

    class_names: List[str] = list(dataset_cfg["class_names"])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    io_cfg_path = _ROOT / "configs" / "datasets" / "data_io_config.yaml"
    io_cfg = _load_yaml_if_exists(io_cfg_path)
    dataset_name = str(dataset_cfg.get("name", "")).strip()
    ds_io_cfg = dict((io_cfg.get("datasets", {}) or {}).get(dataset_name, {}) or {})
    if not dataset_cfg.get("all_slides_csv") and ds_io_cfg.get("all_slides_csv"):
        dataset_cfg = dict(dataset_cfg)
        dataset_cfg["all_slides_csv"] = ds_io_cfg["all_slides_csv"]
    csv_schema = dict(ds_io_cfg.get("csv_schema", {}) or {})

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    _prepare_splits(
        dataset_cfg=dataset_cfg,
        features_path=_resolve_path(args.features_path),
        source_type=args.source_type,
        class_names=class_names,
        few_shot_k=args.few_shot_k,
        seed=seed,
        split_dir=split_dir,
        fixed_train_csv=args.fixed_train_csv,
        fixed_test_csv=args.fixed_test_csv,
        csv_schema=csv_schema,
    )

    # Log split sizes + per-class K-shot distribution so we can verify
    # at a glance whether sampling did what we expected.
    _log_split_summary(logger, seed, split_dir, args.few_shot_k, fixed_csv=bool(args.fixed_train_csv))

    train_ds = SlideFeatureDataset(
        source_type=args.source_type,
        features_path=str(_resolve_path(args.features_path)),
        csv_path=split_dir / "kshot.csv",
        class_to_idx=class_to_idx,
        class_names=class_names,
    )
    test_ds = SlideFeatureDataset(
        source_type=args.source_type,
        features_path=str(_resolve_path(args.features_path)),
        csv_path=split_dir / "test.csv",
        class_to_idx=class_to_idx,
        class_names=class_names,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Training split is empty after split generation.")
    if len(test_ds) == 0:
        raise RuntimeError("Test split is empty after split generation.")

    train_bs = min(int(backbone_cfg.get("batch_size", 4)), len(train_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=int(backbone_cfg.get("num_workers", 0)),
        drop_last=True if len(train_ds) > train_bs else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(8, train_bs * 2),
        shuffle=False,
        num_workers=int(backbone_cfg.get("num_workers", 0)),
    )

    prompt_kwargs = {
        "n_ctx": int(backbone_cfg.get("n_ctx", 8)),
        "ctx_init": backbone_cfg.get("ctx_init"),
        "csc": bool(backbone_cfg.get("csc", False)),
        "class_token_position": backbone_cfg.get("class_token_position", "end"),
        "omega": float(backbone_cfg.get("omega", 0.8)),
    }

    # Log the prompt-learner kwargs — these are distinct from the
    # TrainingConfig block printed by each Stage*Trainer, and cover
    # n_ctx / ctx_init / csc / omega which live on the PromptLearner.
    from sldpc.utils.run_logging import print_config_block, print_divider
    print_config_block(
        logger,
        f"PromptLearner kwargs (seed={seed})",
        prompt_kwargs,
    )

    load_kwargs: Dict[str, Any] = {"hf_model_id": args.hf_model_id}
    if args.hf_revision:
        load_kwargs["revision"] = args.hf_revision
    if args.hf_local_files_only:
        load_kwargs["local_files_only"] = True

    titan, prompt_learner = get_backbone(
        "titan",
        classnames=class_names,
        device=device,
        prompt_kwargs=prompt_kwargs,
        load_kwargs=load_kwargs,
    )
    model = PromptedTitan(titan=titan, prompt_learner=prompt_learner).to(device)

    # ---- Zero-shot baseline (TITAN only, no checkpoints, no training) ----
    # Runs once per seed against the same test loader Stage-1/Stage-2 use.
    # Weights are cached at process level since they only depend on
    # (titan_model, prompts), not on seed.
    zero_shot_metrics: Optional[Dict[str, float]] = None
    if not args.skip_zero_shot:
        from sldpc.utils.zero_shot import (
            load_class_prompts, evaluate_zero_shot, TEMPLATES,
        )
        prompts_yaml = _resolve_zero_shot_prompts_yaml(args)
        if not prompts_yaml.exists():
            logger.warning(
                "Zero-shot prompts yaml not found at %s; skipping zero-shot baseline. "
                "Pass --zero-shot-prompts-yaml to override, or create the file.",
                prompts_yaml,
            )
        else:
            print_divider(logger, "ZERO-SHOT TITAN BASELINE")
            class_prompts_list = load_class_prompts(prompts_yaml, class_names)
            for cn, syns in zip(class_names, class_prompts_list):
                logger.info(
                    "  class %-10s : %d synonyms x %d templates = %d prompts",
                    cn, len(syns), len(TEMPLATES), len(syns) * len(TEMPLATES),
                )
            cache_key = (
                str(args.hf_model_id),
                tuple(tuple(syns) for syns in class_prompts_list),
            )
            weights = _get_or_build_zs_weights(titan, class_prompts_list, device, cache_key)
            zero_shot_metrics = evaluate_zero_shot(
                titan=titan,
                weights=weights,
                test_loader=test_loader,
                device=device,
                num_classes=num_classes,
            )
            logger.info(
                "Zero-shot done. ACC=%.2f  F1=%.2f  AUC=%.2f",
                float(zero_shot_metrics.get("ACC", float("nan"))),
                float(zero_shot_metrics.get("F1", float("nan"))),
                float(zero_shot_metrics.get("AUC", float("nan"))),
            )

    skip_stage1, stage2_init = _resolve_stage2_init(args)
    if skip_stage1 and stage2_init == "cpi":
        logger.warning("skip-stage1 with stage2-init=cpi is invalid; switching stage2-init to random.")
        stage2_init = "random"

    stage1_metrics: Optional[Dict[str, float]] = None
    stage2_metrics: Optional[Dict[str, float]] = None
    dynamic_omega: Optional[Dict[str, Any]] = None

    if not skip_stage1:
        print_divider(logger, "STAGE-1 (CoOp base prompt P)")
        stage1_cfg = TrainingConfig(
            lr=float(backbone_cfg.get("stage1_lr", 1e-3)),
            weight_decay=float(backbone_cfg.get("weight_decay", 0.0)),
            epochs=int(backbone_cfg.get("stage1_epochs", 50)),
            batch_size=train_bs,
            patience=int(backbone_cfg.get("patience", 100)),
            tau=float(backbone_cfg.get("tau", 0.07)),
            monitor_metric=str(backbone_cfg.get("monitor_metric", "F1")),
            output_dir=str(output_dir / "stage1"),
            # Tau-scale toggles (legacy-aligned defaults).
            stage1_apply_tau=bool(backbone_cfg.get("stage1_apply_tau", False)),
            eval_apply_tau_in_softmax=bool(backbone_cfg.get("eval_apply_tau_in_softmax", False)),
            # Optional scheduler (default off).
            use_lr_scheduler=bool(backbone_cfg.get("use_lr_scheduler", False)),
            scheduler_factor=float(backbone_cfg.get("scheduler_factor", 0.1)),
            scheduler_patience=int(backbone_cfg.get("scheduler_patience", 8)),
        )
        logger.info(
            "Stage-1 loss path : CE(%s, labels)  [tau %s]",
            "cos/tau" if stage1_cfg.stage1_apply_tau else "cos",
            "USED" if stage1_cfg.stage1_apply_tau else "NOT used (legacy)",
        )
        stage1 = Stage1Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            num_classes=num_classes,
            cfg=stage1_cfg,
            device=device,
        )
        stage1_metrics = stage1.fit()
        stage1.load_best_into_model()
        logger.info(
            "Stage-1 done. Best %s=%.4f  (loaded back into ctx_learnable).",
            stage1_cfg.monitor_metric,
            float(stage1_metrics.get(stage1_cfg.monitor_metric, float("nan"))),
        )

    if not args.skip_stage2:
        print_divider(logger, "STAGE-2 (dual-prompt P' with DHNO+SICL)")
        if stage2_init == "cpi":
            logger.info("Stage-2 init : CPI — clone ctx_learnable (Stage-1 best) -> ctx_frozen (P).")
            model.prompt_learner.clone_learnable_to_frozen()
        else:
            logger.info("Stage-2 init : RANDOM — re-init ctx_learnable from N(0, 0.02), then clone to ctx_frozen.")
            with torch.no_grad():
                torch.nn.init.normal_(model.prompt_learner.ctx_learnable, std=0.02)
                model.prompt_learner.ctx_frozen.data.copy_(model.prompt_learner.ctx_learnable.data)
                model.prompt_learner.ctx_frozen.requires_grad_(False)

        feat_bank, label_bank, class_to_indices = build_feature_bank(
            model=model,
            feature_dataset=train_ds,
            device=device,
        )

        requested_topk = int(backbone_cfg.get("topk", min(8, num_classes)))
        effective_topk = min(max(1, requested_topk), num_classes)
        if effective_topk != requested_topk:
            logger.info(
                "Stage-2 topk : requested=%d, clamped to num_classes=%d -> effective_topk=%d",
                requested_topk, num_classes, effective_topk,
            )

        stage2_cfg = TrainingConfig(
            lr=float(backbone_cfg.get("stage2_lr", 1e-3)),
            weight_decay=float(backbone_cfg.get("weight_decay", 0.0)),
            epochs=int(backbone_cfg.get("stage2_epochs", 50)),
            batch_size=train_bs,
            patience=int(backbone_cfg.get("patience", 100)),
            tau=float(backbone_cfg.get("tau", 0.07)),
            topk=effective_topk,
            dhno_mode=str(backbone_cfg.get("stage2_dhno_mode", "full")),
            negative_sampling=str(backbone_cfg.get("stage2_negative_sampling", "hns")),
            stage2_loss=str(backbone_cfg.get("stage2_loss", "symmetric")),
            monitor_metric=str(backbone_cfg.get("monitor_metric", "F1")),
            output_dir=str(output_dir / "stage2"),
            # Tau-scale toggles (legacy-aligned defaults).
            stage2_ce_apply_tau=bool(backbone_cfg.get("stage2_ce_apply_tau", False)),
            eval_apply_tau_in_softmax=bool(backbone_cfg.get("eval_apply_tau_in_softmax", False)),
            # Optional scheduler (default off).
            use_lr_scheduler=bool(backbone_cfg.get("use_lr_scheduler", False)),
            scheduler_factor=float(backbone_cfg.get("scheduler_factor", 0.1)),
            scheduler_patience=int(backbone_cfg.get("scheduler_patience", 8)),
        )
        _log_stage2_loss_path(logger, stage2_cfg)

        eval_mode = str(backbone_cfg.get("stage2_eval_mode", "fused"))
        stage2 = Stage2Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            num_classes=num_classes,
            cfg=stage2_cfg,
            device=device,
            feature_bank=feat_bank,
            label_bank=label_bank,
            class_to_indices=class_to_indices,
            rng_seed=seed,
            eval_mode=eval_mode,
        )
        stage2_metrics = stage2.fit()
        stage2.load_best_into_model()

        if args.dynamic_omega_b:
            dynamic_omega = _search_best_omega(
                model=model,
                val_loader=test_loader,
                num_classes=num_classes,
                tau=float(backbone_cfg.get("tau", 0.07)),
                metric_name=str(backbone_cfg.get("monitor_metric", "F1")),
                omega_values=[float(v) for v in args.omega_search_values],
                device=device,
                apply_tau_in_softmax=bool(backbone_cfg.get("eval_apply_tau_in_softmax", False)),
            )

    _dump_final_report(
        output_dir,
        seed=seed,
        stage1_metrics=stage1_metrics,
        stage2_metrics=stage2_metrics,
        zero_shot_metrics=zero_shot_metrics,
        dynamic_omega=dynamic_omega,
        args_snapshot={
            "dataset": args.dataset,
            "config": str(args.config),
            "backbone_config": str(args.backbone_config),
            "features_path": str(args.features_path),
            "fixed_train_csv": args.fixed_train_csv,
            "fixed_test_csv": args.fixed_test_csv,
            "few_shot_k": int(args.few_shot_k),
            "best_metric": str(backbone_cfg.get("monitor_metric", "F1")),
            "skip_stage1": skip_stage1,
            "skip_stage2": bool(args.skip_stage2),
            "skip_zero_shot": bool(args.skip_zero_shot),
            "stage2_init": stage2_init,
        },
    )

    return {
        "seed": seed,
        "status": "ok",
        "run_dir": str(output_dir),
        "zero_shot": zero_shot_metrics,
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "dynamic_omega": dynamic_omega,
    }


def _summarize_numeric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "n": 1}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
        "n": len(values),
    }


def _build_seed_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _aggregate(stage_key: str) -> Dict[str, Dict[str, float]]:
        ok = [r for r in results
              if r.get("status") == "ok" and isinstance(r.get(stage_key), dict)]
        out: Dict[str, Dict[str, float]] = {}
        for metric in ("ACC", "F1", "AUC"):
            vals: List[float] = []
            for r in ok:
                v = r[stage_key].get(metric)
                if v is not None:
                    vals.append(float(v))
            if vals:
                out[metric] = _summarize_numeric(vals)
        return out

    dyn_ok = [r for r in results if r.get("status") == "ok" and isinstance(r.get("dynamic_omega"), dict)]
    dynamic_summary: Dict[str, Any] = {}
    if dyn_ok:
        best_omegas = [float(r["dynamic_omega"].get("best_omega", 0.0)) for r in dyn_ok]
        dynamic_summary["best_omega"] = _summarize_numeric(best_omegas)

    return {
        "n_total": len(results),
        "n_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_failed": sum(1 for r in results if r.get("status") == "failed"),
        "n_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "n_dry_run": sum(1 for r in results if r.get("status") == "dry_run"),
        "zero_shot": _aggregate("zero_shot"),
        "stage1": _aggregate("stage1"),
        "stage2": _aggregate("stage2"),
        "dynamic_omega": dynamic_summary,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run() -> None:
    args = _parse_args()

    # Resolve all paths and CLI/yaml merge BEFORE touching the logger,
    # so we have an output_dir to point the global logfile at.
    _resolve_dataset_inputs(args)

    args.config = str(_resolve_path(args.config))
    args.backbone_config = str(_resolve_path(args.backbone_config))
    args.features_path = str(_resolve_path(args.features_path))
    if args.fixed_train_csv:
        args.fixed_train_csv = str(_resolve_path(args.fixed_train_csv))
    if args.fixed_test_csv:
        args.fixed_test_csv = str(_resolve_path(args.fixed_test_csv))

    dataset_cfg = _load_yaml(Path(args.config))
    base_backbone_cfg = _load_yaml(Path(args.backbone_config))
    backbone_cfg = _build_backbone_cfg(base_backbone_cfg, args)
    _validate_runtime_args(args, backbone_cfg)

    seeds = _resolve_seeds(args)
    device = "dry-run" if args.dry_run else _select_device(args.device)

    output_root = _resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Set up root logging (stdout + global run.log) ------------------
    from sldpc.utils.run_logging import (
        setup_run_logging, push_seed_logfile, pop_seed_logfile,
        print_banner, print_config_block, print_divider,
    )
    setup_run_logging(output_root, level=args.log_level)
    logger = logging.getLogger("titan_pipeline")

    # --- Run-level banner ----------------------------------------------
    split_mode = (
        "FIXED train/test CSV (pool fixed across seeds; only K-shot varies)"
        if args.fixed_train_csv and args.fixed_test_csv
        else "AUTO 20/80 stratified split per seed"
    )
    print_banner(
        logger,
        title=f"SLDPC-TITAN  |  dataset={dataset_cfg.get('name', args.dataset)}  |  K-shot={args.few_shot_k}",
        subtitle=f"split={split_mode}  |  n_seeds={len(seeds)}  |  device={device}",
    )
    logger.info("Seeds          : %s", seeds)
    logger.info("Dataset yaml   : %s", args.config)
    logger.info("Backbone yaml  : %s", args.backbone_config)
    logger.info("Features path  : %s", args.features_path)
    logger.info("Output dir     : %s", output_root)
    if args.fixed_train_csv:
        logger.info("Fixed train CSV: %s", args.fixed_train_csv)
        logger.info("Fixed test  CSV: %s", args.fixed_test_csv)

    # Full effective backbone config so every knob is traceable.
    print_config_block(logger, "Effective backbone config", backbone_cfg)

    results: List[Dict[str, Any]] = []
    multi_seed = len(seeds) > 1

    for seed_idx, seed in enumerate(seeds, start=1):
        run_dir = output_root / f"seed_{seed}" if multi_seed else output_root
        run_dir.mkdir(parents=True, exist_ok=True)

        # Open a per-seed log file so each seed has a self-contained trail.
        push_seed_logfile(run_dir)
        try:
            print_banner(
                logger,
                title=f"SEED [{seed_idx}/{len(seeds)}]  seed={seed}",
                subtitle=f"run_dir={run_dir}",
            )
            res = _run_one_seed(
                args,
                seed=seed,
                output_dir=run_dir,
                dataset_cfg=dataset_cfg,
                backbone_cfg=backbone_cfg,
                device=device,
            )
            # Per-seed final summary line.
            _log_seed_summary(logger, seed, res, monitor=backbone_cfg.get("monitor_metric", "F1"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Seed %s failed", seed)
            res = {
                "seed": seed,
                "status": "failed",
                "run_dir": str(run_dir),
                "error": str(exc),
            }
        finally:
            pop_seed_logfile()
        results.append(res)

    summary = {
        "dataset": dataset_cfg.get("name"),
        "config": args.config,
        "backbone_config": args.backbone_config,
        "features_path": args.features_path,
        "fixed_train_csv": args.fixed_train_csv,
        "fixed_test_csv": args.fixed_test_csv,
        "split_mode": "fixed_csv" if args.fixed_train_csv else "auto_split",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "best_metric": backbone_cfg.get("monitor_metric", "F1"),
        "results": results,
        "aggregate": _build_seed_summary(results),
    }

    _write_json(output_root / "seed_summary.json", summary)

    # --- Run-level outro ------------------------------------------------
    print_divider(logger, "AGGREGATE")
    agg = summary["aggregate"]
    if agg:
        for stage_key in ("zero_shot", "stage1", "stage2"):
            if stage_key in agg and agg[stage_key]:
                parts = []
                for metric, stats in sorted(agg[stage_key].items()):
                    if isinstance(stats, dict) and "mean" in stats:
                        parts.append(f"{metric}={stats['mean']:.2f}±{stats.get('std', 0):.2f}")
                if parts:
                    logger.info("  %-9s  %s", stage_key, "  ".join(parts))
    logger.info("Summary written to %s", output_root / "seed_summary.json")

    failed = [r for r in results if r.get("status") == "failed"]
    if failed and not args.dry_run:
        raise SystemExit(1)


def _log_seed_summary(
    logger: logging.Logger,
    seed: int,
    res: Dict[str, Any],
    monitor: str = "F1",
) -> None:
    """Print a one-liner summary of this seed's best metrics."""
    def _fmt_stage(stage_key: str) -> str:
        m = res.get(stage_key) or {}
        if not m or not isinstance(m, dict):
            return f"{stage_key}=(skipped)"
        parts = []
        for mk in ("AUC", "F1", "ACC"):
            if mk in m:
                parts.append(f"{mk}={m[mk]:.2f}")
        return f"{stage_key}[{' '.join(parts)}]" if parts else f"{stage_key}=(no metrics)"
    logger.info(
        "SEED %s DONE  status=%s  best_on=%s   %s   %s   %s",
        seed,
        res.get("status", "ok"),
        monitor,
        _fmt_stage("zero_shot"),
        _fmt_stage("stage1"),
        _fmt_stage("stage2"),
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()