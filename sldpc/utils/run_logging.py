"""Unified run-logging utilities for SLDPC TITAN pipeline.

Provides three things:

1. :func:`setup_run_logging` — attach a rotating-free file handler to
   the root logger so **everything** that gets logged (including
   sub-library INFO/DEBUG) is persisted to ``<output_dir>/run.log``.
   Safe to call multiple times; per-seed file handlers are also
   supported via :func:`push_seed_logfile` / :func:`pop_seed_logfile`.

2. :func:`print_banner` — pretty fixed-width banners for stdout/log.

3. :func:`print_config_block` — dump a ``dict`` to the log with keys
   sorted and long values truncated, so every run records the full
   effective configuration at a glance.

Why a custom module (not just ``logging.basicConfig``):
- The pipeline creates one top-level logger per run AND one nested
  logfile per seed. ``basicConfig`` only configures the root once and
  cannot cleanly install/remove per-seed handlers.
- We want the same stream-handler format across libraries, including
  third-party (torchvision, transformers) loggers.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# --- Formatters --------------------------------------------------------------

_STREAM_FMT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
_FILE_FMT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _make_formatter(fmt: str = _STREAM_FMT) -> logging.Formatter:
    return logging.Formatter(fmt=fmt, datefmt=_DATEFMT)


# --- Root configuration ------------------------------------------------------

_GLOBAL_FILE_HANDLER: Optional[logging.FileHandler] = None
_STAGE_HANDLER_STACK: list = []          # seed-scoped file handlers


def setup_run_logging(
    output_dir: Path,
    level: str = "INFO",
    *,
    filename: str = "run.log",
) -> Path:
    """Configure root logger for this run.

    - Ensures a single StreamHandler on stdout with the canonical format.
    - Attaches a global FileHandler at ``output_dir/filename``. Any
      previously-attached global FileHandler (from an earlier call in
      the same process) is removed and closed cleanly.

    Returns the path to the log file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / filename

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 1) StreamHandler: install exactly one on stdout with our format.
    has_stream = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and getattr(h, "stream", None) in (sys.stdout, sys.stderr)
        for h in root.handlers
    )
    if not has_stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(_make_formatter(_STREAM_FMT))
        root.addHandler(sh)
    else:
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setFormatter(_make_formatter(_STREAM_FMT))

    # 2) Global FileHandler: replace any previous one.
    global _GLOBAL_FILE_HANDLER
    if _GLOBAL_FILE_HANDLER is not None:
        root.removeHandler(_GLOBAL_FILE_HANDLER)
        try:
            _GLOBAL_FILE_HANDLER.close()
        except Exception:
            pass

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(_make_formatter(_FILE_FMT))
    root.addHandler(fh)
    _GLOBAL_FILE_HANDLER = fh

    return log_path


def push_seed_logfile(output_dir: Path, *, filename: str = "run.log") -> Path:
    """Attach an extra FileHandler for a per-seed directory.

    The handler is pushed onto an internal stack; call
    :func:`pop_seed_logfile` when the seed finishes so subsequent seeds
    don't also write into this file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / filename

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(_make_formatter(_FILE_FMT))

    root = logging.getLogger()
    root.addHandler(fh)
    _STAGE_HANDLER_STACK.append(fh)
    return log_path


def pop_seed_logfile() -> None:
    """Remove the most recently pushed per-seed FileHandler."""
    if not _STAGE_HANDLER_STACK:
        return
    fh = _STAGE_HANDLER_STACK.pop()
    root = logging.getLogger()
    root.removeHandler(fh)
    try:
        fh.close()
    except Exception:
        pass


# --- Banners & config blocks ------------------------------------------------

_BANNER_WIDTH = 78


def print_banner(logger: logging.Logger, title: str, subtitle: str = "") -> None:
    """Emit a fixed-width banner:
        ============================================================
          TITLE
          subtitle
        ============================================================
    """
    bar = "=" * _BANNER_WIDTH
    logger.info(bar)
    logger.info("  %s", title)
    if subtitle:
        logger.info("  %s", subtitle)
    logger.info(bar)


def print_divider(logger: logging.Logger, title: str = "") -> None:
    if title:
        fill = max(2, _BANNER_WIDTH - len(title) - 4)
        logger.info("-- %s %s", title, "-" * fill)
    else:
        logger.info("-" * _BANNER_WIDTH)


def print_config_block(
    logger: logging.Logger,
    title: str,
    cfg: Dict[str, Any],
    *,
    max_value_len: int = 120,
) -> None:
    """Log a dict as sorted ``key: value`` lines.

    Non-scalar values (lists, dicts) are JSON-serialized and truncated
    at ``max_value_len`` chars to keep the log readable.
    """
    print_divider(logger, title)
    if not cfg:
        logger.info("  (empty)")
        return

    def _fmt(v: Any) -> str:
        if isinstance(v, (str, int, float, bool)) or v is None:
            s = str(v)
        else:
            try:
                s = json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                s = repr(v)
        if len(s) > max_value_len:
            s = s[: max_value_len - 3] + "..."
        return s

    keys = sorted(cfg.keys())
    width = max(len(k) for k in keys)
    for k in keys:
        logger.info("  %-*s : %s", width, k, _fmt(cfg[k]))
