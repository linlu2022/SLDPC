#!/usr/bin/env python
"""Thin CLI entrypoint for TITAN experiments."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sldpc.trainers.titan_pipeline import main


if __name__ == "__main__":
    main()
