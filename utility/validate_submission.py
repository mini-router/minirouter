#!/usr/bin/env python3
"""CLI wrapper: validate submissions/final_model (or --dir) before opening a PR.

Usage::

    python utility/validate_submission.py
    python utility/validate_submission.py --dir submissions/final_model
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.validate_submission import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
