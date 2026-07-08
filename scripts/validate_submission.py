#!/usr/bin/env python3
"""Validate a miner submission bundle before packaging / opening a PR.

Zero-network, no API keys, no GPU. Checks that ``submissions/final_model/`` (or a
``--dir``) is a complete, well-formed bundle so mistakes are caught locally instead
of later in the validator backend / PR-automation queue (see ``CONTRIBUTOIN.md``).

Checks:
  * required files present: ``best_theta.npy``, ``summary.json``
  * optional files reported if present: ``history.json``, ``eval.json``
  * ``best_theta.npy`` loads as a finite float vector whose length matches the
    coordinator ``ParamSpec.n_total`` (13312 by default = 6*1024 head + 7168 SVF)
  * ``summary.json`` is valid JSON; its useful keys are sanity-checked and a
    ``summary["n_total"]`` that disagrees with the theta length is warned about

Exit code: ``0`` if the bundle is valid (warnings allowed), ``1`` on any error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.coordinator.params import make_spec  # noqa: E402

REQUIRED_FILES: tuple[str, ...] = ("best_theta.npy", "summary.json")
OPTIONAL_FILES: tuple[str, ...] = ("history.json", "eval.json")
SUMMARY_EXPECTED_KEYS: tuple[str, ...] = ("benchmark", "pool", "n_total", "best_fitness")


def validate_bundle(
    directory: Path,
    expected_n_total: int | None = None,
) -> tuple[list[str], list[str], dict]:
    """Validate a submission-bundle directory.

    Returns ``(errors, warnings, info)``. An empty ``errors`` list means the bundle
    is valid. Expected failure modes (missing files, corrupt ``.npy``, bad JSON) are
    reported as strings rather than raised, so a single pass surfaces every problem.

    Args:
        directory: Path to the bundle (e.g. ``submissions/final_model``).
        expected_n_total: Expected theta length; defaults to the coordinator's
            ``ParamSpec.n_total`` (13312).
    """
    errors: list[str] = []
    warnings: list[str] = []
    info: dict = {}

    if expected_n_total is None:
        expected_n_total = make_spec().n_total
    info["expected_n_total"] = expected_n_total

    if not directory.is_dir():
        errors.append(f"submission directory not found: {directory}")
        return errors, warnings, info

    present = {p.name for p in directory.iterdir() if p.is_file()}
    for name in REQUIRED_FILES:
        if name not in present:
            errors.append(f"missing required file: {name}")
    info["optional_present"] = [n for n in OPTIONAL_FILES if n in present]

    theta_len = _check_theta(directory / "best_theta.npy", expected_n_total, errors, warnings, info)
    _check_summary(directory / "summary.json", theta_len, errors, warnings, info)

    return errors, warnings, info


def _check_theta(
    path: Path,
    expected_n_total: int,
    errors: list[str],
    warnings: list[str],
    info: dict,
) -> int | None:
    """Validate ``best_theta.npy``; record its length in ``info`` and return it."""
    if not path.is_file():
        return None  # already reported as a missing required file
    try:
        theta = np.load(path, allow_pickle=False)
    except Exception as exc:  # corrupt, pickled, or not an array
        errors.append(f"best_theta.npy failed to load: {exc}")
        return None

    arr = np.asarray(theta)
    if arr.ndim != 1:
        warnings.append(
            f"best_theta.npy is {arr.shape}; expected a 1-D vector (flattened for the length check)"
        )
    flat = arr.reshape(-1)
    theta_len = int(flat.size)
    info["theta_len"] = theta_len

    if not np.issubdtype(arr.dtype, np.floating):
        errors.append(f"best_theta.npy must be a float array, got dtype {arr.dtype}")
    elif not np.all(np.isfinite(flat)):
        n_bad = int((~np.isfinite(flat)).sum())
        errors.append(f"best_theta.npy contains {n_bad} non-finite value(s) (NaN/Inf)")

    if theta_len != expected_n_total:
        errors.append(
            f"best_theta.npy has length {theta_len}, expected n_total={expected_n_total}"
        )
    return theta_len


def _check_summary(
    path: Path,
    theta_len: int | None,
    errors: list[str],
    warnings: list[str],
    info: dict,
) -> None:
    """Validate ``summary.json``: valid JSON object, expected keys, n_total agreement."""
    if not path.is_file():
        return  # already reported as a missing required file
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        errors.append(f"summary.json is not valid JSON: {exc}")
        return
    if not isinstance(summary, dict):
        errors.append(f"summary.json must be a JSON object, got {type(summary).__name__}")
        return

    missing = [k for k in SUMMARY_EXPECTED_KEYS if k not in summary]
    if missing:
        warnings.append(f"summary.json missing useful key(s): {', '.join(missing)}")
    info["benchmark"] = summary.get("benchmark")
    info["pool"] = summary.get("pool")

    s_n_total = summary.get("n_total")
    if s_n_total is not None and theta_len is not None and s_n_total != theta_len:
        warnings.append(
            f"summary.json n_total={s_n_total} disagrees with best_theta.npy length {theta_len}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate a miner submission bundle offline (before opening a PR)."
    )
    ap.add_argument(
        "--dir",
        default=str(_REPO / "submissions" / "final_model"),
        help="bundle directory (default: submissions/final_model)",
    )
    ap.add_argument(
        "--n-total",
        type=int,
        default=None,
        help="override the expected theta length (default: coordinator ParamSpec.n_total)",
    )
    args = ap.parse_args(argv)

    directory = Path(args.dir)
    errors, warnings, info = validate_bundle(directory, expected_n_total=args.n_total)

    for w in warnings:
        print(f"[warn] {w}")
    if errors:
        for e in errors:
            print(f"[error] {e}")
        print(f"\nFAIL: {directory} is not a valid submission bundle ({len(errors)} error(s)).")
        return 1

    optional = info.get("optional_present") or []
    print(f"OK: {directory} is a valid submission bundle.")
    print(f"  theta length : {info.get('theta_len')} (expected {info.get('expected_n_total')})")
    print(f"  benchmark    : {info.get('benchmark')}")
    print(f"  pool         : {info.get('pool')}")
    print(f"  optional     : {', '.join(optional) if optional else '(none)'}")
    if warnings:
        print(f"  warnings     : {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
