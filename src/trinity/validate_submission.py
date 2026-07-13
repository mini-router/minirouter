"""Validate a miner submission bundle before opening a PR.

Checks that ``submissions/final_model/`` (or another directory) contains the
required artifacts with a coherent ``best_theta.npy`` / ``summary.json`` layout.
No network calls; safe to run offline without API keys.

Example::

    python utility/validate_submission.py --dir submissions/final_model
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trinity.coordinator import params as P

REQUIRED_FILES = ("best_theta.npy", "summary.json")
OPTIONAL_FILES = ("history.json", "eval.json")
SUMMARY_USEFUL_KEYS = ("benchmark", "n_total", "best_fitness", "pool")


@dataclass
class ValidationResult:
    """Outcome of validating one submission directory."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.ok = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        self.info.append(message)


def _check_required_files(bundle_dir: Path, result: ValidationResult) -> None:
    for name in REQUIRED_FILES:
        path = bundle_dir / name
        if not path.is_file():
            result.add_error(f"missing required file: {name}")
        elif path.stat().st_size == 0:
            result.add_error(f"required file is empty: {name}")


def _check_optional_files(bundle_dir: Path, result: ValidationResult) -> None:
    for name in OPTIONAL_FILES:
        path = bundle_dir / name
        if path.is_file():
            result.add_info(f"optional file present: {name}")
        else:
            result.add_info(f"optional file absent: {name}")


def _load_theta(path: Path, result: ValidationResult) -> np.ndarray | None:
    try:
        theta = np.load(path)
    except Exception as exc:  # noqa: BLE001 — surface any load failure to the miner
        result.add_error(f"best_theta.npy could not be loaded: {exc}")
        return None

    if not isinstance(theta, np.ndarray):
        result.add_error("best_theta.npy did not contain a numpy array")
        return None
    if theta.ndim != 1:
        result.add_error(
            f"best_theta.npy must be a 1-D vector, got shape {tuple(theta.shape)}"
        )
        return None
    if theta.size == 0:
        result.add_error("best_theta.npy is empty")
        return None
    if not np.issubdtype(theta.dtype, np.floating):
        result.add_error(
            f"best_theta.npy must be floating-point, got dtype {theta.dtype}"
        )
        return None
    if not np.isfinite(theta).all():
        result.add_error("best_theta.npy contains NaN or Inf values")
        return None
    return theta


def _load_summary(path: Path, result: ValidationResult) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        result.add_error(f"summary.json could not be parsed: {exc}")
        return None

    if not isinstance(data, dict):
        result.add_error("summary.json must be a JSON object")
        return None
    return data


def _check_optional_json(bundle_dir: Path, name: str, result: ValidationResult) -> None:
    path = bundle_dir / name
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        result.add_error(f"{name} is present but not valid JSON: {exc}")
        return
    if not isinstance(data, (dict, list)):
        result.add_warning(f"{name} parsed but is not an object/array")


def validate_bundle(
    bundle_dir: Path | str,
    *,
    expected_n_total: int | None = None,
) -> ValidationResult:
    """Validate a submission directory.

    Parameters
    ----------
    bundle_dir:
        Path to the bundle (typically ``submissions/final_model``).
    expected_n_total:
        Fallback θ length when the bundle does not declare one in
        ``summary.json``. Defaults to the canonical :func:`params.make_spec`
        ``n_total`` (13,312).

    Returns
    -------
    ValidationResult
        ``ok`` is True only when every hard check passed.
    """
    root = Path(bundle_dir)
    result = ValidationResult(ok=True)

    if not root.exists():
        result.add_error(f"bundle directory does not exist: {root}")
        return result
    if not root.is_dir():
        result.add_error(f"bundle path is not a directory: {root}")
        return result

    _check_required_files(root, result)
    _check_optional_files(root, result)

    theta_path = root / "best_theta.npy"
    summary_path = root / "summary.json"
    theta: np.ndarray | None = None
    summary: dict[str, Any] | None = None
    n_expected = expected_n_total if expected_n_total is not None else P.make_spec().n_total

    if summary_path.is_file() and summary_path.stat().st_size > 0:
        summary = _load_summary(summary_path, result)
        if summary is not None:
            summary_n_total: int | None = None
            present = [k for k in SUMMARY_USEFUL_KEYS if k in summary]
            missing = [k for k in SUMMARY_USEFUL_KEYS if k not in summary]
            if present:
                result.add_info(f"summary.json keys present: {', '.join(present)}")
            if missing:
                result.add_warning(
                    f"summary.json missing recommended keys: {', '.join(missing)}"
                )
            if "n_total" in summary:
                try:
                    summary_n_total = int(summary["n_total"])
                except (TypeError, ValueError):
                    result.add_warning("summary.json n_total is not an integer")
                else:
                    n_expected = summary_n_total
                    result.add_info(f"summary.json n_total={summary_n_total}")
            if "benchmark" in summary:
                result.add_info(f"benchmark={summary['benchmark']!r}")
            if "best_fitness" in summary:
                result.add_info(f"best_fitness={summary['best_fitness']}")

    if theta_path.is_file() and theta_path.stat().st_size > 0:
        theta = _load_theta(theta_path, result)
        if theta is not None:
            result.add_info(
                f"best_theta.npy shape={theta.shape} dtype={theta.dtype}"
            )
            if theta.size != n_expected:
                result.add_warning(
                    f"best_theta.npy length {theta.size} != expected n_total={n_expected}"
                )
            else:
                result.add_info(f"theta length matches n_total={n_expected}")

    if theta is not None and summary is not None and "n_total" in summary:
        try:
            summary_n = int(summary["n_total"])
        except (TypeError, ValueError):
            pass
        else:
            if summary_n != theta.size:
                result.add_warning(
                    f"summary.json n_total={summary_n} disagrees with "
                    f"best_theta.npy length={theta.size}"
                )

    for name in OPTIONAL_FILES:
        _check_optional_json(root, name, result)

    return result


def format_report(result: ValidationResult, bundle_dir: Path | str) -> str:
    """Render a human-readable validation report."""
    lines = [f"Submission bundle: {bundle_dir}", ""]
    for message in result.info:
        lines.append(f"  info: {message}")
    for message in result.warnings:
        lines.append(f"  warn: {message}")
    for message in result.errors:
        lines.append(f"  error: {message}")
    lines.append("")
    if result.ok:
        lines.append("OK — bundle looks submit-ready.")
    else:
        lines.append(f"FAILED — {len(result.errors)} error(s). Fix before opening a PR.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_dir = repo_root / "submissions" / "final_model"

    ap = argparse.ArgumentParser(
        description="Validate a miner submission bundle (offline, no API calls)."
    )
    ap.add_argument(
        "--dir",
        type=Path,
        default=default_dir,
        help=f"bundle directory (default: {default_dir})",
    )
    ap.add_argument(
        "--n-total",
        type=int,
        default=None,
        help="override expected θ length (default: canonical ParamSpec.n_total)",
    )
    args = ap.parse_args(argv)

    result = validate_bundle(args.dir, expected_n_total=args.n_total)
    print(format_report(result, args.dir))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
