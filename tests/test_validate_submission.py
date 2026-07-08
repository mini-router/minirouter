"""Unit tests for the offline submission-bundle validator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.coordinator import params as P  # noqa: E402
from trinity.validate_submission import (  # noqa: E402
    format_report,
    main,
    validate_bundle,
)


def _write_valid_bundle(tmp: Path, *, n_total: int | None = None) -> Path:
    spec = P.make_spec()
    n = spec.n_total if n_total is None else n_total
    theta = np.zeros(n, dtype=np.float64)
    np.save(tmp / "best_theta.npy", theta)
    (tmp / "summary.json").write_text(
        json.dumps(
            {
                "benchmark": "math500",
                "pool": ["a", "b", "c"],
                "n_total": n,
                "best_fitness": 0.8,
            }
        ),
        encoding="utf-8",
    )
    return tmp


def test_valid_bundle_passes(tmp_path: Path):
    bundle = _write_valid_bundle(tmp_path)
    result = validate_bundle(bundle)
    assert result.ok
    assert not result.errors
    assert any("matches n_total" in line for line in result.info)


def test_missing_required_files_fail(tmp_path: Path):
    result = validate_bundle(tmp_path)
    assert not result.ok
    assert any("best_theta.npy" in e for e in result.errors)
    assert any("summary.json" in e for e in result.errors)


def test_wrong_theta_length_fails(tmp_path: Path):
    _write_valid_bundle(tmp_path, n_total=128)
    # overwrite summary to claim the wrong length so we only trip the shape check
    (tmp_path / "summary.json").write_text(
        json.dumps({"benchmark": "math500", "n_total": 128, "best_fitness": 0.1}),
        encoding="utf-8",
    )
    result = validate_bundle(tmp_path)
    assert not result.ok
    assert any("length 128" in e for e in result.errors)


def test_summary_n_total_mismatch_warns(tmp_path: Path):
    bundle = _write_valid_bundle(tmp_path)
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    summary["n_total"] = 999
    (bundle / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    result = validate_bundle(bundle)
    assert result.ok  # length of theta is still correct — warning only
    assert any("disagrees" in w for w in result.warnings)


def test_invalid_summary_json_fails(tmp_path: Path):
    spec = P.make_spec()
    np.save(tmp_path / "best_theta.npy", np.zeros(spec.n_total, dtype=np.float64))
    (tmp_path / "summary.json").write_text("{not-json", encoding="utf-8")
    result = validate_bundle(tmp_path)
    assert not result.ok
    assert any("summary.json" in e for e in result.errors)


def test_nan_theta_fails(tmp_path: Path):
    bundle = _write_valid_bundle(tmp_path)
    bad = np.zeros(P.make_spec().n_total, dtype=np.float64)
    bad[0] = np.nan
    np.save(bundle / "best_theta.npy", bad)
    result = validate_bundle(bundle)
    assert not result.ok
    assert any("NaN" in e for e in result.errors)


def test_optional_files_reported(tmp_path: Path):
    bundle = _write_valid_bundle(tmp_path)
    (bundle / "history.json").write_text("[]", encoding="utf-8")
    (bundle / "eval.json").write_text("{}", encoding="utf-8")
    result = validate_bundle(bundle)
    assert result.ok
    assert any("history.json" in line and "present" in line for line in result.info)
    assert any("eval.json" in line and "present" in line for line in result.info)


def test_format_report_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    bundle = _write_valid_bundle(tmp_path)
    result = validate_bundle(bundle)
    text = format_report(result, bundle)
    assert "OK" in text
    assert main(["--dir", str(bundle)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--dir", str(empty)]) == 1
