"""Tests for the offline submission-bundle validator (scripts/validate_submission.py).

Covers the acceptance criteria from the feature issue: missing required file, wrong
theta shape/length, non-finite theta, malformed summary.json, n_total mismatch (warn),
and a valid fixture (pass).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_submission.py"
_spec = importlib.util.spec_from_file_location("validate_submission", _SCRIPT)
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)

N = 8  # small stand-in theta length; passed explicitly so tests don't depend on 13312


def _write_bundle(directory: Path, *, theta=None, summary=None, extra=None):
    directory.mkdir(parents=True, exist_ok=True)
    if theta is not None:
        np.save(directory / "best_theta.npy", theta)
    if summary is not None:
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for name, content in (extra or {}).items():
        (directory / name).write_text(content, encoding="utf-8")
    return directory


def test_valid_bundle_passes(tmp_path):
    _write_bundle(
        tmp_path,
        theta=np.zeros(N, dtype=np.float64),
        summary={"benchmark": "math500", "pool": ["a", "b", "c"], "n_total": N, "best_fitness": 0.8},
    )
    errors, warnings, info = vs.validate_bundle(tmp_path, expected_n_total=N)
    assert errors == []
    assert warnings == []
    assert info["theta_len"] == N


def test_missing_required_file(tmp_path):
    # only summary.json, no best_theta.npy
    _write_bundle(tmp_path, summary={"n_total": N})
    errors, _, _ = vs.validate_bundle(tmp_path, expected_n_total=N)
    assert any("best_theta.npy" in e for e in errors)


def test_wrong_theta_length(tmp_path):
    _write_bundle(tmp_path, theta=np.zeros(N + 5, dtype=np.float64), summary={"n_total": N})
    errors, _, _ = vs.validate_bundle(tmp_path, expected_n_total=N)
    assert any("length" in e and "expected n_total" in e for e in errors)


def test_non_finite_theta(tmp_path):
    theta = np.zeros(N, dtype=np.float64)
    theta[0] = np.nan
    theta[1] = np.inf
    _write_bundle(tmp_path, theta=theta, summary={"n_total": N})
    errors, _, _ = vs.validate_bundle(tmp_path, expected_n_total=N)
    assert any("non-finite" in e for e in errors)


def test_non_float_theta_errors(tmp_path):
    _write_bundle(tmp_path, theta=np.zeros(N, dtype=np.int64), summary={"n_total": N})
    errors, _, _ = vs.validate_bundle(tmp_path, expected_n_total=N)
    assert any("float array" in e for e in errors)


def test_malformed_summary_json(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.save(tmp_path / "best_theta.npy", np.zeros(N, dtype=np.float64))
    (tmp_path / "summary.json").write_text("{not valid json", encoding="utf-8")
    errors, _, _ = vs.validate_bundle(tmp_path, expected_n_total=N)
    assert any("summary.json is not valid JSON" in e for e in errors)


def test_n_total_mismatch_is_warning_not_error(tmp_path):
    _write_bundle(
        tmp_path,
        theta=np.zeros(N, dtype=np.float64),
        summary={"benchmark": "math500", "pool": [], "n_total": 999, "best_fitness": 0.1},
    )
    errors, warnings, _ = vs.validate_bundle(tmp_path, expected_n_total=N)
    assert errors == []  # theta itself is the right length
    assert any("disagrees" in w for w in warnings)


def test_optional_files_reported(tmp_path):
    _write_bundle(
        tmp_path,
        theta=np.zeros(N, dtype=np.float64),
        summary={"benchmark": "math500", "pool": [], "n_total": N, "best_fitness": 0.1},
        extra={"history.json": "{}", "eval.json": "{}"},
    )
    _, _, info = vs.validate_bundle(tmp_path, expected_n_total=N)
    assert set(info["optional_present"]) == {"history.json", "eval.json"}


def test_missing_directory(tmp_path):
    errors, _, _ = vs.validate_bundle(tmp_path / "nope", expected_n_total=N)
    assert any("not found" in e for e in errors)


def test_default_n_total_is_13312(tmp_path):
    # Locks the default against the coordinator ParamSpec (6*1024 + 7168).
    _write_bundle(
        tmp_path,
        theta=np.zeros(13312, dtype=np.float64),
        summary={"benchmark": "math500", "pool": [], "n_total": 13312, "best_fitness": 0.5},
    )
    errors, warnings, info = vs.validate_bundle(tmp_path)  # no override -> uses ParamSpec
    assert errors == []
    assert info["expected_n_total"] == 13312


def test_main_returns_nonzero_on_bad_bundle(tmp_path, capsys):
    _write_bundle(tmp_path, summary={"n_total": N})  # missing theta
    rc = vs.main(["--dir", str(tmp_path), "--n-total", str(N)])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_returns_zero_on_valid_bundle(tmp_path, capsys):
    _write_bundle(
        tmp_path,
        theta=np.zeros(N, dtype=np.float64),
        summary={"benchmark": "math500", "pool": [], "n_total": N, "best_fitness": 0.5},
    )
    rc = vs.main(["--dir", str(tmp_path), "--n-total", str(N)])
    assert rc == 0
    assert "OK" in capsys.readouterr().out
