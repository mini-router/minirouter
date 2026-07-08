"""Offline checks for remote GPU helper script defaults (issue #33)."""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_REMOTE = _REPO_ROOT / "scripts" / "run_remote.sh"
_SETUP_REMOTE = _REPO_ROOT / "scripts" / "setup_remote.sh"

_DEFAULT_GPU_RE = re.compile(r'TRINITY_GPU_INDEX:-(\d+)')
_LEGACY_GPU0_DEFAULT = re.compile(r'TRINITY_GPU_INDEX:-0')


def test_run_remote_defaults_to_allocated_gpu_five() -> None:
    text = _RUN_REMOTE.read_text()
    match = _DEFAULT_GPU_RE.search(text.split("CMD=")[0])
    assert match is not None, "run_remote.sh must declare a TRINITY_GPU_INDEX default"
    assert match.group(1) == "5"
    assert "[run_remote] using GPU index" in text


def test_setup_remote_defaults_to_allocated_gpu_five() -> None:
    text = _SETUP_REMOTE.read_text()
    matches = _DEFAULT_GPU_RE.findall(text)
    assert matches, "setup_remote.sh must declare TRINITY_GPU_INDEX defaults"
    assert all(value == "5" for value in matches), matches
    assert _LEGACY_GPU0_DEFAULT.search(text) is None
