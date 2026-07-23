"""Offline security tests for code-benchmark subprocess grading (reward.py)."""
from __future__ import annotations

import pytest

from trinity.orchestration import reward as R

try:  # resource is POSIX-only; the rlimit tests are gated on it.
    import resource as _resource  # noqa: F401

    _POSIX_RLIMIT = True
except ImportError:  # pragma: no cover - non-POSIX CI.
    _POSIX_RLIMIT = False


def test_sandbox_replaces_home_so_secrets_are_unreachable(tmp_path, monkeypatch):
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    secrets_dir = real_home / ".config" / "trinity"
    secrets_dir.mkdir(parents=True)
    (secrets_dir / "secrets.env").write_text("API_KEY=leaked")
    monkeypatch.setenv("HOME", str(real_home))

    leak_probe = """
import pathlib
p = pathlib.Path("~/.config/trinity/secrets.env").expanduser()
print("LEAK", p.read_text() if p.exists() else "NO-FILE")
"""
    tests = [{"input": "", "output": "LEAK NO-FILE"}]
    assert R.run_pass_at_1(leak_probe, tests, timeout_s=5) is True


def test_sandbox_still_grades_valid_code():
    code = "import sys\nprint(int(sys.stdin.read()) * 2)"
    tests = [{"input": "21\n", "output": "42"}]
    assert R.run_pass_at_1(code, tests, timeout_s=5) is True


def test_sandbox_env_uses_private_home_only():
    env = R._sandbox_env(home_dir="/tmp/private-home")
    assert env["HOME"] == "/tmp/private-home"
    assert env["TMPDIR"] == "/tmp/private-home"
    assert "API_KEY" not in env
    assert "GITHUB_ACCESS_TOKEN" not in env


# --- Resource-limit defense in depth (issue #71 item #2) -------------------


@pytest.mark.skipif(not _POSIX_RLIMIT, reason="resource limits are POSIX-only")
def test_rlimit_preexec_returns_callable_on_posix():
    hook = R._rlimit_preexec()
    assert callable(hook)


@pytest.mark.skipif(not _POSIX_RLIMIT, reason="resource limits are POSIX-only")
def test_memory_bomb_candidate_is_killed_not_passed():
    # A candidate that tries to allocate ~4 GiB must be killed by RLIMIT_AS
    # (2 GiB cap), so it never passes its test. Without the limit this would
    # balloon host memory before the wall-clock timeout.
    bomb = "x = bytearray(4 * 1024 * 1024 * 1024)\nprint('OK')"
    tests = [{"input": "", "output": "OK"}]
    assert R.run_pass_at_1(bomb, tests, timeout_s=10) is False


@pytest.mark.skipif(not _POSIX_RLIMIT, reason="resource limits are POSIX-only")
def test_rlimits_do_not_break_ordinary_grading():
    # A normal, small solution must still pass with the limits in place.
    code = "import sys\nprint(int(sys.stdin.read()) + 1)"
    tests = [{"input": "41\n", "output": "42"}]
    assert R.run_pass_at_1(code, tests, timeout_s=5) is True
