"""Offline security tests for code-benchmark subprocess grading (reward.py)."""
from __future__ import annotations

from trinity.orchestration import reward as R


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
