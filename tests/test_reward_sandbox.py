"""Isolation tests for untrusted code execution — issue #71.

`run_pass_at_1` grades LiveCodeBench/BigCodeBench by executing model-generated
(and, in the validator, untrusted) code. These tests pin the isolation the fix
enforces:

  * `_sandbox_env()` no longer forwards the real HOME (secrets live under `~`);
  * a candidate cannot read `~/.config/trinity/secrets.env` (the PoC);
  * a candidate has no external network egress (where rootless netns is available);
  * resource limits bound memory;
  * strict mode fails closed when network isolation is unavailable;
  * legitimate solutions still grade correctly.

Network/rlimit checks need POSIX + fork; they skip cleanly elsewhere.
"""
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.orchestration import reward as R  # noqa: E402

_POSIX = os.name == "posix" and hasattr(os, "fork")
_LEAK_HOME = (
    'import pathlib;'
    'p=pathlib.Path("~/.config/trinity/secrets.env").expanduser();'
    'print(p.read_text() if p.exists() else "NO-FILE")'
)


def test_sandbox_env_does_not_forward_real_home(monkeypatch):
    monkeypatch.setenv("HOME", "/home/victim")
    monkeypatch.setenv("FIREWORKS_API_KEY", "should-not-appear")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-appear")
    env = R._sandbox_env("/run/sbx")
    # HOME is redirected to the private run dir, not the invoking user's home.
    assert env["HOME"] == "/run/sbx"
    assert env["HOME"] != "/home/victim"
    assert env.get("TMPDIR") == "/run/sbx"
    # No credential-shaped variables are passed through.
    assert not any("KEY" in k or "SECRET" in k or "TOKEN" in k for k in env)


@pytest.mark.skipif(not _POSIX, reason="needs POSIX fork/preexec")
def test_home_secret_is_unreadable_by_child(monkeypatch, tmp_path):
    # Point the PARENT's HOME at a dir holding a fake secret; the child must NOT
    # be able to reach it, because its HOME is redirected to a private run dir.
    fake_home = tmp_path / "victim_home"
    secret = fake_home / ".config" / "trinity" / "secrets.env"
    secret.parent.mkdir(parents=True)
    secret.write_text("FIREWORKS_API_KEY=leak-me\n")
    monkeypatch.setenv("HOME", str(fake_home))

    ok, out = R._exec_script_capture(_LEAK_HOME, stdin_data="", timeout_s=15)
    assert ok is True          # the script itself runs fine...
    assert "NO-FILE" in out    # ...but finds no secret under ~
    assert "leak-me" not in out


@pytest.mark.skipif(
    not _POSIX or not R.network_isolation_available(),
    reason="rootless network namespace not available on this host",
)
def test_no_external_network_egress():
    script = (
        'import socket\n'
        's=socket.socket();s.settimeout(3)\n'
        's.connect(("1.1.1.1", 53))\n'
        'print("NET-OK")\n'
    )
    ok, out = R._exec_script_capture(script, stdin_data="", timeout_s=15)
    assert ok is False          # connect() must fail (network unreachable)
    assert "NET-OK" not in out


@pytest.mark.skipif(not _POSIX, reason="needs POSIX rlimits")
def test_address_space_limit_enforced(monkeypatch):
    # Tighten the memory cap, then try to allocate well past it.
    monkeypatch.setenv("TRINITY_SANDBOX_MEM_MB", "128")
    hog = 'x = bytearray(400 * 1024 * 1024)\nprint(len(x))\n'
    ok, out = R._exec_script_capture(hog, stdin_data="", timeout_s=15)
    assert ok is False          # MemoryError / killed by the AS rlimit
    assert "419430400" not in out


@pytest.mark.skipif(not _POSIX, reason="needs POSIX fork")
def test_strict_mode_fails_closed(monkeypatch):
    # Force "no network isolation available" and demand strict mode.
    monkeypatch.setattr(R, "_netns_supported", False)
    monkeypatch.setattr(R, "_strict_warned", False)
    monkeypatch.setenv("TRINITY_SANDBOX_STRICT", "1")
    ok, out = R._exec_script_capture('print("ran")', stdin_data="", timeout_s=10)
    assert ok is False          # refuses to execute rather than run unisolated
    assert out == ""


def test_timeout_is_enforced():
    ok, out = R._exec_script_capture(
        "while True:\n    pass\n", stdin_data="", timeout_s=2
    )
    assert ok is False


def test_legitimate_grading_still_works():
    good = "import sys\nn=int(sys.stdin.read())\nprint(n*n)"
    bad = "import sys\nn=int(sys.stdin.read())\nprint(n+1)"
    tests = [{"input": "5\n", "output": "25"}, {"input": "3\n", "output": "9"}]
    assert R.run_pass_at_1(good, tests, timeout_s=15) is True
    assert R.run_pass_at_1(bad, tests, timeout_s=15) is False
    # assert-based (BigCodeBench) flavor
    assert R.run_pass_at_1("def add(a,b):\n    return a+b", ["assert add(2,3)==5"],
                           timeout_s=15) is True
