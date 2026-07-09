from __future__ import annotations

import os
import select
import subprocess
from pathlib import Path

import pty

from eval_backend.core.config import Settings
from eval_backend.services import eval_runner


def test_remote_attempt_uses_trinity_remote_host(tmp_path, monkeypatch):
    settings = Settings(
        workspace_root=tmp_path / "workspaces",
        artifact_root=tmp_path / "artifacts",
        local_repo_dir=tmp_path,
        trinity_remote_host="my-gpu-host",
    )
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    local_results_path = tmp_path / "results.json"
    ssh_hosts: list[str] = []

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "ssh":
            ssh_hosts.append(cmd[1])
        return subprocess.CompletedProcess(cmd, 0)

    class FakeProc:
        def poll(self):
            return 0

        def wait(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProc())
    monkeypatch.setattr(pty, "openpty", lambda: (3, 4))
    monkeypatch.setattr(select, "select", lambda *args, **kwargs: ([], [], []))
    monkeypatch.setattr(os, "read", lambda *args, **kwargs: b"")
    monkeypatch.setattr(os, "close", lambda *args, **kwargs: None)

    command, rc, stdout, stderr = eval_runner._remote_attempt(
        settings,
        checkpoint_path,
        local_results_path,
        "sub-remote-host",
        {},
    )

    assert ssh_hosts
    assert ssh_hosts[0] == "my-gpu-host"
    assert rc == 0
    assert isinstance(command, str)
    assert stdout == ""
    assert stderr == ""
