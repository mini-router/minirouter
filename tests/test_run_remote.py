from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _make_fake_remote(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    remote_root = tmp_path / "remote_root"
    remote_repo = remote_root / "trinity"
    (remote_repo / "experiments" / "math500" / "run-1").mkdir(parents=True)
    (remote_repo / "experiments" / "math500" / "run-1" / "best_theta.npy").write_bytes(
        b"theta-bytes"
    )
    (remote_repo / "experiments" / "math500" / "run-1" / "summary.json").write_text(
        "{\"benchmark\": \"math500\"}",
        encoding="utf-8",
    )
    (remote_repo / "experiments" / "math500" / "run-1" / "history.json").write_text(
        "[]",
        encoding="utf-8",
    )
    (remote_repo / "cost_ledger.jsonl").write_text(
        "{\"provider\":\"fake\",\"m\":\"model\",\"p\":1,\"c\":2}\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh_log = tmp_path / "ssh.log"
    rsync_log = tmp_path / "rsync.log"

    _write_executable(
        bin_dir / "ssh",
        f"""#!/usr/bin/env python3
from pathlib import Path
import os
import sys

remote_root = Path(os.environ["FAKE_REMOTE_ROOT"])
cmd = " ".join(sys.argv[2:])
if cmd.startswith("test -d "):
    remote_path = cmd.removeprefix("test -d ").strip()
    raise SystemExit(0 if (remote_root / remote_path).is_dir() else 1)
if cmd.startswith("test -f "):
    remote_path = cmd.removeprefix("test -f ").strip()
    raise SystemExit(0 if (remote_root / remote_path).is_file() else 1)
with Path({str(ssh_log)!r}).open("a", encoding="utf-8") as handle:
    handle.write(cmd + "\\n")
raise SystemExit(0)
""",
    )

    _write_executable(
        bin_dir / "rsync",
        f"""#!/usr/bin/env python3
from pathlib import Path
import os
import shutil
import sys

remote_root = Path(os.environ["FAKE_REMOTE_ROOT"])
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
if len(args) < 2:
    raise SystemExit(2)
src, dest = args[-2], args[-1]
with Path({str(rsync_log)!r}).open("a", encoding="utf-8") as handle:
    handle.write(src + " -> " + dest + "\\n")
remote_path = src.split(":", 1)[1]
source = remote_root / remote_path
dest_path = Path(dest)
if src.endswith("/"):
    dest_path.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = dest_path / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
else:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest_path)
raise SystemExit(0)
""",
    )

    return remote_root, bin_dir, ssh_log, rsync_log


def test_run_remote_syncs_artifacts_back(tmp_path):
    remote_root, bin_dir, ssh_log, rsync_log = _make_fake_remote(tmp_path)
    local_sync = tmp_path / "local"
    local_sync.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
            "FAKE_REMOTE_ROOT": str(remote_root),
            "TRINITY_GPU_HOST": "fake-host",
            "TRINITY_REMOTE_DIR": "trinity",
            "TRINITY_SYNC_DIR": str(local_sync),
            "TRINITY_SYNC_ENABLED": "1",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO / "scripts" / "run_remote.sh"),
            "train",
            "--benchmark",
            "math500",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (local_sync / "experiments" / "math500" / "run-1" / "best_theta.npy").read_bytes() == b"theta-bytes"
    assert (
        local_sync / "experiments" / "math500" / "run-1" / "summary.json"
    ).read_text(encoding="utf-8") == "{\"benchmark\": \"math500\"}"
    assert (local_sync / "experiments" / "math500" / "run-1" / "history.json").exists()
    assert (local_sync / "cost_ledger.jsonl").read_text(encoding="utf-8") == (
        "{\"provider\":\"fake\",\"m\":\"model\",\"p\":1,\"c\":2}\n"
    )
    assert "experiments" in rsync_log.read_text(encoding="utf-8")
    assert "cost_ledger.jsonl" in rsync_log.read_text(encoding="utf-8")
    assert "python -m trinity.train" in ssh_log.read_text(encoding="utf-8")


def test_run_remote_can_disable_sync(tmp_path):
    remote_root, bin_dir, ssh_log, rsync_log = _make_fake_remote(tmp_path)
    local_sync = tmp_path / "local"
    local_sync.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
            "FAKE_REMOTE_ROOT": str(remote_root),
            "TRINITY_GPU_HOST": "fake-host",
            "TRINITY_REMOTE_DIR": "trinity",
            "TRINITY_SYNC_DIR": str(local_sync),
            "TRINITY_SYNC_ENABLED": "0",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO / "scripts" / "run_remote.sh"),
            "eval",
            "--benchmark",
            "math500",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not rsync_log.exists()
    assert not (local_sync / "experiments").exists()
    assert "sync disabled" in result.stdout
    assert "python -m trinity.eval" in ssh_log.read_text(encoding="utf-8")
