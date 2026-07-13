from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _make_fake_tools(tmp_path: Path) -> tuple[Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh_log = tmp_path / "ssh.log"
    rsync_log = tmp_path / "rsync.log"

    _write_executable(
        bin_dir / "ssh",
        f"""#!/usr/bin/env python3
from pathlib import Path
import sys

with Path({str(ssh_log)!r}).open("a", encoding="utf-8") as handle:
    handle.write(" ".join(sys.argv[1:]) + "\\n")
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

args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
if len(args) < 2:
    raise SystemExit(2)
src, dest = args[-2], args[-1]
with Path({str(rsync_log)!r}).open("a", encoding="utf-8") as handle:
    handle.write(src + " -> " + dest + "\\n")
dest_path = Path(os.environ["FAKE_REMOTE_ROOT"]) / dest.split(":", 1)[1]
dest_path.parent.mkdir(parents=True, exist_ok=True)
if src.endswith("secrets.env"):
    shutil.copy2(Path(src), dest_path)
else:
    dest_path.mkdir(parents=True, exist_ok=True)
raise SystemExit(0)
""",
    )

    return bin_dir, ssh_log, rsync_log


def test_setup_remote_copies_secrets_file(tmp_path):
    bin_dir, ssh_log, rsync_log = _make_fake_tools(tmp_path)
    remote_root = tmp_path / "remote_root"
    remote_root.mkdir()
    local_secrets = tmp_path / "secrets.env"
    local_secrets.write_text("API_KEY=local-secret\nBACKEND_URL=https://example.test\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
            "FAKE_REMOTE_ROOT": str(remote_root),
            "TRINITY_GPU_HOST": "fake-host",
            "TRINITY_REMOTE_DIR": "trinity",
            "TRINITY_GPU_INDEX": "2",
            "TRINITY_SECRETS_FILE": str(local_secrets),
        }
    )

    result = subprocess.run(
        ["bash", str(REPO / "scripts" / "setup_remote.sh")],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (remote_root / "trinity" / "secrets.env").read_text(encoding="utf-8") == local_secrets.read_text(
        encoding="utf-8"
    )
    assert "secrets.env -> fake-host:trinity/secrets.env" in rsync_log.read_text(encoding="utf-8")
    assert "secrets.env" in result.stdout
    assert "GPU 2" in result.stdout
    assert "bash -s" in ssh_log.read_text(encoding="utf-8")


def test_setup_remote_missing_secrets_file_fails(tmp_path):
    bin_dir, ssh_log, rsync_log = _make_fake_tools(tmp_path)
    remote_root = tmp_path / "remote_root"
    remote_root.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
            "FAKE_REMOTE_ROOT": str(remote_root),
            "TRINITY_GPU_HOST": "fake-host",
            "TRINITY_REMOTE_DIR": "trinity",
            "TRINITY_SECRETS_FILE": str(tmp_path / "missing.env"),
        }
    )

    result = subprocess.run(
        ["bash", str(REPO / "scripts" / "setup_remote.sh")],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "missing local secrets file" in result.stderr
    assert not rsync_log.exists()
    assert not ssh_log.exists()
