"""Load simple KEY=VALUE env files without requiring shell `export`."""
from __future__ import annotations

import os
import re
from pathlib import Path

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_env_line(line: str) -> tuple[str, str] | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if raw.startswith("export "):
        raw = raw[len("export ") :].lstrip()
    if "=" not in raw:
        return None
    key, value = raw.split("=", 1)
    key = key.strip()
    if not _KEY_RE.match(key):
        return None
    value = value.strip()
    if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        # Quoted value: keep everything between the quotes verbatim, including `#`.
        value = value[1:-1]
    else:
        # Unquoted value: strip a trailing inline comment introduced by ` #`
        # (whitespace + hash), matching common `.env` conventions. A bare `#`
        # with no leading whitespace is treated as literal value text.
        match = re.search(r"\s#", value)
        if match:
            value = value[: match.start()].rstrip()
    value = os.path.expanduser(os.path.expandvars(value))
    return key, value


def load_env_file(path: str | Path) -> Path | None:
    """Load env vars from a file if it exists.

    Existing process env wins. The file may contain plain `KEY=VALUE` lines or
    `export KEY=VALUE`.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key not in os.environ:
            os.environ[key] = value
    return p


def load_project_env(*, repo_root: str | Path | None = None) -> Path | None:
    """Load the first matching secrets file for this project."""
    root = Path(repo_root).expanduser() if repo_root is not None else Path(__file__).resolve().parents[3]
    candidates = [
        os.environ.get("TRINITY_SECRETS_FILE"),
        root / "secrets.env",
        root / ".env",
        Path.home() / ".config" / "trinity" / "secrets.env",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        loaded = load_env_file(candidate)
        if loaded is not None:
            return loaded
    return None
