"""Load simple KEY=VALUE env files without requiring shell `export`."""
from __future__ import annotations

import os
import re
from pathlib import Path

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_trailing_env_comment(text: str) -> str:
    """Drop a trailing inline comment (` # ...`) from an unquoted env value."""
    pos = text.find(" #")
    return text[:pos].rstrip() if pos != -1 else text


def _parse_env_value(text: str) -> str:
    """Parse the RHS of a KEY=VALUE line, honoring quotes and inline comments."""
    text = text.strip()
    if not text:
        return text
    if text[0] not in {"'", '"'}:
        return _strip_trailing_env_comment(text)

    quote = text[0]
    i = 1
    while i < len(text):
        if text[i] == quote:
            inner = text[1:i]
            tail = text[i + 1 :]
            if not tail or tail.isspace():
                return inner
            rest = tail.lstrip()
            if rest.startswith("#"):
                return inner
            raise ValueError(
                "quoted env value has trailing non-comment text after closing quote: "
                f"{text!r}"
            )
        i += 1
    raise ValueError(f"quoted env value is missing a closing quote: {text!r}")


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
    value = _parse_env_value(value)
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
    for lineno, line in enumerate(p.read_text().splitlines(), start=1):
        try:
            parsed = _parse_env_line(line)
        except ValueError as exc:
            raise ValueError(f"{p}:{lineno}: {exc}") from exc
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
