"""Offline unit tests for secrets env-file loading (``trinity.envfile``).

No real secrets files are read; tests use temporary paths only.
"""
import os

import pytest

from trinity.envfile import _parse_env_line, load_env_file, load_project_env


@pytest.fixture
def isolated_key(monkeypatch):
    """Ensure TEST_TRINITY_ENV_KEY is unset before/after each test."""
    key = "TEST_TRINITY_ENV_KEY"
    monkeypatch.delenv(key, raising=False)
    yield key
    monkeypatch.delenv(key, raising=False)


def test_load_env_file_missing_returns_none(tmp_path):
    assert load_env_file(tmp_path / "nope.env") is None


def test_load_env_file_sets_unset_vars(tmp_path, isolated_key):
    path = tmp_path / "secrets.env"
    path.write_text(f"{isolated_key}=hello\n", encoding="utf-8")
    assert load_env_file(path) == path
    assert os.environ[isolated_key] == "hello"


def test_load_env_file_does_not_override_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("TRINITY_KEEP_ME", "original")
    path = tmp_path / "secrets.env"
    path.write_text("TRINITY_KEEP_ME=replaced\n", encoding="utf-8")
    load_env_file(path)
    assert os.environ["TRINITY_KEEP_ME"] == "original"


def test_load_env_file_supports_export_prefix(tmp_path, isolated_key):
    path = tmp_path / "secrets.env"
    path.write_text(f"export {isolated_key}=from_export\n", encoding="utf-8")
    load_env_file(path)
    assert os.environ[isolated_key] == "from_export"


def test_load_env_file_strips_double_quotes(tmp_path, isolated_key):
    path = tmp_path / "secrets.env"
    path.write_text(f'{isolated_key}="hello world"\n', encoding="utf-8")
    load_env_file(path)
    assert os.environ[isolated_key] == "hello world"


def test_load_env_file_skips_comments_and_blanks(tmp_path, isolated_key):
    path = tmp_path / "secrets.env"
    path.write_text(
        f"# comment\n\n{isolated_key}=ok\n# tail\n",
        encoding="utf-8",
    )
    load_env_file(path)
    assert os.environ[isolated_key] == "ok"


def test_load_env_file_skips_invalid_keys(tmp_path, isolated_key):
    path = tmp_path / "secrets.env"
    path.write_text(
        "bad-key=nope\n"
        f"{isolated_key}=yes\n",
        encoding="utf-8",
    )
    load_env_file(path)
    assert "bad-key" not in os.environ
    assert os.environ[isolated_key] == "yes"


def test_parse_strips_inline_comment_unquoted():
    assert _parse_env_line("OPENROUTER_API_KEY=sk-abc123  # production key") == (
        "OPENROUTER_API_KEY",
        "sk-abc123",
    )


def test_parse_keeps_hash_inside_quotes():
    assert _parse_env_line('KEY="value # kept"') == ("KEY", "value # kept")
    assert _parse_env_line("KEY='value # kept'") == ("KEY", "value # kept")


def test_parse_bare_hash_without_space_is_literal():
    # A `#` not preceded by whitespace is part of the value (e.g. a URL fragment).
    assert _parse_env_line("KEY=abc#frag") == ("KEY", "abc#frag")


def test_parse_full_line_comment_still_skipped():
    assert _parse_env_line("# whole line") is None


def test_load_env_file_strips_inline_comment(tmp_path, isolated_key):
    path = tmp_path / "secrets.env"
    path.write_text(f"{isolated_key}=hello  # trailing note\n", encoding="utf-8")
    load_env_file(path)
    assert os.environ[isolated_key] == "hello"


def test_load_project_env_prefers_repo_secrets(tmp_path, isolated_key, monkeypatch):
    monkeypatch.delenv("TRINITY_SECRETS_FILE", raising=False)
    (tmp_path / "secrets.env").write_text(f"{isolated_key}=from_repo\n", encoding="utf-8")
    (tmp_path / ".env").write_text(f"{isolated_key}=from_dot\n", encoding="utf-8")
    loaded = load_project_env(repo_root=tmp_path)
    assert loaded == tmp_path / "secrets.env"
    assert os.environ[isolated_key] == "from_repo"
