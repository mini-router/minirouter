"""Offline unit tests for secrets env-file loading (``trinity.envfile``).

No real secrets files are read; tests use temporary paths only.
"""
import os

import pytest

import trinity.envfile as trinity_envfile
from trinity.envfile import load_env_file, load_project_env


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


def test_load_env_file_strips_unquoted_inline_comments(tmp_path, isolated_key):
    path = tmp_path / "secrets.env"
    path.write_text(f"{isolated_key}=sk-abc123  # production key\n", encoding="utf-8")
    load_env_file(path)
    assert os.environ[isolated_key] == "sk-abc123"


def test_load_env_file_strips_quoted_value_inline_comments(tmp_path, isolated_key):
    path = tmp_path / "secrets.env"
    path.write_text(f'{isolated_key}="sk-abc123"  # production key\n', encoding="utf-8")
    load_env_file(path)
    assert os.environ[isolated_key] == "sk-abc123"


def test_load_env_file_preserves_hash_inside_quotes(tmp_path, isolated_key):
    path = tmp_path / "secrets.env"
    path.write_text(f'{isolated_key}="value # kept"\n', encoding="utf-8")
    load_env_file(path)
    assert os.environ[isolated_key] == "value # kept"


def test_load_env_file_rejects_quoted_value_with_trailing_garbage(tmp_path, isolated_key):
    path = tmp_path / "secrets.env"
    path.write_text(f'{isolated_key}="abc"oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match="trailing non-comment"):
        load_env_file(path)


def test_load_env_file_rejects_quoted_value_with_trailing_non_comment_text(tmp_path, isolated_key):
    path = tmp_path / "secrets.env"
    path.write_text(f'{isolated_key}="abc" garbage\n', encoding="utf-8")
    with pytest.raises(ValueError, match="trailing non-comment"):
        load_env_file(path)


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


def test_load_project_env_prefers_repo_secrets(tmp_path, isolated_key, monkeypatch):
    monkeypatch.delenv("TRINITY_SECRETS_FILE", raising=False)
    (tmp_path / "secrets.env").write_text(f"{isolated_key}=from_repo\n", encoding="utf-8")
    (tmp_path / ".env").write_text(f"{isolated_key}=from_dot\n", encoding="utf-8")
    loaded = load_project_env(repo_root=tmp_path)
    assert loaded == tmp_path / "secrets.env"
    assert os.environ[isolated_key] == "from_repo"


def test_load_project_env_default_repo_root_matches_module_layout(
    tmp_path, isolated_key, monkeypatch
):
    # Simulate the real src/trinity/envfile.py depth, where parents[2] is the
    # repo root (regression test: a previous version used parents[3], which
    # pointed one directory above the repo).
    fake_repo_root = tmp_path / "repo"
    fake_module_dir = fake_repo_root / "src" / "trinity"
    fake_module_dir.mkdir(parents=True)
    monkeypatch.setattr(trinity_envfile, "__file__", str(fake_module_dir / "envfile.py"))
    monkeypatch.delenv("TRINITY_SECRETS_FILE", raising=False)
    (fake_repo_root / "secrets.env").write_text(
        f"{isolated_key}=from_default_root\n", encoding="utf-8"
    )

    loaded = load_project_env()

    assert loaded == fake_repo_root / "secrets.env"
    assert os.environ[isolated_key] == "from_default_root"
