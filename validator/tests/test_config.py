from __future__ import annotations

from pathlib import Path

import pytest

from eval_backend.core import config
from eval_backend.core.config import Settings, _parse_env_file, _secrets_file_candidates


@pytest.fixture
def repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "_repo_root", lambda: tmp_path)
    monkeypatch.delenv("TRINITY_SECRETS_FILE", raising=False)
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return tmp_path


def test_secrets_file_candidates_match_trinity_envfile_order(repo_root: Path) -> None:
    (repo_root / "secrets.env").write_text("GITHUB_WEBHOOK_SECRET=from-secrets\n", encoding="utf-8")
    (repo_root / ".env").write_text("GITHUB_WEBHOOK_SECRET=from-dot\n", encoding="utf-8")

    candidates = _secrets_file_candidates(repo_root)
    assert candidates[0] == repo_root / "secrets.env"
    assert candidates[1] == repo_root / ".env"
    assert candidates[2] == Path.home() / ".config" / "trinity" / "secrets.env"


def test_settings_load_prefers_repo_secrets_over_dot_env(repo_root: Path) -> None:
    (repo_root / "secrets.env").write_text("GITHUB_WEBHOOK_SECRET=from-secrets\n", encoding="utf-8")
    (repo_root / ".env").write_text("GITHUB_WEBHOOK_SECRET=from-dot\n", encoding="utf-8")

    settings = Settings.load()
    assert settings.github_webhook_secret == "from-secrets"
    assert settings.trinity_secrets_file == str(repo_root / "secrets.env")


def test_settings_load_uses_dot_env_when_repo_secrets_missing(repo_root: Path) -> None:
    (repo_root / ".env").write_text("GITHUB_WEBHOOK_SECRET=from-dot\n", encoding="utf-8")

    settings = Settings.load()
    assert settings.github_webhook_secret == "from-dot"
    assert settings.trinity_secrets_file == str(repo_root / ".env")


def test_settings_load_honors_trinity_secrets_file_override(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = repo_root / "custom.env"
    custom.write_text('GITHUB_WEBHOOK_SECRET="custom-secret"\n', encoding="utf-8")
    (repo_root / "secrets.env").write_text("GITHUB_WEBHOOK_SECRET=ignored\n", encoding="utf-8")
    monkeypatch.setenv("TRINITY_SECRETS_FILE", str(custom))

    settings = Settings.load()
    assert settings.github_webhook_secret == "custom-secret"
    assert settings.trinity_secrets_file == str(custom)


def test_settings_load_process_env_wins_over_secrets_file(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo_root / "secrets.env").write_text("GITHUB_WEBHOOK_SECRET=from-file\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "from-process")

    settings = Settings.load()
    assert settings.github_webhook_secret == "from-process"


def test_parse_env_file_expands_tilde_and_quotes(tmp_path: Path) -> None:
    path = tmp_path / "secrets.env"
    path.write_text('ARTIFACT_ROOT="~/artifacts"\n', encoding="utf-8")

    values = _parse_env_file(path)
    assert values["ARTIFACT_ROOT"] == str(Path.home() / "artifacts")
