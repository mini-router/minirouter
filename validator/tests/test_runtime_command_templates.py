from __future__ import annotations

from eval_backend.core.config import Settings


def test_eval_command_templates_are_forced_to_submission_only(monkeypatch):
    monkeypatch.setenv(
        "REMOTE_EVAL_COMMAND_TEMPLATE",
        "PYTHONPATH=src python -m trinity.eval --benchmark {benchmark} --provider {provider} --models {models_config} --theta {checkpoint_path} --out {results_path}",
    )
    monkeypatch.setenv(
        "LOCAL_EVAL_COMMAND_TEMPLATE",
        "cd {repo_dir} && source .venv/bin/activate && PYTHONPATH=src python -u -m trinity.eval --benchmark {benchmark} --provider {provider} --models {models_config} --theta {checkpoint_path} --out {results_path}",
    )

    settings = Settings.load()

    assert "--submission-only" in settings.remote_eval_command_template
    assert "--submission-only" in settings.local_eval_command_template
