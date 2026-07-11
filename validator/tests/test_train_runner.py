from __future__ import annotations

import json
from pathlib import Path

from eval_backend.core.config import Settings
from eval_backend.models import Artifact, Submission, TrainRun
from eval_backend.services import train_runner


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path / "workspaces",
        artifact_root=tmp_path / "artifacts",
        local_repo_dir=tmp_path,
        eval_timeout_seconds=30,
        train_max_items=8,
        train_generations=3,
        train_popsize=4,
        train_m_cma=2,
    )


def _submission(session) -> Submission:
    sub = Submission(
        id="sub-train-1",
        source="upload",
        miner_id="miner-a",
        benchmark_names_json=["math500"],
        status="queued",
    )
    session.add(sub)
    session.flush()
    return sub


def test_run_train_job_creates_output_artifact(validator_session, tmp_path, monkeypatch):
    session = validator_session
    settings = _build_settings(tmp_path)
    submission = _submission(session)
    train = TrainRun(
        submission_id=submission.id,
        source="manual",
        benchmark_names_json=["math500"],
        status="queued",
        phase="queued",
    )
    session.add(train)
    session.flush()

    def _fake_run_bash_stream(command, cwd, timeout, env=None, on_line=None):
        assert env is not None
        ledger_path = Path(env["TRINITY_COST_LEDGER"])
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(
            json.dumps({"provider": "chutes", "m": "google/gemma-4-31B-turbo-TEE", "p": 1000, "c": 200})
            + "\n",
            encoding="utf-8",
        )
        run_dir = Path(settings.local_repo_dir) / "experiments" / "math500" / f"train-{train.id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best_theta.npy").write_bytes(b"theta-bytes")
        (run_dir / "summary.json").write_text(
            json.dumps({"benchmark": "math500", "n_total": 8, "best_fitness": 0.5}),
            encoding="utf-8",
        )
        (run_dir / "history.json").write_text("[]", encoding="utf-8")
        return 0, "train stdout", ""

    monkeypatch.setattr(train_runner, "_run_bash_stream", _fake_run_bash_stream)

    result = train_runner.run_train_job(session, train, settings)

    assert result.train.status == "completed"
    assert result.train.output_artifact_id is not None
    assert result.output_artifact is not None
    assert result.output_artifact.meta_json["checkpoint_path"].endswith("best_theta.npy")
    assert submission.latest_train_id == train.id
    assert result.train.metrics_json is not None
    assert result.train.cost_usd and result.train.cost_usd > 0
    assert submission.cost_usd and submission.cost_usd > 0


def test_run_train_job_uses_submission_source_bundle_when_pipeline_trains(validator_session, tmp_path, monkeypatch):
    session = validator_session
    settings = _build_settings(tmp_path)
    settings.pipeline_mode = "train_eval"
    submission = _submission(session)

    source_root = tmp_path / "artifacts" / "extracted" / "sub-train-source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "src").mkdir(parents=True, exist_ok=True)
    artifact = Artifact(
        id="artifact-source",
        storage_backend="local",
        storage_uri=str(source_root),
        file_names_json=["src/main.py"],
        sha256="abc123",
        size_bytes=1,
        mime_type="application/gzip",
        submission_id=submission.id,
        meta_json={"extracted_root": str(source_root)},
    )
    session.add(artifact)
    session.flush()
    submission.submission_artifact_id = artifact.id

    train = TrainRun(
        submission_id=submission.id,
        source="github_pr",
        benchmark_names_json=["math500"],
        status="queued",
        phase="queued",
    )
    session.add(train)
    session.flush()

    def _fake_run_bash_stream(command, cwd, timeout, env=None, on_line=None):
        assert str(source_root) in command
        assert str(tmp_path / ".venv" / "bin" / "activate") in command
        assert cwd == source_root
        ledger_path = Path(env["TRINITY_COST_LEDGER"])
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(
            json.dumps({"provider": "chutes", "m": "google/gemma-4-31B-turbo-TEE", "p": 1000, "c": 200})
            + "\n",
            encoding="utf-8",
        )
        run_dir = source_root / "experiments" / "math500" / f"train-{train.id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best_theta.npy").write_bytes(b"theta-bytes")
        (run_dir / "summary.json").write_text(
            json.dumps({"benchmark": "math500", "n_total": 8, "best_fitness": 0.5}),
            encoding="utf-8",
        )
        return 0, "train stdout", ""

    monkeypatch.setattr(train_runner, "_run_bash_stream", _fake_run_bash_stream)

    result = train_runner.run_train_job(session, train, settings)

    assert result.train.status == "completed"
    assert submission.checkpoint_path is None
    assert result.output_artifact is not None
    assert result.output_artifact.meta_json["checkpoint_path"].endswith("best_theta.npy")
