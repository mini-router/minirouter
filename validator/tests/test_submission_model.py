from __future__ import annotations

from eval_backend.models import Artifact, Submission


def test_submission_checkpoint_path_ignores_source_archive() -> None:
    submission = Submission(id="sub-1", source="github_pr", benchmark_names_json=["math500"], status="queued")
    artifact = Artifact(
        id="artifact-1",
        storage_backend="local",
        storage_uri="/tmp/source.tar.gz",
        file_names_json=["src/main.py"],
        sha256="abc123",
        size_bytes=1,
        mime_type="application/gzip",
        submission_id=submission.id,
        meta_json={"extracted_root": "/tmp/extracted"},
    )
    submission.submission_artifact = artifact

    assert submission.checkpoint_path is None


def test_submission_checkpoint_path_accepts_direct_checkpoint_file() -> None:
    submission = Submission(id="sub-2", source="upload", benchmark_names_json=["math500"], status="queued")
    artifact = Artifact(
        id="artifact-2",
        storage_backend="local",
        storage_uri="/tmp/best_theta.npy",
        file_names_json=["best_theta.npy"],
        sha256="def456",
        size_bytes=1,
        mime_type="application/octet-stream",
        submission_id=submission.id,
        meta_json=None,
    )
    submission.submission_artifact = artifact

    assert submission.checkpoint_path == "/tmp/best_theta.npy"
