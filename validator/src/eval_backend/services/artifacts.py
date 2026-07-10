from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4
import hashlib

from sqlalchemy.orm import Session

from ..models import Artifact
from .storage import StoredArtifact


def _list_files(path: Path) -> list[str]:
    if path.is_file():
        return [path.name]
    if not path.exists():
        return []
    files: list[str] = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            files.append(str(item.relative_to(path)))
    return files


def _file_names(stored: StoredArtifact) -> list[str]:
    if stored.extracted_root is not None:
        return _list_files(stored.extracted_root)
    return [stored.path.name]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for item in sorted(path.rglob("*")):
            if item.is_file():
                digest.update(str(item.relative_to(path)).encode("utf-8"))
                with item.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persist_stored_artifact(
    session: Session,
    stored: StoredArtifact,
    *,
    storage_backend: str,
    storage_uri: str | None = None,
    submission_id: str | None = None,
    train_id: int | None = None,
    evaluation_id: int | None = None,
    meta_json: dict[str, Any] | None = None,
    sha256_override: str | None = None,
    size_bytes_override: int | None = None,
) -> Artifact:
    artifact = Artifact(
        id=str(uuid4()),
        storage_backend=storage_backend,
        storage_uri=storage_uri or str(stored.path),
        file_names_json=_file_names(stored),
        sha256=sha256_override or stored.sha256 or _sha256_path(stored.path),
        size_bytes=size_bytes_override
        if size_bytes_override is not None
        else (
            sum(item.stat().st_size for item in stored.path.rglob("*") if item.is_file())
            if stored.path.is_dir()
            else (stored.path.stat().st_size if stored.path.exists() else None)
        ),
        mime_type=None,
        submission_id=submission_id,
        train_id=train_id,
        evaluation_id=evaluation_id,
        meta_json=meta_json,
    )
    session.add(artifact)
    session.flush()
    return artifact
