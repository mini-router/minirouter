from __future__ import annotations

import hashlib
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from ..core.config import Settings


@dataclass(slots=True)
class StoredArtifact:
    name: str
    path: Path
    sha256: str
    extracted_root: Path | None = None
    checkpoint_path: Path | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)


def _safe_member_path(base: Path, member_name: str) -> Path:
    candidate = (base / member_name).resolve()
    if base.resolve() not in candidate.parents and candidate != base.resolve():
        raise ValueError(f"Archive member escapes extraction root: {member_name}")
    return candidate


def _extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise ValueError("Symlinks are not allowed in submission archives")
            _safe_member_path(destination, member.name)
        tar.extractall(destination)


def _extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            _safe_member_path(destination, member)
        zf.extractall(destination)


def _find_checkpoint(root: Path) -> Path | None:
    candidates = [
        "best_theta.npy",
        "head_params.pt",
        "theta.npy",
        "theta.pt",
        "checkpoint.pt",
    ]
    for name in candidates:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def store_upload(upload: UploadFile, settings: Settings, submission_id: str) -> StoredArtifact:
    filename = upload.filename or "submission.bin"
    safe_name = os.path.basename(filename)
    upload_path = settings.artifact_root / "uploads" / submission_id / safe_name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    _write_upload(upload, upload_path)

    extracted_root: Path | None = None
    checkpoint_path: Path | None = None
    suffix = "".join(upload_path.suffixes).lower()
    if suffix.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
        extracted_root = settings.artifact_root / "extracted" / submission_id
        extracted_root.mkdir(parents=True, exist_ok=True)
        if suffix.endswith(".zip"):
            _extract_zip(upload_path, extracted_root)
        else:
            _extract_tar(upload_path, extracted_root)
        checkpoint_path = _find_checkpoint(extracted_root)
    elif upload_path.suffix.lower() in {".npy", ".pt", ".pth"}:
        checkpoint_path = upload_path

    return StoredArtifact(
        name=safe_name,
        path=upload_path,
        sha256=_sha256(upload_path),
        extracted_root=extracted_root,
        checkpoint_path=checkpoint_path,
    )
