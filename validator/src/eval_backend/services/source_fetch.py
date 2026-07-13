from __future__ import annotations

import asyncio
import shutil
import tarfile
from pathlib import Path

import httpx

from ..core.config import Settings
from ..models import Submission
from .github import GITHUB_API_BASE, _repo_name, _repo_owner


def _safe_member_path(base: Path, member_name: str) -> Path:
    candidate = (base / member_name).resolve()
    base_resolved = base.resolve()
    if base_resolved not in candidate.parents and candidate != base_resolved:
        raise ValueError(f"Archive member escapes extraction root: {member_name}")
    return candidate


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise ValueError("Symlinks are not allowed in submission archives")
            _safe_member_path(destination, member.name)
        tar.extractall(destination)


def _pick_repo_root(extracted_root: Path) -> Path:
    children = [path for path in extracted_root.iterdir() if path.is_dir()]
    if len(children) == 1:
        return children[0]
    return extracted_root


def find_submission_checkpoint(root: Path) -> Path | None:
    preferred = root / "submissions" / "final_model" / "best_theta.npy"
    if preferred.exists():
        return preferred
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


async def fetch_github_pr_source(
    settings: Settings,
    submission: Submission,
    *,
    destination_root: Path | None = None,
) -> Path:
    owner = _repo_owner(submission.repo_full_name)
    repo = _repo_name(submission.repo_full_name)
    if owner is None or repo is None or not submission.head_sha:
        raise ValueError("submission does not have GitHub source metadata")

    root = destination_root or (settings.workspace_root.expanduser().resolve() / "github_sources" / submission.id)
    root = root.expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    archive_path = root / "source.tar.gz"
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
    }
    if settings.github_access_token:
        headers["Authorization"] = f"Bearer {settings.github_access_token}"

    async with httpx.AsyncClient(
        base_url=GITHUB_API_BASE,
        headers=headers,
        follow_redirects=True,
        timeout=120.0,
    ) as client:
        async with client.stream(
            "GET",
            f"/repos/{owner}/{repo}/tarball/{submission.head_sha}",
        ) as response:
            response.raise_for_status()
            with archive_path.open("wb") as handle:
                async for chunk in response.aiter_bytes():
                    handle.write(chunk)

    extracted_root = root / "source"
    extracted_root.mkdir(parents=True, exist_ok=True)
    _safe_extract_tar(archive_path, extracted_root)
    archive_path.unlink(missing_ok=True)
    return _pick_repo_root(extracted_root)


def fetch_github_pr_source_sync(
    settings: Settings,
    submission: Submission,
    *,
    destination_root: Path | None = None,
) -> Path:
    return asyncio.run(
        fetch_github_pr_source(
            settings,
            submission,
            destination_root=destination_root,
        )
    )
