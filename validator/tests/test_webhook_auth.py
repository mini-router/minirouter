from __future__ import annotations

import hashlib
import hmac

import pytest
from fastapi import HTTPException

from eval_backend.api.routes import _verify_github_signature, _verify_shared_secret


def _signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


@pytest.mark.parametrize("secret", ["", "replace-me", "   "])
def test_github_signature_requires_configured_secret(secret: str) -> None:
    with pytest.raises(HTTPException) as excinfo:
        _verify_github_signature(b"{}", None, secret)
    assert excinfo.value.status_code == 500
    assert "GITHUB_WEBHOOK_SECRET" in str(excinfo.value.detail)


def test_github_signature_rejects_missing_header() -> None:
    with pytest.raises(HTTPException) as excinfo:
        _verify_github_signature(b"{}", None, "super-secret")
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "missing github signature"


def test_github_signature_accepts_valid_signature() -> None:
    body = b'{"action":"opened"}'
    _verify_github_signature(body, _signature("super-secret", body), "super-secret")


@pytest.mark.parametrize("secret", ["", "replace-me", "   "])
def test_shared_secret_requires_configured_secret(secret: str) -> None:
    with pytest.raises(HTTPException) as excinfo:
        _verify_shared_secret("abc", secret)
    assert excinfo.value.status_code == 500
    assert "GITHUB_WEBHOOK_SECRET" in str(excinfo.value.detail)


def test_shared_secret_rejects_invalid_secret() -> None:
    with pytest.raises(HTTPException) as excinfo:
        _verify_shared_secret("wrong", "super-secret")
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "invalid webhook secret"


def test_shared_secret_accepts_valid_secret() -> None:
    _verify_shared_secret("super-secret", "super-secret")
