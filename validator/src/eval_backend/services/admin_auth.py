from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import Settings
from ..models import AdminSession, AdminUser

PASSWORD_ITERATIONS = 390_000
PASSWORD_ALGORITHM = "pbkdf2_sha256"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def hash_password(password: str, *, salt: bytes | None = None, iterations: int = PASSWORD_ITERATIONS) -> str:
    salt_bytes = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, iterations)
    return f"{PASSWORD_ALGORITHM}${iterations}${_b64(salt_bytes)}${_b64(derived)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, hash_text = encoded.split("$", 3)
        if algorithm != PASSWORD_ALGORITHM:
            return False
        iterations = int(iterations_text)
        salt = _unb64(salt_text)
        expected = _unb64(hash_text)
    except Exception:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def seed_admin_user(session: Session, settings: Settings) -> AdminUser:
    username = settings.admin_username.strip()
    password = settings.admin_password
    existing = session.execute(select(AdminUser).where(AdminUser.username == username)).scalar_one_or_none()
    password_hash = hash_password(password)
    if existing is None:
        user = AdminUser(username=username, password_hash=password_hash, is_active=True)
        session.add(user)
        session.flush()
        return user

    existing.password_hash = password_hash
    existing.is_active = True
    existing.updated_at = _utcnow()
    session.flush()
    return existing


def create_admin_session(session: Session, user: AdminUser, settings: Settings) -> tuple[AdminSession, str]:
    raw_token = secrets.token_urlsafe(32)
    token_hash = hash_token(raw_token)
    expires_at = _utcnow() + timedelta(hours=max(1, settings.admin_session_ttl_hours))
    admin_session = AdminSession(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    session.add(admin_session)
    session.flush()
    return admin_session, raw_token


def authenticate_admin_token(session: Session, token: str) -> AdminUser:
    token = token.strip()
    if not token:
        raise HTTPException(status_code=401, detail="missing admin token")
    token_hash = hash_token(token)
    admin_session = session.execute(
        select(AdminSession).where(AdminSession.token_hash == token_hash)
    ).scalar_one_or_none()
    if admin_session is None:
        raise HTTPException(status_code=401, detail="invalid admin token")
    if admin_session.revoked_at is not None or admin_session.expires_at <= _utcnow():
        raise HTTPException(status_code=401, detail="expired admin token")
    user = session.get(AdminUser, admin_session.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="inactive admin user")
    admin_session.last_used_at = _utcnow()
    session.flush()
    return user


def revoke_admin_token(session: Session, token: str) -> None:
    token_hash = hash_token(token.strip())
    admin_session = session.execute(
        select(AdminSession).where(AdminSession.token_hash == token_hash)
    ).scalar_one_or_none()
    if admin_session is None:
        return
    admin_session.revoked_at = _utcnow()
    session.flush()
