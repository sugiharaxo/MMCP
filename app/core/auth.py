"""
Authentication layer for MMCP settings endpoints.

Zero-friction auth with opt-in security via .admin_token file.
"""

import os
import secrets
from pathlib import Path

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.core.logger import logger

# Security scheme for Bearer token
security = HTTPBearer()


def get_admin_token_path() -> Path:
    """Get the path to the admin token file."""
    return Path(settings.root_dir) / ".admin_token"


def ensure_admin_token() -> str:
    """
    Ensure admin token exists. Generate if missing.

    Called at startup if require_auth is True.

    Returns:
        The admin token (existing or newly generated)
    """
    token_path = get_admin_token_path()

    if token_path.exists():
        # Read existing token
        token = token_path.read_text(encoding="utf-8").strip()
        logger.info("Using existing admin token from .admin_token file")
        return token

    # Generate new token (32 characters, URL-safe)
    token = secrets.token_urlsafe(32)
    token_path.write_text(token, encoding="utf-8")

    # Set restrictive permissions on non-Windows
    if os.name != "nt":
        try:
            os.chmod(token_path, 0o600)
        except OSError as e:
            logger.warning(f"Failed to set admin token file permissions: {e}")

    logger.info(f"Auth enabled. Admin token saved to {token_path}")
    logger.warning(
        f"IMPORTANT: Save this token securely. You'll need it to access settings endpoints: {token}"
    )
    return token


async def verify_admin(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> bool:
    """
    Verify admin token from Authorization header.

    Dependency for FastAPI routes that require authentication.

    Args:
        credentials: HTTPBearer credentials from Authorization header

    Returns:
        True if token is valid

    Raises:
        HTTPException: If token is missing or invalid
    """
    if not settings.require_auth:
        # Auth not required - allow access
        return True

    token_path = get_admin_token_path()
    if not token_path.exists():
        # Token file doesn't exist - this shouldn't happen if ensure_admin_token() was called
        logger.error("Admin token file not found but require_auth is True")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is required but token file is missing. Check server logs.",
        )

    expected_token = token_path.read_text(encoding="utf-8").strip()
    provided_token = credentials.credentials

    if not secrets.compare_digest(expected_token, provided_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True
