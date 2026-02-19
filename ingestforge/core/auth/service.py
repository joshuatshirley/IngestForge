"""Authentication Service.

Handles password hashing, verification, and JWT token management.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Parameter Validation).
"""

from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import jwt
from passlib.context import CryptContext

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
SECRET_KEY = os.getenv(
    "INGESTFORGE_JWT_SECRET_KEY",
    "DEVELOPMENT_SECRET_KEY_CHANGE_IN_PRODUCTION",  # Fallback for dev only
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 Hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Service for managing user security and identity."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Create a bcrypt hash from a plain password."""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Check if a plain password matches its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(data: Dict[str, Any]) -> str:
        """Generate a signed JWT token."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Validate and decode a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except Exception as e:
            logger.debug(f"Token verification failed: {e}")
            return None
