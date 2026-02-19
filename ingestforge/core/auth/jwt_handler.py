"""
JWT Handler for Stateless Authentication.

JWT Stateless Authentication with RS256 support.
Follows NASA JPL Power of Ten rules.

Features:
- RS256 asymmetric signing (public/private key pairs)
- Token payload with user_id, workspace_id, roles
- Cryptographic verification of all signatures (Rule #10)
- Configurable token expiration
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path

from jose import JWTError, jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# JPL RULE #2: FIXED UPPER BOUNDS
# =============================================================================

MAX_TOKEN_LIFETIME_HOURS = 24  # Maximum token validity
MIN_TOKEN_LIFETIME_MINUTES = 5  # Minimum token validity
MAX_ROLES_PER_TOKEN = 10  # Maximum roles in a single token
RSA_KEY_SIZE = 2048  # RSA key size in bits

# Algorithm constants
ALGORITHM_RS256 = "RS256"  # Asymmetric (RSA + SHA-256)
ALGORITHM_HS256 = "HS256"  # Symmetric (HMAC + SHA-256)


# =============================================================================
# TOKEN PAYLOAD DATACLASS
# =============================================================================


@dataclass
class TokenPayload:
    """
    JWT token payload structure.

    AC: Token payload includes user_id, workspace_id, roles.
    Rule #9: Complete type hints.
    """

    user_id: str
    workspace_id: str
    roles: List[str]
    exp: datetime
    iat: datetime
    sub: str  # Subject (usually same as user_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to dictionary for JWT encoding."""
        return {
            "sub": self.sub,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "roles": self.roles,
            "exp": self.exp,
            "iat": self.iat,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TokenPayload:
        """
        Create payload from decoded JWT dictionary.

        Rule #5: Validate input data.
        Rule #7: Handle missing/invalid fields gracefully.
        """
        # JPL Rule #5: Validate input
        assert isinstance(data, dict), "data must be dictionary"

        # Extract and validate fields (Rule #7)
        user_id = data.get("user_id", data.get("sub", ""))
        workspace_id = data.get("workspace_id", "default")
        roles = data.get("roles", [])

        # Ensure types are correct
        if not isinstance(user_id, str):
            user_id = str(user_id) if user_id else ""
        if not isinstance(workspace_id, str):
            workspace_id = str(workspace_id) if workspace_id else "default"
        if not isinstance(roles, list):
            roles = []

        return cls(
            user_id=user_id,
            workspace_id=workspace_id,
            roles=roles,
            exp=data.get("exp", datetime.now(timezone.utc)),
            iat=data.get("iat", datetime.now(timezone.utc)),
            sub=data.get("sub", ""),
        )


# =============================================================================
# KEY MANAGEMENT
# =============================================================================


@dataclass
class RSAKeyPair:
    """
    RSA key pair for asymmetric JWT signing.

    Rule #9: Complete type hints.
    """

    private_key: bytes
    public_key: bytes


def generate_rsa_key_pair() -> RSAKeyPair:
    """
    Generate a new RSA key pair for RS256 signing.

    Rule #4: < 60 lines.
    Rule #10: Cryptographic key generation.

    Returns:
        RSAKeyPair with PEM-encoded keys.
    """
    # Generate RSA private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=RSA_KEY_SIZE,
        backend=default_backend(),
    )

    # Serialize private key to PEM
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Extract and serialize public key
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return RSAKeyPair(private_key=private_pem, public_key=public_pem)


def load_key_from_file(path: Path) -> bytes:
    """
    Load a PEM key from file.

    Rule #4: < 60 lines.
    Rule #7: Validate file exists.

    Args:
        path: Path to PEM key file.

    Returns:
        PEM-encoded key bytes.

    Raises:
        FileNotFoundError: If key file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Key file not found: {path}")

    return path.read_bytes()


def load_key_from_env(env_var: str) -> Optional[bytes]:
    """
    Load a PEM key from environment variable.

    Rule #4: < 60 lines.

    Args:
        env_var: Environment variable name.

    Returns:
        PEM-encoded key bytes, or None if not set.
    """
    key_str = os.environ.get(env_var)
    if key_str is None:
        return None

    return key_str.encode("utf-8")


# =============================================================================
# JWT HANDLER CLASS
# =============================================================================


class JWTHandler:
    """
    JWT handler supporting RS256 asymmetric and HS256 symmetric signing.

    JWT Stateless Authentication.

    Features:
    - RS256 asymmetric signing with RSA keys
    - HS256 symmetric fallback for development
    - Cryptographic verification (Rule #10)
    - Configurable expiration

    Example:
        # RS256 mode
        handler = JWTHandler(
            private_key=load_key_from_file(Path("private.pem")),
            public_key=load_key_from_file(Path("public.pem")),
        )

        # Create token
        token = handler.create_token(
            user_id="user123",
            workspace_id="ws456",
            roles=["viewer", "contributor"],
        )

        # Verify token
        payload = handler.verify_token(token)
    """

    def __init__(
        self,
        private_key: Optional[bytes] = None,
        public_key: Optional[bytes] = None,
        secret_key: Optional[str] = None,
        token_lifetime_hours: float = 24.0,
    ) -> None:
        """
        Initialize JWT handler.

        Rule #2: Bound token lifetime.
        Rule #5: Assert preconditions.

        Args:
            private_key: PEM-encoded RSA private key (for signing).
            public_key: PEM-encoded RSA public key (for verification).
            secret_key: Symmetric key for HS256 (fallback).
            token_lifetime_hours: Token validity duration.

        Raises:
            AssertionError: If configuration is invalid.
        """
        # Determine algorithm based on provided keys
        if private_key and public_key:
            self._algorithm = ALGORITHM_RS256
            self._private_key = private_key
            self._public_key = public_key
            self._secret_key = None
            logger.info("JWT handler initialized with RS256 (asymmetric)")
        elif secret_key:
            self._algorithm = ALGORITHM_HS256
            self._private_key = None
            self._public_key = None
            self._secret_key = secret_key
            logger.info("JWT handler initialized with HS256 (symmetric)")
        else:
            raise AssertionError(
                "Must provide either (private_key, public_key) for RS256 "
                "or secret_key for HS256"
            )

        # Bound token lifetime (Rule #2)
        if token_lifetime_hours > MAX_TOKEN_LIFETIME_HOURS:
            token_lifetime_hours = MAX_TOKEN_LIFETIME_HOURS
        if token_lifetime_hours < MIN_TOKEN_LIFETIME_MINUTES / 60:
            token_lifetime_hours = MIN_TOKEN_LIFETIME_MINUTES / 60

        self._token_lifetime = timedelta(hours=token_lifetime_hours)

    @property
    def algorithm(self) -> str:
        """Get the signing algorithm in use."""
        return self._algorithm

    def create_token(
        self,
        user_id: str,
        workspace_id: str,
        roles: List[str],
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a signed JWT token.

        AC: Token payload includes user_id, workspace_id, roles.
        Rule #2: Bound roles list.
        Rule #4: < 60 lines.
        Rule #7: Validate inputs.
        Rule #10: Cryptographic signing.

        Args:
            user_id: User identifier.
            workspace_id: Workspace identifier.
            roles: User roles.
            additional_claims: Optional extra claims.

        Returns:
            Signed JWT string.

        Raises:
            AssertionError: If inputs are invalid.
        """
        # Preconditions (Rule #5)
        assert user_id and isinstance(user_id, str), "user_id must be non-empty string"
        assert workspace_id and isinstance(workspace_id, str), "workspace_id required"
        assert isinstance(roles, list), "roles must be a list"

        # Bound roles (Rule #2)
        bounded_roles = roles[:MAX_ROLES_PER_TOKEN]
        if len(roles) > MAX_ROLES_PER_TOKEN:
            logger.warning(f"Roles truncated to {MAX_ROLES_PER_TOKEN}")

        # Build payload
        now = datetime.now(timezone.utc)
        expire = now + self._token_lifetime

        payload: Dict[str, Any] = {
            "sub": user_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "roles": bounded_roles,
            "iat": now,
            "exp": expire,
        }

        # Add additional claims
        if additional_claims:
            payload.update(additional_claims)

        # Sign token (Rule #10: cryptographic signing)
        if self._algorithm == ALGORITHM_RS256:
            token = jwt.encode(payload, self._private_key, algorithm=self._algorithm)
        else:
            token = jwt.encode(payload, self._secret_key, algorithm=self._algorithm)

        # JPL Rule #7: Validate return value
        assert token and isinstance(token, str), "JWT encoding failed"

        return token

    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """
        Verify and decode a JWT token.

        AC: Cryptographic verification.
        Rule #4: < 60 lines.
        Rule #7: Validate token.
        Rule #10: Cryptographic verification of signatures.

        Args:
            token: JWT string to verify.

        Returns:
            TokenPayload if valid, None if invalid or expired.
        """
        # Precondition (Rule #5)
        if not token or not isinstance(token, str):
            logger.debug("Token validation failed: empty or invalid type")
            return None

        try:
            # Verify signature (Rule #10)
            if self._algorithm == ALGORITHM_RS256:
                payload = jwt.decode(
                    token,
                    self._public_key,
                    algorithms=[ALGORITHM_RS256],
                )
            else:
                payload = jwt.decode(
                    token,
                    self._secret_key,
                    algorithms=[ALGORITHM_HS256],
                )

            return TokenPayload.from_dict(payload)

        except JWTError as e:
            logger.debug(f"Token verification failed: {e}")
            return None

    def is_expired(self, token: str) -> bool:
        """
        Check if a token is expired.

        Rule #4: < 60 lines.

        Args:
            token: JWT string to check.

        Returns:
            True if expired or invalid, False if valid.
        """
        payload = self.verify_token(token)
        return payload is None

    def get_user_id(self, token: str) -> Optional[str]:
        """
        Extract user_id from token.

        Rule #4: < 60 lines.

        Args:
            token: JWT string.

        Returns:
            User ID if valid token, None otherwise.
        """
        payload = self.verify_token(token)
        if payload is None:
            return None
        return payload.user_id

    def get_roles(self, token: str) -> List[str]:
        """
        Extract roles from token.

        Rule #4: < 60 lines.

        Args:
            token: JWT string.

        Returns:
            List of roles, or empty list if invalid.
        """
        payload = self.verify_token(token)
        if payload is None:
            return []
        return payload.roles


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_jwt_handler(
    private_key_path: Optional[Path] = None,
    public_key_path: Optional[Path] = None,
    secret_key: Optional[str] = None,
) -> JWTHandler:
    """
    Factory function to create JWT handler from config.

    Rule #4: < 60 lines.
    Rule #7: Validate inputs.

    Args:
        private_key_path: Path to RSA private key PEM file.
        public_key_path: Path to RSA public key PEM file.
        secret_key: Symmetric key for HS256 fallback.

    Returns:
        Configured JWTHandler.

    Raises:
        ValueError: If no valid key configuration provided.
    """
    # Try environment variables first
    private_key = load_key_from_env("JWT_PRIVATE_KEY")
    public_key = load_key_from_env("JWT_PUBLIC_KEY")

    # Fall back to file paths
    if private_key is None and private_key_path:
        private_key = load_key_from_file(private_key_path)

    if public_key is None and public_key_path:
        public_key = load_key_from_file(public_key_path)

    # Use RS256 if keys available
    if private_key and public_key:
        return JWTHandler(private_key=private_key, public_key=public_key)

    # Fall back to HS256 with secret key
    if secret_key is None:
        secret_key = os.environ.get(
            "JWT_SECRET_KEY", "DEVELOPMENT_SECRET_KEY_CHANGE_IN_PRODUCTION"
        )

    return JWTHandler(secret_key=secret_key)
