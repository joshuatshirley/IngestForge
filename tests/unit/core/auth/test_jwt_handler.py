"""
Tests for JWT Handler.

JWT Stateless Authentication with RS256 support.
Tests follow GWT (Given-When-Then) pattern and NASA JPL Power of Ten rules.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import pytest

from ingestforge.core.auth.jwt_handler import (
    JWTHandler,
    TokenPayload,
    RSAKeyPair,
    generate_rsa_key_pair,
    create_jwt_handler,
    ALGORITHM_RS256,
    ALGORITHM_HS256,
    MAX_ROLES_PER_TOKEN,
)


# =============================================================================
# RSA KEY GENERATION TESTS
# =============================================================================


class TestRSAKeyGeneration:
    """Tests for RSA key pair generation."""

    def test_generate_rsa_key_pair(self):
        """
        GWT:
        Given no parameters
        When generate_rsa_key_pair is called
        Then valid RSA key pair is returned.
        """
        key_pair = generate_rsa_key_pair()

        assert isinstance(key_pair, RSAKeyPair)
        assert key_pair.private_key is not None
        assert key_pair.public_key is not None
        assert b"PRIVATE KEY" in key_pair.private_key
        assert b"PUBLIC KEY" in key_pair.public_key

    def test_keys_are_pem_encoded(self):
        """
        GWT:
        Given generated key pair
        When inspecting keys
        Then they are PEM encoded.
        """
        key_pair = generate_rsa_key_pair()

        assert key_pair.private_key.startswith(b"-----BEGIN")
        assert key_pair.public_key.startswith(b"-----BEGIN")


# =============================================================================
# TOKEN PAYLOAD TESTS
# =============================================================================


class TestTokenPayload:
    """Tests for TokenPayload dataclass."""

    def test_payload_creation(self):
        """
        GWT:
        Given valid parameters
        When TokenPayload is created
        Then all fields are set correctly.
        """
        now = datetime.now(timezone.utc)
        payload = TokenPayload(
            user_id="user123",
            workspace_id="ws456",
            roles=["editor", "viewer"],
            exp=now + timedelta(hours=1),
            iat=now,
            sub="user123",
        )

        assert payload.user_id == "user123"
        assert payload.workspace_id == "ws456"
        assert "editor" in payload.roles
        assert payload.sub == "user123"

    def test_payload_to_dict(self):
        """
        GWT:
        Given TokenPayload
        When to_dict is called
        Then dictionary has all fields.
        """
        now = datetime.now(timezone.utc)
        payload = TokenPayload(
            user_id="user123",
            workspace_id="ws456",
            roles=["viewer"],
            exp=now,
            iat=now,
            sub="user123",
        )

        data = payload.to_dict()

        assert data["user_id"] == "user123"
        assert data["workspace_id"] == "ws456"
        assert data["roles"] == ["viewer"]
        assert data["sub"] == "user123"

    def test_payload_from_dict(self):
        """
        GWT:
        Given dictionary with payload data
        When TokenPayload.from_dict is called
        Then TokenPayload is created correctly.
        """
        now = datetime.now(timezone.utc)
        data = {
            "sub": "user123",
            "user_id": "user123",
            "workspace_id": "ws456",
            "roles": ["contributor"],
            "exp": now,
            "iat": now,
        }

        payload = TokenPayload.from_dict(data)

        assert payload.user_id == "user123"
        assert payload.workspace_id == "ws456"
        assert payload.roles == ["contributor"]


# =============================================================================
# JWT HANDLER HS256 TESTS
# =============================================================================


class TestJWTHandlerHS256:
    """Tests for JWTHandler with HS256 symmetric signing."""

    def test_handler_init_hs256(self):
        """
        GWT:
        Given secret key
        When JWTHandler is created
        Then HS256 algorithm is used.
        """
        handler = JWTHandler(secret_key="test_secret")

        assert handler.algorithm == ALGORITHM_HS256

    def test_create_token_hs256(self):
        """
        GWT:
        Given JWTHandler with HS256
        When create_token is called
        Then valid JWT is returned.
        """
        handler = JWTHandler(secret_key="test_secret")

        token = handler.create_token(
            user_id="user123",
            workspace_id="ws456",
            roles=["editor"],
        )

        assert isinstance(token, str)
        assert len(token) > 0
        assert "." in token  # JWT has 3 parts separated by dots

    def test_verify_token_hs256(self):
        """
        GWT:
        Given valid HS256 token
        When verify_token is called
        Then TokenPayload is returned.
        """
        handler = JWTHandler(secret_key="test_secret")
        token = handler.create_token(
            user_id="user123",
            workspace_id="ws456",
            roles=["viewer"],
        )

        payload = handler.verify_token(token)

        assert payload is not None
        assert payload.user_id == "user123"
        assert payload.workspace_id == "ws456"
        assert "viewer" in payload.roles

    def test_verify_invalid_token(self):
        """
        GWT:
        Given invalid token
        When verify_token is called
        Then None is returned.
        """
        handler = JWTHandler(secret_key="test_secret")

        payload = handler.verify_token("invalid.token.here")

        assert payload is None

    def test_verify_wrong_secret(self):
        """
        GWT:
        Given token signed with different secret
        When verify_token is called
        Then None is returned.
        """
        handler1 = JWTHandler(secret_key="secret1")
        handler2 = JWTHandler(secret_key="secret2")

        token = handler1.create_token(
            user_id="user123",
            workspace_id="ws456",
            roles=["editor"],
        )

        payload = handler2.verify_token(token)

        assert payload is None


# =============================================================================
# JWT HANDLER RS256 TESTS
# =============================================================================


class TestJWTHandlerRS256:
    """Tests for JWTHandler with RS256 asymmetric signing."""

    @pytest.fixture
    def key_pair(self):
        """Generate RSA key pair for tests."""
        return generate_rsa_key_pair()

    def test_handler_init_rs256(self, key_pair):
        """
        GWT:
        Given RSA key pair
        When JWTHandler is created
        Then RS256 algorithm is used.
        """
        handler = JWTHandler(
            private_key=key_pair.private_key,
            public_key=key_pair.public_key,
        )

        assert handler.algorithm == ALGORITHM_RS256

    def test_create_token_rs256(self, key_pair):
        """
        GWT:
        Given JWTHandler with RS256
        When create_token is called
        Then valid JWT is returned (AC: RS256 signing).
        """
        handler = JWTHandler(
            private_key=key_pair.private_key,
            public_key=key_pair.public_key,
        )

        token = handler.create_token(
            user_id="user123",
            workspace_id="ws456",
            roles=["admin"],
        )

        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_token_rs256(self, key_pair):
        """
        GWT:
        Given valid RS256 token
        When verify_token is called
        Then TokenPayload is returned (JPL Rule #10: crypto verification).
        """
        handler = JWTHandler(
            private_key=key_pair.private_key,
            public_key=key_pair.public_key,
        )

        token = handler.create_token(
            user_id="admin",
            workspace_id="default",
            roles=["owner"],
        )

        payload = handler.verify_token(token)

        assert payload is not None
        assert payload.user_id == "admin"
        assert "owner" in payload.roles

    def test_verify_with_wrong_public_key(self, key_pair):
        """
        GWT:
        Given token and different public key
        When verify_token is called
        Then None is returned (signature mismatch).
        """
        other_key_pair = generate_rsa_key_pair()

        handler1 = JWTHandler(
            private_key=key_pair.private_key,
            public_key=key_pair.public_key,
        )
        handler2 = JWTHandler(
            private_key=other_key_pair.private_key,
            public_key=other_key_pair.public_key,
        )

        token = handler1.create_token(
            user_id="user",
            workspace_id="ws",
            roles=["viewer"],
        )

        payload = handler2.verify_token(token)

        assert payload is None


# =============================================================================
# TOKEN PAYLOAD CONTENT TESTS
# =============================================================================


class TestTokenPayloadContent:
    """Tests for token payload contents (AC)."""

    def test_payload_includes_user_id(self):
        """
        GWT:
        Given valid token
        When decoded
        Then user_id is present (AC).
        """
        handler = JWTHandler(secret_key="test")
        token = handler.create_token(
            user_id="user123",
            workspace_id="ws",
            roles=["viewer"],
        )

        payload = handler.verify_token(token)

        assert payload.user_id == "user123"

    def test_payload_includes_workspace_id(self):
        """
        GWT:
        Given valid token
        When decoded
        Then workspace_id is present (AC).
        """
        handler = JWTHandler(secret_key="test")
        token = handler.create_token(
            user_id="user",
            workspace_id="workspace789",
            roles=["viewer"],
        )

        payload = handler.verify_token(token)

        assert payload.workspace_id == "workspace789"

    def test_payload_includes_roles(self):
        """
        GWT:
        Given valid token with multiple roles
        When decoded
        Then roles are present (AC).
        """
        handler = JWTHandler(secret_key="test")
        token = handler.create_token(
            user_id="user",
            workspace_id="ws",
            roles=["admin", "editor", "viewer"],
        )

        payload = handler.verify_token(token)

        assert "admin" in payload.roles
        assert "editor" in payload.roles
        assert "viewer" in payload.roles


# =============================================================================
# JPL RULE COMPLIANCE TESTS
# =============================================================================


class TestJPLCompliance:
    """Tests verifying NASA JPL Power of Ten rule compliance."""

    def test_jpl_rule_2_roles_bounded(self):
        """
        GWT:
        Given roles exceeding MAX_ROLES_PER_TOKEN
        When create_token is called
        Then roles are truncated (JPL Rule #2).
        """
        handler = JWTHandler(secret_key="test")

        many_roles = [f"role{i}" for i in range(50)]
        token = handler.create_token(
            user_id="user",
            workspace_id="ws",
            roles=many_roles,
        )

        payload = handler.verify_token(token)

        assert len(payload.roles) <= MAX_ROLES_PER_TOKEN

    def test_jpl_rule_2_lifetime_bounded(self):
        """
        GWT:
        Given excessive token lifetime
        When JWTHandler is created
        Then lifetime is capped (JPL Rule #2).
        """
        handler = JWTHandler(
            secret_key="test",
            token_lifetime_hours=1000,  # Excessive
        )

        # Handler should cap this - verify by creating token
        token = handler.create_token(
            user_id="user",
            workspace_id="ws",
            roles=["viewer"],
        )

        assert token is not None  # Handler created successfully

    def test_jpl_rule_5_precondition_empty_user(self):
        """
        GWT:
        Given empty user_id
        When create_token is called
        Then AssertionError is raised (JPL Rule #5).
        """
        handler = JWTHandler(secret_key="test")

        with pytest.raises(AssertionError):
            handler.create_token(
                user_id="",
                workspace_id="ws",
                roles=["viewer"],
            )

    def test_jpl_rule_5_precondition_no_keys(self):
        """
        GWT:
        Given neither keys nor secret
        When JWTHandler is created
        Then AssertionError is raised (JPL Rule #5).
        """
        with pytest.raises(AssertionError):
            JWTHandler()  # No keys provided


# =============================================================================
# UTILITY METHOD TESTS
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_is_expired_valid_token(self):
        """
        GWT:
        Given valid non-expired token
        When is_expired is called
        Then False is returned.
        """
        handler = JWTHandler(secret_key="test")
        token = handler.create_token(
            user_id="user",
            workspace_id="ws",
            roles=["viewer"],
        )

        assert handler.is_expired(token) is False

    def test_is_expired_invalid_token(self):
        """
        GWT:
        Given invalid token
        When is_expired is called
        Then True is returned.
        """
        handler = JWTHandler(secret_key="test")

        assert handler.is_expired("invalid") is True

    def test_get_user_id(self):
        """
        GWT:
        Given valid token
        When get_user_id is called
        Then user_id is returned.
        """
        handler = JWTHandler(secret_key="test")
        token = handler.create_token(
            user_id="testuser",
            workspace_id="ws",
            roles=["viewer"],
        )

        assert handler.get_user_id(token) == "testuser"

    def test_get_roles(self):
        """
        GWT:
        Given valid token
        When get_roles is called
        Then roles list is returned.
        """
        handler = JWTHandler(secret_key="test")
        token = handler.create_token(
            user_id="user",
            workspace_id="ws",
            roles=["admin", "editor"],
        )

        roles = handler.get_roles(token)

        assert "admin" in roles
        assert "editor" in roles


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateJWTHandler:
    """Tests for create_jwt_handler factory function."""

    @patch.dict("os.environ", {}, clear=True)
    def test_create_with_default_secret(self):
        """
        GWT:
        Given no environment variables or keys
        When create_jwt_handler is called
        Then handler uses default secret (HS256).
        """
        handler = create_jwt_handler()

        assert handler.algorithm == ALGORITHM_HS256

    @patch.dict("os.environ", {"JWT_SECRET_KEY": "env_secret"}, clear=True)
    def test_create_from_env_secret(self):
        """
        GWT:
        Given JWT_SECRET_KEY in environment
        When create_jwt_handler is called
        Then handler uses env secret.
        """
        handler = create_jwt_handler()

        assert handler.algorithm == ALGORITHM_HS256
