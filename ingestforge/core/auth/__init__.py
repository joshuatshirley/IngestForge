"""
Authentication and Security Services.

JWT Stateless Authentication with RS256 support.

Components:
- JWTHandler: RS256/HS256 token signing and verification
- JWTAuthMiddleware: FastAPI middleware for global auth
- JWTAuthDependency: FastAPI dependency for route-level auth
- RequireRole: Role-based access control dependency
- AuthService: Password hashing and legacy token support
"""

from ingestforge.core.auth.service import AuthService
from ingestforge.core.auth.jwt_handler import (
    JWTHandler,
    TokenPayload,
    RSAKeyPair,
    generate_rsa_key_pair,
    create_jwt_handler,
    ALGORITHM_RS256,
    ALGORITHM_HS256,
    MAX_TOKEN_LIFETIME_HOURS,
    MAX_ROLES_PER_TOKEN,
)
from ingestforge.core.auth.middleware import (
    JWTAuthMiddleware,
    JWTAuthDependency,
    RequireRole,
    map_roles_to_acl,
    get_highest_role,
)

__all__ = [
    # JWT Handler
    "JWTHandler",
    "TokenPayload",
    "RSAKeyPair",
    "generate_rsa_key_pair",
    "create_jwt_handler",
    "ALGORITHM_RS256",
    "ALGORITHM_HS256",
    "MAX_TOKEN_LIFETIME_HOURS",
    "MAX_ROLES_PER_TOKEN",
    # Middleware
    "JWTAuthMiddleware",
    "JWTAuthDependency",
    "RequireRole",
    "map_roles_to_acl",
    "get_highest_role",
    # Legacy
    "AuthService",
]
