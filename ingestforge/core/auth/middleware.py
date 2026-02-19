"""
FastAPI JWT Authentication Middleware.

FastAPI middleware for JWT validation.
Follows NASA JPL Power of Ten rules.

Features:
- Middleware-based JWT validation
- Integration with JWTHandler (RS256/HS256)
- Maps roles to IFPermissionRegistry
- Clear 401 errors for expired/invalid tokens
"""

from __future__ import annotations

from typing import Callable, List, Optional
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ingestforge.core.auth.jwt_handler import (
    JWTHandler,
    TokenPayload,
    create_jwt_handler,
)
from ingestforge.core.security.acl import Role
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# JPL RULE #2: FIXED UPPER BOUNDS
# =============================================================================

MAX_TOKEN_LENGTH = 4096  # Maximum token string length


# =============================================================================
# ROLE TO PERMISSION MAPPING
# =============================================================================

ROLE_MAP = {
    "owner": Role.OWNER,
    "admin": Role.OWNER,  # Admin maps to owner
    "editor": Role.EDITOR,
    "contributor": Role.CONTRIBUTOR,
    "viewer": Role.VIEWER,
}


def map_roles_to_acl(roles: List[str]) -> List[Role]:
    """
    Map JWT role strings to ACL Role enums.

    Map roles to IFPermissionRegistry.
    Rule #4: < 60 lines.

    Args:
        roles: List of role strings from JWT.

    Returns:
        List of ACL Role enums.
    """
    mapped: List[Role] = []

    for role_str in roles:
        role_lower = role_str.lower()
        if role_lower in ROLE_MAP:
            mapped.append(ROLE_MAP[role_lower])

    return mapped


def get_highest_role(roles: List[str]) -> Optional[Role]:
    """
    Get the highest privilege role from a list.

    Rule #4: < 60 lines.

    Args:
        roles: List of role strings.

    Returns:
        Highest Role enum, or None if no valid roles.
    """
    mapped = map_roles_to_acl(roles)

    if not mapped:
        return None

    # Role hierarchy: OWNER > EDITOR > CONTRIBUTOR > VIEWER
    hierarchy = [Role.OWNER, Role.EDITOR, Role.CONTRIBUTOR, Role.VIEWER]

    for role in hierarchy:
        if role in mapped:
            return role

    return None


# =============================================================================
# JWT SECURITY SCHEME
# =============================================================================

security = HTTPBearer(auto_error=False)


async def get_token_from_header(request: Request) -> Optional[str]:
    """
    Extract JWT from Authorization header.

    Rule #2: Bound token length.
    Rule #4: < 60 lines.
    Rule #7: Validate token format.

    Args:
        request: FastAPI request.

    Returns:
        Token string or None.
    """
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        return None

    # Validate format
    parts = auth_header.split()
    if len(parts) != 2:
        return None

    scheme, token = parts

    if scheme.lower() != "bearer":
        return None

    # Bound token length (Rule #2)
    if len(token) > MAX_TOKEN_LENGTH:
        logger.warning("Token exceeds maximum length")
        return None

    return token


# =============================================================================
# AUTHENTICATION DEPENDENCY
# =============================================================================


class JWTAuthDependency:
    """
    FastAPI dependency for JWT authentication.

    FastAPI middleware for JWT validation.

    Example:
        auth = JWTAuthDependency()

        @app.get("/protected")
        async def protected(user: TokenPayload = Depends(auth)):
            return {"user_id": user.user_id}
    """

    def __init__(
        self,
        handler: Optional[JWTHandler] = None,
        required: bool = True,
    ) -> None:
        """
        Initialize JWT auth dependency.

        Rule #5: Validate parameters, store handler instance.

        Args:
            handler: JWTHandler instance (created if None).
            required: If True, raises 401 on missing/invalid token.
        """
        # JPL Rule #5: Validate types
        assert handler is None or isinstance(
            handler, JWTHandler
        ), "handler must be JWTHandler"
        assert isinstance(required, bool), "required must be bool"

        self._handler = handler or create_jwt_handler()
        self._required = required

    async def __call__(self, request: Request) -> Optional[TokenPayload]:
        """
        Validate JWT and return payload.

        AC: Reject expired tokens with clear 401 error.
        Rule #4: < 60 lines.
        Rule #7: Validate token.
        Rule #10: Cryptographic verification.

        Args:
            request: FastAPI request.

        Returns:
            TokenPayload if valid.

        Raises:
            HTTPException: 401 if token invalid/missing (when required).
        """
        token = await get_token_from_header(request)

        if token is None:
            if self._required:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing authentication token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return None

        # Verify token (Rule #10: cryptographic verification)
        payload = self._handler.verify_token(token)

        if payload is None:
            if self._required:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return None

        # Attach user info to request state
        request.state.user = payload
        request.state.user_id = payload.user_id
        request.state.workspace_id = payload.workspace_id
        request.state.roles = payload.roles

        return payload


# =============================================================================
# MIDDLEWARE CLASS
# =============================================================================


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for global JWT authentication.

    FastAPI middleware for JWT validation.

    Note: This middleware allows certain paths without authentication.
    For more granular control, use JWTAuthDependency on specific routes.

    Example:
        app.add_middleware(
            JWTAuthMiddleware,
            handler=create_jwt_handler(),
            exclude_paths=["/health", "/v1/auth/login"],
        )
    """

    def __init__(
        self,
        app,
        handler: Optional[JWTHandler] = None,
        exclude_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize JWT middleware.

        Rule #5: Validate parameter types.

        Args:
            app: FastAPI app.
            handler: JWTHandler instance.
            exclude_paths: Paths that don't require authentication.
        """
        # JPL Rule #5: Validate types
        assert handler is None or isinstance(
            handler, JWTHandler
        ), "handler must be JWTHandler"
        assert exclude_paths is None or isinstance(
            exclude_paths, list
        ), "exclude_paths must be list"

        super().__init__(app)
        self._handler = handler or create_jwt_handler()
        self._exclude_paths = set(exclude_paths or [])

        # Default excluded paths
        self._exclude_paths.update(
            [
                "/health",
                "/v1/health",
                "/docs",
                "/openapi.json",
                "/v1/auth/login",
            ]
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request and validate JWT.

        AC: Reject expired tokens with 401.
        Rule #4: < 60 lines.
        Rule #7: Validate request.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from downstream handler.
        """
        path = request.url.path

        # Skip excluded paths
        if path in self._exclude_paths:
            return await call_next(request)

        # Skip preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Extract and validate token
        token = await get_token_from_header(request)

        if token is None:
            return Response(
                content='{"detail": "Missing authentication token"}',
                status_code=401,
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )

        payload = self._handler.verify_token(token)

        if payload is None:
            return Response(
                content='{"detail": "Invalid or expired token"}',
                status_code=401,
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Attach user to request state
        request.state.user = payload
        request.state.user_id = payload.user_id
        request.state.workspace_id = payload.workspace_id
        request.state.roles = payload.roles

        return await call_next(request)


# =============================================================================
# ROLE-BASED ACCESS CONTROL DEPENDENCY
# =============================================================================


class RequireRole:
    """
    FastAPI dependency that requires specific roles.

    Example:
        @app.post("/admin-only")
        async def admin_only(
            user: TokenPayload = Depends(RequireRole(["admin", "owner"]))
        ):
            return {"message": "Welcome, admin!"}
    """

    def __init__(
        self,
        allowed_roles: List[str],
        handler: Optional[JWTHandler] = None,
    ) -> None:
        """
        Initialize role requirement.

        Rule #5: Validate parameter types.

        Args:
            allowed_roles: List of role names that can access.
            handler: JWTHandler instance.
        """
        # JPL Rule #5: Validate types
        assert isinstance(allowed_roles, list), "allowed_roles must be list"
        assert len(allowed_roles) > 0, "allowed_roles cannot be empty"
        assert handler is None or isinstance(
            handler, JWTHandler
        ), "handler must be JWTHandler"

        self._allowed_roles = set(r.lower() for r in allowed_roles)
        self._auth = JWTAuthDependency(handler=handler, required=True)

    async def __call__(self, request: Request) -> TokenPayload:
        """
        Validate JWT and check roles.

        Rule #4: < 60 lines.

        Args:
            request: FastAPI request.

        Returns:
            TokenPayload if authorized.

        Raises:
            HTTPException: 401 if not authenticated, 403 if not authorized.
        """
        payload = await self._auth(request)

        # Check if user has any allowed role
        user_roles = set(r.lower() for r in payload.roles)

        if not user_roles.intersection(self._allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {', '.join(self._allowed_roles)}",
            )

        return payload
