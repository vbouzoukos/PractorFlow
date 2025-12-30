"""
Authentication module.

Provides JWT-based authentication for the PractorFlow API.

Supports:
- Local authentication with app secret
- Open mode (no credentials required)
- OIDC authentication with external providers
"""

from api.auth.schemas import (
    AuthError,
    TokenRequest,
    TokenResponse,
    UserContext,
)
from api.auth.jwt_handler import (
    JWTHandler,
    JWTError,
    JWTExpiredError,
    JWTInvalidError,
)
from api.auth.service import (
    AuthService,
    AuthenticationError,
)
from api.auth.dependencies import (
    get_auth_service,
    get_current_user,
    get_optional_user,
    require_authenticated_user,
    set_auth_service,
)

__all__ = [
    # Schemas
    "AuthError",
    "TokenRequest",
    "TokenResponse",
    "UserContext",
    # JWT Handler
    "JWTHandler",
    "JWTError",
    "JWTExpiredError",
    "JWTInvalidError",
    # Service
    "AuthService",
    "AuthenticationError",
    # Dependencies
    "get_auth_service",
    "get_current_user",
    "get_optional_user",
    "require_authenticated_user",
    "set_auth_service",
]