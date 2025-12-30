"""
Authentication dependencies for FastAPI.

Provides dependency functions for extracting and validating JWTs
from request headers and returning user context to protected endpoints.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth.schemas import UserContext
from api.auth.service import AuthService, AuthenticationError


# Optional bearer token extraction (allows missing token)
_optional_bearer = HTTPBearer(auto_error=False)

# Required bearer token extraction (raises 401 if missing)
_required_bearer = HTTPBearer(auto_error=True)


# Global auth service reference (set during app startup)
_auth_service: Optional[AuthService] = None


def set_auth_service(service: AuthService) -> None:
    """
    Set the global auth service instance.
    
    Called during application startup to inject the auth service.
    
    Args:
        service: AuthService instance.
    """
    global _auth_service
    _auth_service = service


def get_auth_service() -> AuthService:
    """
    Get the auth service instance.
    
    Returns:
        AuthService instance.
    
    Raises:
        RuntimeError: If auth service is not initialized.
    """
    if _auth_service is None:
        raise RuntimeError(
            "AuthService not initialized. Call set_auth_service() during startup."
        )
    return _auth_service


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_bearer),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserContext:
    """
    Get the current authenticated user from the request.
    
    Behavior depends on authentication mode:
    - Open mode: Returns anonymous context if no token provided,
                 validates token if provided.
    - Secure mode: Requires valid token, returns 401 if missing or invalid.
    
    Args:
        credentials: Optional bearer token from Authorization header.
        auth_service: Authentication service instance.
    
    Returns:
        UserContext with user information.
    
    Raises:
        HTTPException: 401 if authentication fails in secure mode.
    """
    # No token provided
    if credentials is None:
        if auth_service.is_open_mode:
            # Open mode: allow anonymous access
            return auth_service.get_open_mode_context()
        else:
            # Secure mode: token required
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Token provided - validate it
    try:
        return auth_service.validate_token(credentials.credentials)
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.description or e.error,
            headers={"WWW-Authenticate": f'Bearer error="{e.error}"'},
        )


async def require_authenticated_user(
    credentials: HTTPAuthorizationCredentials = Depends(_required_bearer),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserContext:
    """
    Require an authenticated user for the request.
    
    Always requires a valid JWT token regardless of authentication mode.
    Use this for endpoints that must have an authenticated user even
    in open mode.
    
    Args:
        credentials: Bearer token from Authorization header (required).
        auth_service: Authentication service instance.
    
    Returns:
        UserContext with user information.
    
    Raises:
        HTTPException: 401 if token is missing or invalid.
    """
    try:
        return auth_service.validate_token(credentials.credentials)
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.description or e.error,
            headers={"WWW-Authenticate": f'Bearer error="{e.error}"'},
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_bearer),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[UserContext]:
    """
    Get the current user if authenticated, otherwise None.
    
    Does not require authentication. Useful for endpoints that behave
    differently for authenticated vs anonymous users.
    
    Args:
        credentials: Optional bearer token from Authorization header.
        auth_service: Authentication service instance.
    
    Returns:
        UserContext if valid token provided, None otherwise.
    """
    if credentials is None:
        return None
    
    try:
        return auth_service.validate_token(credentials.credentials)
    except AuthenticationError:
        return None