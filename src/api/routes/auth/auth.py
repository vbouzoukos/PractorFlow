"""
Authentication API endpoints.

Provides endpoints for:
- Obtaining JWT tokens via local or OIDC authentication
- Checking authentication status
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth.dependencies import get_auth_service, get_current_user
from api.auth.schemas import TokenRequest, TokenResponse, UserContext, AuthError
from api.auth.service import AuthService, AuthenticationError


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/token",
    response_model=TokenResponse,
    responses={
        401: {"model": AuthError, "description": "Authentication failed"},
        500: {"model": AuthError, "description": "Server error"},
    },
    summary="Obtain access token",
    description=(
        "Authenticate and obtain a JWT access token. "
        "In local mode, provide app_secret (or omit if open mode). "
        "In OIDC mode, provide identity_token from external provider."
    ),
)
async def obtain_token(
    request: TokenRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """
    Obtain a JWT access token.
    
    Authentication method depends on server configuration:
    - Local mode with secret: Provide app_secret
    - Local mode open: No credentials required
    - OIDC mode: Provide identity_token
    
    Args:
        request: Token request with credentials.
        auth_service: Authentication service instance.
    
    Returns:
        TokenResponse with JWT access token.
    
    Raises:
        HTTPException: 401 if authentication fails.
    """
    try:
        return await auth_service.authenticate(request)
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.to_auth_error().model_dump(),
            headers={"WWW-Authenticate": f'Bearer error="{e.error}"'},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "server_error",
                "error_description": str(e),
            },
        )


@router.get(
    "/me",
    response_model=UserContext,
    responses={
        401: {"model": AuthError, "description": "Not authenticated"},
    },
    summary="Get current user",
    description="Get information about the currently authenticated user.",
)
async def get_me(
    current_user: UserContext = Depends(get_current_user),
) -> UserContext:
    """
    Get current user information.
    
    Returns the user context extracted from the JWT token.
    In open mode without a token, returns anonymous user context.
    
    Args:
        current_user: Current user from authentication.
    
    Returns:
        UserContext with user information.
    """
    return current_user


@router.get(
    "/status",
    summary="Get authentication status",
    description="Get information about the authentication configuration.",
)
async def get_auth_status(
    auth_service: AuthService = Depends(get_auth_service),
) -> dict:
    """
    Get authentication status and configuration.
    
    Returns information about the current authentication mode
    without revealing sensitive configuration.
    
    Args:
        auth_service: Authentication service instance.
    
    Returns:
        Dictionary with authentication status.
    """
    return {
        "provider": auth_service.provider_name,
        "requires_credentials": auth_service.requires_credentials,
        "is_open_mode": auth_service.is_open_mode,
    }