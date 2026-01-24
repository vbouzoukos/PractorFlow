"""
Authentication Service.

Unified authentication service that coordinates authentication flow
from credential validation through JWT issuance.
"""

from typing import Optional

from api.auth.jwt_handler import JWTHandler, JWTError
from api.auth.providers import AuthProvider, AuthResult, LocalAuthProvider, OIDCAuthProvider
from api.auth.schemas import TokenRequest, TokenResponse, UserContext, AuthError
from api.config import AuthConfig


class AuthenticationError(Exception):
    """Exception raised for authentication failures."""
    
    def __init__(self, error: str, description: Optional[str] = None):
        self.error = error
        self.description = description
        super().__init__(description or error)
    
    def to_auth_error(self) -> AuthError:
        """Convert to AuthError schema for API response."""
        return AuthError(
            error=self.error,
            error_description=self.description,
        )


class AuthService:
    """
    Unified authentication service.
    
    Selects the appropriate provider based on configuration and coordinates
    the authentication flow from credential validation through JWT issuance.
    """
    
    def __init__(self, config: AuthConfig):
        """
        Initialize authentication service.
        
        Args:
            config: Authentication configuration settings.
        """
        self._config = config
        self._jwt_handler = JWTHandler(config.jwt)
        self._provider = self._create_provider()
    
    def _create_provider(self) -> AuthProvider:
        """
        Create the appropriate authentication provider based on configuration.
        
        Returns:
            AuthProvider instance (LocalAuthProvider or OIDCAuthProvider).
        """
        if self._config.is_oidc_mode:
            return OIDCAuthProvider(self._config.oidc)
        else:
            return LocalAuthProvider(self._config.app_secret)
    
    @property
    def provider_name(self) -> str:
        """Get the active provider name."""
        return self._provider.provider_name
    
    @property
    def is_open_mode(self) -> bool:
        """Check if authentication is in open mode."""
        return self._config.is_open_mode
    
    @property
    def requires_credentials(self) -> bool:
        """Check if authentication requires credential validation."""
        return self._provider.requires_credentials
    
    async def authenticate(self, request: TokenRequest) -> TokenResponse:
        """
        Authenticate a token request and issue a JWT.
        
        Validates credentials using the configured provider and issues
        an API JWT upon successful authentication.
        
        Args:
            request: Token request with credentials.
        
        Returns:
            TokenResponse with the issued JWT.
        
        Raises:
            AuthenticationError: If authentication fails.
        """
        # Build credentials dictionary based on provider type
        if self._config.is_oidc_mode:
            credentials = {
                "identity_token": request.identity_token,
            }
        else:
            credentials = {
                "app_secret": request.app_secret,
                "username": request.username,
            }
        
        # Validate credentials with provider
        result: AuthResult = await self._provider.validate(credentials)
        
        if not result.success:
            raise AuthenticationError(
                error=result.error or "authentication_failed",
                description=result.error_description,
            )
        
        # Issue API JWT
        try:
            token = self._jwt_handler.create_token(result.user_id)
        except JWTError as e:
            raise AuthenticationError(
                error=e.error,
                description=e.description,
            )
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=self._jwt_handler.expiry_seconds,
        )
    
    def validate_token(self, token: str) -> UserContext:
        """
        Validate a JWT and extract user context.
        
        Args:
            token: JWT access token.
        
        Returns:
            UserContext with user information.
        
        Raises:
            AuthenticationError: If token validation fails.
        """
        try:
            return self._jwt_handler.validate_token(token)
        except JWTError as e:
            raise AuthenticationError(
                error=e.error,
                description=e.description,
            )
    
    def get_open_mode_context(self, username: Optional[str] = None) -> UserContext:
        """
        Get a user context for open mode requests.
        
        Used when authentication is in open mode and no token is provided.
        
        Args:
            username: Optional username to use as user_id.
        
        Returns:
            UserContext with anonymous or provided user_id.
        """
        import uuid
        
        user_id = username if username else f"anonymous_{uuid.uuid4().hex[:8]}"
        
        return UserContext(
            user_id=user_id,
            is_authenticated=False,
        )