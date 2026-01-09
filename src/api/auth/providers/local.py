"""
Local authentication provider.

Validates credentials against a configured application secret.
When the secret is empty, operates in open mode and issues tokens
without credential validation.
"""

import secrets
import uuid

from api.auth.providers.base import AuthProvider, AuthResult


class LocalAuthProvider(AuthProvider):
    """
    Local authentication provider.
    
    Validates the provided app_secret against the configured secret.
    In open mode (no configured secret), tokens are issued without validation.
    """
    
    def __init__(self, app_secret: str = ""):
        """
        Initialize local authentication provider.
        
        Args:
            app_secret: Configured application secret.
                       Empty string enables open mode.
        """
        self._app_secret = app_secret
    
    @property
    def provider_name(self) -> str:
        """Get the provider name identifier."""
        return "local"
    
    @property
    def requires_credentials(self) -> bool:
        """
        Check if this provider requires credential validation.
        
        Returns:
            False if app_secret is empty (open mode), True otherwise.
        """
        return bool(self._app_secret)
    
    @property
    def is_open_mode(self) -> bool:
        """Check if provider is operating in open mode."""
        return not self._app_secret
    
    async def validate(
        self,
        credentials: dict,
    ) -> AuthResult:
        """
        Validate local credentials.
        
        In open mode, always succeeds and generates a user ID if not provided.
        In secure mode, validates the provided app_secret.
        
        Args:
            credentials: Dictionary with optional keys:
                        - app_secret: Secret to validate
                        - username: Optional username for user_id
        
        Returns:
            AuthResult with success status and user information.
        """
        provided_secret = credentials.get("app_secret", "")
        username = credentials.get("username")
        
        # Open mode: issue token without validation
        if self.is_open_mode:
            user_id = username if username else self._generate_anonymous_id()
            return AuthResult(
                success=True,
                user_id=user_id,
            )
        
        # Secure mode: validate the provided secret
        if not provided_secret:
            return AuthResult(
                success=False,
                error="missing_credentials",
                error_description="Application secret is required",
            )
        
        # Use constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(provided_secret, self._app_secret):
            return AuthResult(
                success=False,
                error="invalid_credentials",
                error_description="Invalid application secret",
            )
        
        # Validation successful
        user_id = username if username else "local_user"
        return AuthResult(
            success=True,
            user_id=user_id,
        )
    
    def _generate_anonymous_id(self) -> str:
        """
        Generate an anonymous user ID for open mode.
        
        Returns:
            Anonymous user identifier string.
        """
        return f"anonymous_{uuid.uuid4().hex[:8]}"