"""
Local authentication provider.

Validates credentials against a configured application secret.
When the secret is empty, operates in open mode and issues tokens
without credential validation.
"""

import secrets
import uuid

from api.auth.providers.base import AuthProvider, AuthResult
from api.config import AuthConfig


class LocalAuthProvider(AuthProvider):
    """
    Local authentication provider.
    
    Validates the provided app_secret against the configured secret.
    In open mode (no configured secret), tokens are issued without validation.
    """
    
    def __init__(self, config: AuthConfig):
        """
        Initialize local authentication provider.
        
        Args:
            config: Authentication configuration settings.
        """
        self._app_secret = config.app_secret
        self._admin_secret = config.admin_secret
    
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
        If admin_secret is not set, llm_admin permission is granted automatically.
        In secure mode, validates the provided app_secret.
        If admin_secret is also provided and valid, llm_admin permission is granted.
        
        Args:
            credentials: Dictionary with optional keys:
                        - app_secret: Secret to validate
                        - admin_secret: Secret for llm_admin permission
                        - username: Optional username for user_id
        
        Returns:
            AuthResult with success status and user information.
        """
        provided_secret = credentials.get("app_secret", "")
        provided_admin_secret = credentials.get("admin_secret", "")
        username = credentials.get("username")
        
        # Open mode: issue token without validation
        if self.is_open_mode:
            user_id = username if username else self._generate_anonymous_id()
            if self._admin_secret:
                if (
                    provided_admin_secret
                    and secrets.compare_digest(provided_admin_secret, self._admin_secret)
                ):
                    permissions = ["llm_admin"]
                else:
                    permissions = []
            else:
                permissions = ["llm_admin"]
            return AuthResult(
                success=True,
                user_id=user_id,
                permissions=permissions,
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
        
        # Validation successful - check for admin permission
        permissions = []
        if self._admin_secret:
            if (
                provided_admin_secret
                and secrets.compare_digest(provided_admin_secret, self._admin_secret)
            ):
                permissions.append("llm_admin")
        else:
            permissions.append("llm_admin")
        
        user_id = username if username else "local_user"
        return AuthResult(
            success=True,
            user_id=user_id,
            permissions=permissions,
        )
    
    def _generate_anonymous_id(self) -> str:
        """
        Generate an anonymous user ID for open mode.
        
        Returns:
            Anonymous user identifier string.
        """
        return f"anonymous_{uuid.uuid4().hex[:8]}"