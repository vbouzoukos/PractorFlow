"""
JWT Handler.

Provides JWT token creation and validation functionality.
All tokens are issued by the API regardless of authentication source.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional

import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError

from api.auth.schemas import UserContext
from api.config import JWTConfig


class JWTError(Exception):
    """Base exception for JWT-related errors."""
    
    def __init__(self, error: str, description: Optional[str] = None):
        self.error = error
        self.description = description
        super().__init__(description or error)


class JWTExpiredError(JWTError):
    """Raised when a JWT token has expired."""
    
    def __init__(self):
        super().__init__(
            error="token_expired",
            description="The access token has expired",
        )


class JWTInvalidError(JWTError):
    """Raised when a JWT token is invalid."""
    
    def __init__(self, reason: Optional[str] = None):
        super().__init__(
            error="invalid_token",
            description=reason or "The access token is invalid",
        )


class JWTHandler:
    """
    Handles JWT token creation and validation.
    
    All tokens issued by this handler use the configured algorithm
    and secret key. The subject claim maps to the user field in
    the Session model.
    """
    
    def __init__(self, config: JWTConfig):
        """
        Initialize JWT handler.
        
        Args:
            config: JWT configuration settings.
        """
        self._config = config
        self._algorithm = config.algorithm
        self._secret_key = config.secret_key
        self._expiry_minutes = config.token_expiry_minutes
    
    @property
    def expiry_seconds(self) -> int:
        """Get token expiry time in seconds."""
        return self._expiry_minutes * 60
    
    def create_token(
        self,
        user_id: str,
        permissions: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new JWT token for the given user.
        
        Args:
            user_id: User identifier to set as the subject claim.
                    Maps to the user field in Session model.
            permissions: Optional list of granted permissions to embed
                        in the token (e.g. ["llm_admin"]).
        
        Returns:
            Encoded JWT token string.
        
        Raises:
            JWTError: If token creation fails.
        """
        if not self._secret_key:
            raise JWTError(
                error="configuration_error",
                description="JWT secret key is not configured",
            )
        
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=self._expiry_minutes)
        
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": expires_at,
            "iss": "practorflow-api",
            "permissions": permissions or [],
        }
        
        try:
            token = jwt.encode(
                payload,
                self._secret_key,
                algorithm=self._algorithm,
            )
            return token
        except Exception as e:
            raise JWTError(
                error="token_creation_failed",
                description=f"Failed to create token: {str(e)}",
            )
    
    def validate_token(self, token: str) -> UserContext:
        """
        Validate a JWT token and extract user context.
        
        Args:
            token: Encoded JWT token string.
        
        Returns:
            UserContext with user information from token claims.
        
        Raises:
            JWTExpiredError: If token has expired.
            JWTInvalidError: If token is invalid or malformed.
        """
        if not self._secret_key:
            raise JWTError(
                error="configuration_error",
                description="JWT secret key is not configured",
            )
        
        try:
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
                options={
                    "require": ["sub", "exp", "iat"],
                },
            )
            
            user_id = payload.get("sub")
            if not user_id:
                raise JWTInvalidError("Token missing subject claim")
            
            permissions = payload.get("permissions", [])
            
            return UserContext(
                user_id=user_id,
                is_authenticated=True,
                permissions=permissions,
            )
        
        except ExpiredSignatureError:
            raise JWTExpiredError()
        
        except InvalidTokenError as e:
            raise JWTInvalidError(str(e))
        
        except Exception as e:
            raise JWTInvalidError(f"Token validation failed: {str(e)}")