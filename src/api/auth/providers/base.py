"""
Authentication provider base class.

Defines the abstract interface that all authentication providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AuthResult:
    """
    Result from authentication provider validation.
    
    Contains normalized user information regardless of authentication source.
    """
    
    success: bool
    user_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    error: Optional[str] = None
    error_description: Optional[str] = None


class AuthProvider(ABC):
    """
    Abstract base class for authentication providers.
    
    Implementations validate credentials specific to their type
    and return normalized user information.
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the provider name identifier.
        
        Returns:
            Provider name string (e.g., "local", "oidc").
        """
        pass  # pragma: no cover
    
    @abstractmethod
    async def validate(
        self,
        credentials: dict,
    ) -> AuthResult:
        """
        Validate credentials and return authentication result.
        
        Args:
            credentials: Dictionary containing provider-specific credentials.
                        Local: {"app_secret": "...", "username": "..."}
                        OIDC: {"identity_token": "..."}
        
        Returns:
            AuthResult with success status and user information or error details.
        """
        pass  # pragma: no cover
    
    @property
    @abstractmethod
    def requires_credentials(self) -> bool:
        """
        Check if this provider requires credential validation.
        
        Returns:
            True if credentials must be validated, False for open mode.
        """
        pass  # pragma: no cover