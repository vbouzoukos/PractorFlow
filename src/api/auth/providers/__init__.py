"""
Authentication providers module.

Provides authentication provider implementations for different authentication sources.
"""

from api.auth.providers.base import AuthProvider, AuthResult
from api.auth.providers.local import LocalAuthProvider
from api.auth.providers.oidc import OIDCAuthProvider

__all__ = [
    "AuthProvider",
    "AuthResult",
    "LocalAuthProvider",
    "OIDCAuthProvider",
]