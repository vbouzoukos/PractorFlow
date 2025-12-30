"""
OIDC authentication provider.

Validates identity tokens from external OIDC providers and extracts
user information for API JWT issuance.
"""

from typing import Any, Dict, Optional

import httpx
import jwt
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError

from api.auth.providers.base import AuthProvider, AuthResult
from api.config import OIDCConfig


class OIDCAuthProvider(AuthProvider):
    """
    OIDC authentication provider.
    
    Validates identity tokens from external OIDC providers by:
    1. Fetching the provider's JWKS (JSON Web Key Set)
    2. Validating the token signature and claims
    3. Extracting user information from the token
    """
    
    def __init__(self, config: OIDCConfig):
        """
        Initialize OIDC authentication provider.
        
        Args:
            config: OIDC configuration settings.
        """
        self._config = config
        self._issuer_url = config.issuer_url.rstrip("/")
        self._audience = config.audience
        self._client_id = config.client_id
        self._jwks_client: Optional[PyJWKClient] = None
        self._oidc_config: Optional[Dict[str, Any]] = None
    
    @property
    def provider_name(self) -> str:
        """Get the provider name identifier."""
        return "oidc"
    
    @property
    def requires_credentials(self) -> bool:
        """OIDC always requires credential validation."""
        return True
    
    async def validate(
        self,
        credentials: dict,
    ) -> AuthResult:
        """
        Validate OIDC identity token.
        
        Args:
            credentials: Dictionary with key:
                        - identity_token: JWT from OIDC provider
        
        Returns:
            AuthResult with success status and user information.
        """
        identity_token = credentials.get("identity_token")
        
        if not identity_token:
            return AuthResult(
                success=False,
                error="missing_credentials",
                error_description="Identity token is required",
            )
        
        try:
            # Fetch OIDC configuration if not cached
            if self._oidc_config is None:
                await self._fetch_oidc_config()
            
            # Initialize JWKS client if not cached
            if self._jwks_client is None:
                jwks_uri = self._oidc_config.get("jwks_uri")
                if not jwks_uri:
                    return AuthResult(
                        success=False,
                        error="provider_error",
                        error_description="OIDC provider missing JWKS URI",
                    )
                self._jwks_client = PyJWKClient(jwks_uri)
            
            # Get signing key from JWKS
            signing_key = self._jwks_client.get_signing_key_from_jwt(identity_token)
            
            # Validate and decode the token
            payload = jwt.decode(
                identity_token,
                signing_key.key,
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
                audience=self._audience or self._client_id,
                issuer=self._issuer_url,
                options={
                    "require": ["sub", "exp", "iat", "iss"],
                },
            )
            
            # Extract user identifier
            user_id = self._extract_user_id(payload)
            
            return AuthResult(
                success=True,
                user_id=user_id,
            )
        
        except InvalidTokenError as e:
            return AuthResult(
                success=False,
                error="invalid_token",
                error_description=f"Token validation failed: {str(e)}",
            )
        
        except httpx.HTTPError as e:
            return AuthResult(
                success=False,
                error="provider_error",
                error_description=f"Failed to contact OIDC provider: {str(e)}",
            )
        
        except Exception as e:
            return AuthResult(
                success=False,
                error="validation_error",
                error_description=f"Unexpected error during validation: {str(e)}",
            )
    
    async def _fetch_oidc_config(self) -> None:
        """
        Fetch OIDC provider configuration from well-known endpoint.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        well_known_url = f"{self._issuer_url}/.well-known/openid-configuration"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(well_known_url, timeout=10.0)
            response.raise_for_status()
            self._oidc_config = response.json()
    
    def _extract_user_id(self, payload: Dict[str, Any]) -> str:
        """
        Extract user identifier from token payload.
        
        Attempts to extract a meaningful user ID from standard claims,
        falling back to the subject claim.
        
        Args:
            payload: Decoded JWT payload.
        
        Returns:
            User identifier string.
        """
        # Try common user identifier claims in order of preference
        for claim in ["preferred_username", "email", "sub"]:
            if claim in payload and payload[claim]:
                return str(payload[claim])
        
        # Fallback to subject (always required)
        return str(payload["sub"])
    
    def clear_cache(self) -> None:
        """Clear cached OIDC configuration and JWKS client."""
        self._oidc_config = None
        self._jwks_client = None