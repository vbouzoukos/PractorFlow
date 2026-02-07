"""
Authentication schemas.

Pydantic models for authentication requests and responses.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    """
    Request model for obtaining a JWT token.
    
    In local mode: Provide app_secret (or omit if open mode).
    In OIDC mode: Provide identity_token from external provider.
    """
    
    app_secret: Optional[str] = Field(
        default=None,
        description="Application secret for local authentication",
    )
    admin_secret: Optional[str] = Field(
        default=None,
        description="Admin secret for obtaining llm_admin permission (local mode only)",
    )
    identity_token: Optional[str] = Field(
        default=None,
        description="Identity token from OIDC provider",
    )
    username: Optional[str] = Field(
        default=None,
        description="Optional username for token subject (local mode only)",
    )


class TokenResponse(BaseModel):
    """Response model containing the issued JWT token."""
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class UserContext(BaseModel):
    """
    User context extracted from authenticated request.
    
    Maps to the user field in Session model.
    """
    
    user_id: str = Field(..., description="User identifier (JWT subject claim)")
    is_authenticated: bool = Field(
        default=True,
        description="Whether user is authenticated (False in open mode)",
    )
    permissions: List[str] = Field(
        default_factory=list,
        description="List of granted permissions (e.g. llm_admin)",
    )


class AuthError(BaseModel):
    """Error response for authentication failures."""
    
    error: str = Field(..., description="Error code")
    error_description: Optional[str] = Field(
        default=None,
        description="Human-readable error description",
    )