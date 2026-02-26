"""
API Tool endpoint schemas.

Pydantic models for API tool CRUD request/response payloads.
Secrets are masked in standard responses and only exposed
via the dedicated secrets endpoint.
"""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator

from practorflow.llm.tools.api.models.enums import (
    AuthKeyLocation,
    AuthType,
    BodyContentType,
    HttpMethod,
    ParamLocation,
    ParamType,
    ParseType,
    RetryBackoff,
)


# ---------------------------------------------------------------------------
# Nested sub-models (mirrors library models for request input)
# ---------------------------------------------------------------------------

class ToolParameterSchema(BaseModel):
    """Tool parameter definition."""
    name: str = Field(..., description="Parameter name")
    location: ParamLocation = Field(..., description="Where the parameter is placed")
    type: ParamType = Field(default=ParamType.STRING, description="Parameter data type")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default_value: Optional[Any] = Field(default=None, description="Default value")
    enum_values: Optional[List[str]] = Field(default=None, description="Allowed values")


class OutputMappingSchema(BaseModel):
    """Output field extraction mapping."""
    field_name: str = Field(..., description="Output field name")
    json_path: str = Field(..., description="JSONPath expression")


class ErrorMappingSchema(BaseModel):
    """Error field extraction mapping."""
    field_name: str = Field(..., description="Error field name")
    json_path: str = Field(..., description="JSONPath expression")


# ---------------------------------------------------------------------------
# Create request
# ---------------------------------------------------------------------------

class ApiToolCreateRequest(BaseModel):
    """
    Request model for creating a new API tool configuration.
    
    Secrets (auth_secret, auth_username, auth_password) are accepted
    as plaintext and encrypted before storage.
    Identity fields (tool_id, user_id, system, created_at, updated_at)
    are set server-side.
    """

    # Basic info
    name: str = Field(..., description="Tool name identifier")
    enabled: bool = Field(default=True, description="Whether tool is active")

    # HTTP configuration
    base_url: str = Field(..., description="Base URL for API")
    method: HttpMethod = Field(default=HttpMethod.GET, description="HTTP method")
    path: str = Field(..., description="Endpoint path (supports templating)")
    body_content_type: BodyContentType = Field(default=BodyContentType.NONE, description="Request body type")

    # Authentication (plaintext — will be encrypted before storage)
    auth_type: AuthType = Field(default=AuthType.NONE, description="Authentication type")
    auth_secret: Optional[str] = Field(default=None, description="Plaintext token/key for API Key or Bearer auth")
    auth_username: Optional[str] = Field(default=None, description="Plaintext username for Basic auth")
    auth_password: Optional[str] = Field(default=None, description="Plaintext password for Basic auth")
    auth_secret_expires_at: Optional[datetime] = Field(default=None, description="Secret expiration")
    auth_key_location: Optional[AuthKeyLocation] = Field(default=None, description="API Key placement location")
    auth_key_name: Optional[str] = Field(default=None, description="API Key parameter name (e.g., appid, X-API-Key)")

    # Reliability
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout")
    retry_max_attempts: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    retry_backoff: RetryBackoff = Field(default=RetryBackoff.EXPONENTIAL, description="Retry backoff strategy")
    retry_on_status: List[int] = Field(default=[429, 500, 502, 503], description="Status codes to retry on")
    rate_limit_rpm: int = Field(default=0, ge=0, description="Requests per minute limit (0=unlimited)")
    rate_limit_concurrent: int = Field(default=0, ge=0, description="Concurrent request limit (0=unlimited)")

    # Agent discovery metadata
    description: str = Field(..., description="What the tool does")
    purpose: str = Field(default="", description="When and why to use it")
    keywords: List[str] = Field(..., min_length=1, description="Trigger words for matching")
    category: str = Field(default="", description="Domain grouping")
    tags: List[str] = Field(default_factory=list, description="Additional classification")
    use_when: List[str] = Field(default_factory=list, description="Positive use case examples")
    do_not_use_when: List[str] = Field(default_factory=list, description="Anti-patterns to avoid")
    requires: List[str] = Field(default_factory=list, description="Input prerequisites")
    returns: str = Field(default="", description="Output description")

    # Parameters
    parameters: List[ToolParameterSchema] = Field(default_factory=list, description="Tool parameters")

    # Response handling
    success_codes: List[int] = Field(default=[200], description="HTTP codes indicating success")
    parse_as: ParseType = Field(default=ParseType.JSON, description="Response parsing type")
    output_mapping: List[OutputMappingSchema] = Field(default_factory=list, description="Output field extraction")
    error_mapping: List[ErrorMappingSchema] = Field(default_factory=list, description="Error field extraction")

    # System tool flag (requires llm_admin)
    system: bool = Field(default=False, description="Whether tool is system-level (requires llm_admin)")

    @model_validator(mode="after")
    def validate_auth_fields(self) -> "ApiToolCreateRequest":
        """Validate authentication field requirements based on auth_type."""
        if self.auth_type == AuthType.API_KEY:
            if not self.auth_secret:
                raise ValueError("auth_secret is required when auth_type is api_key")
            if not self.auth_key_name:
                raise ValueError("auth_key_name is required when auth_type is api_key")
            if not self.auth_key_location:
                raise ValueError("auth_key_location is required when auth_type is api_key")
        elif self.auth_type == AuthType.BEARER:
            if not self.auth_secret:
                raise ValueError("auth_secret is required when auth_type is bearer")
        elif self.auth_type == AuthType.BASIC:
            if not self.auth_username:
                raise ValueError("auth_username is required when auth_type is basic")
            if not self.auth_password:
                raise ValueError("auth_password is required when auth_type is basic")
        return self


# ---------------------------------------------------------------------------
# Update request
# ---------------------------------------------------------------------------

class ApiToolUpdateRequest(BaseModel):
    """
    Request model for updating an API tool configuration.
    
    All fields are optional — only provided fields are updated.
    Secrets are accepted as plaintext and re-encrypted before storage.
    """

    # Basic info
    name: Optional[str] = Field(default=None, description="Tool name identifier")
    enabled: Optional[bool] = Field(default=None, description="Whether tool is active")

    # HTTP configuration
    base_url: Optional[str] = Field(default=None, description="Base URL for API")
    method: Optional[HttpMethod] = Field(default=None, description="HTTP method")
    path: Optional[str] = Field(default=None, description="Endpoint path (supports templating)")
    body_content_type: Optional[BodyContentType] = Field(default=None, description="Request body type")

    # Authentication (plaintext — will be encrypted before storage)
    auth_type: Optional[AuthType] = Field(default=None, description="Authentication type")
    auth_secret: Optional[str] = Field(default=None, description="Plaintext token/key for API Key or Bearer auth")
    auth_username: Optional[str] = Field(default=None, description="Plaintext username for Basic auth")
    auth_password: Optional[str] = Field(default=None, description="Plaintext password for Basic auth")
    auth_secret_expires_at: Optional[datetime] = Field(default=None, description="Secret expiration")
    auth_key_location: Optional[AuthKeyLocation] = Field(default=None, description="API Key placement location")
    auth_key_name: Optional[str] = Field(default=None, description="API Key parameter name (e.g., appid, X-API-Key)")

    # Reliability
    timeout_seconds: Optional[int] = Field(default=None, ge=1, le=300, description="Request timeout")
    retry_max_attempts: Optional[int] = Field(default=None, ge=0, le=10, description="Max retry attempts")
    retry_backoff: Optional[RetryBackoff] = Field(default=None, description="Retry backoff strategy")
    retry_on_status: Optional[List[int]] = Field(default=None, description="Status codes to retry on")
    rate_limit_rpm: Optional[int] = Field(default=None, ge=0, description="Requests per minute limit (0=unlimited)")
    rate_limit_concurrent: Optional[int] = Field(default=None, ge=0, description="Concurrent request limit (0=unlimited)")

    # Agent discovery metadata
    description: Optional[str] = Field(default=None, description="What the tool does")
    purpose: Optional[str] = Field(default=None, description="When and why to use it")
    keywords: Optional[List[str]] = Field(default=None, min_length=1, description="Trigger words for matching")
    category: Optional[str] = Field(default=None, description="Domain grouping")
    tags: Optional[List[str]] = Field(default=None, description="Additional classification")
    use_when: Optional[List[str]] = Field(default=None, description="Positive use case examples")
    do_not_use_when: Optional[List[str]] = Field(default=None, description="Anti-patterns to avoid")
    requires: Optional[List[str]] = Field(default=None, description="Input prerequisites")
    returns: Optional[str] = Field(default=None, description="Output description")

    # Parameters
    parameters: Optional[List[ToolParameterSchema]] = Field(default=None, description="Tool parameters")

    # Response handling
    success_codes: Optional[List[int]] = Field(default=None, description="HTTP codes indicating success")
    parse_as: Optional[ParseType] = Field(default=None, description="Response parsing type")
    output_mapping: Optional[List[OutputMappingSchema]] = Field(default=None, description="Output field extraction")
    error_mapping: Optional[List[ErrorMappingSchema]] = Field(default=None, description="Error field extraction")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ApiToolResponse(BaseModel):
    """
    Response model for a single API tool configuration.
    
    Secrets are masked: returns ``"***"`` if a value is present,
    ``null`` if not set.
    """

    # Identity
    tool_id: str = Field(..., description="Unique tool ID")
    user_id: str = Field(..., description="Owner user ID")
    system: bool = Field(..., description="Whether tool is system-level")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    # Basic info
    name: str = Field(..., description="Tool name identifier")

    # HTTP configuration
    base_url: str = Field(..., description="Base URL for API")
    method: HttpMethod = Field(..., description="HTTP method")
    path: str = Field(..., description="Endpoint path")
    body_content_type: BodyContentType = Field(..., description="Request body type")

    # Authentication (masked)
    auth_type: AuthType = Field(..., description="Authentication type")
    auth_secret: Optional[str] = Field(default=None, description="Masked secret ('***' if set, null if not)")
    auth_username: Optional[str] = Field(default=None, description="Masked username ('***' if set, null if not)")
    auth_password: Optional[str] = Field(default=None, description="Masked password ('***' if set, null if not)")
    auth_secret_expires_at: Optional[datetime] = Field(default=None, description="Secret expiration")
    auth_key_location: Optional[AuthKeyLocation] = Field(default=None, description="API Key placement location")
    auth_key_name: Optional[str] = Field(default=None, description="API Key parameter name")

    # Reliability
    timeout_seconds: int = Field(..., description="Request timeout")
    retry_max_attempts: int = Field(..., description="Max retry attempts")
    retry_backoff: RetryBackoff = Field(..., description="Retry backoff strategy")
    retry_on_status: List[int] = Field(..., description="Status codes to retry on")
    rate_limit_rpm: int = Field(..., description="Requests per minute limit")
    rate_limit_concurrent: int = Field(..., description="Concurrent request limit")

    # Agent discovery metadata
    description: str = Field(..., description="What the tool does")
    purpose: str = Field(..., description="When and why to use it")
    keywords: List[str] = Field(..., description="Trigger words for matching")
    category: str = Field(..., description="Domain grouping")
    tags: List[str] = Field(..., description="Additional classification")
    use_when: List[str] = Field(..., description="Positive use case examples")
    do_not_use_when: List[str] = Field(..., description="Anti-patterns to avoid")
    requires: List[str] = Field(..., description="Input prerequisites")
    returns: str = Field(..., description="Output description")

    # Parameters
    parameters: List[ToolParameterSchema] = Field(..., description="Tool parameters")

    # Response handling
    success_codes: List[int] = Field(..., description="HTTP codes indicating success")
    parse_as: ParseType = Field(..., description="Response parsing type")
    output_mapping: List[OutputMappingSchema] = Field(..., description="Output field extraction")
    error_mapping: List[ErrorMappingSchema] = Field(..., description="Error field extraction")

    @staticmethod
    def mask_secret(value: Optional[str]) -> Optional[str]:
        """Return '***' if value is present, None otherwise."""
        return "***" if value else None

    @classmethod
    def from_config(cls, config) -> "ApiToolResponse":
        """
        Build response from an ApiToolConfig, masking secrets.
        
        Args:
            config: ApiToolConfig instance.
        
        Returns:
            ApiToolResponse with masked secrets.
        """
        return cls(
            tool_id=config.tool_id,
            user_id=config.user_id,
            system=config.system,
            created_at=config.created_at,
            updated_at=config.updated_at,
            name=config.name,
            base_url=config.base_url,
            method=config.method,
            path=config.path,
            body_content_type=config.body_content_type,
            auth_type=config.auth_type,
            auth_secret=cls.mask_secret(config.auth_secret),
            auth_username=cls.mask_secret(config.auth_username),
            auth_password=cls.mask_secret(config.auth_password),
            auth_secret_expires_at=config.auth_secret_expires_at,
            auth_key_location=config.auth_key_location,
            auth_key_name=config.auth_key_name,
            timeout_seconds=config.timeout_seconds,
            retry_max_attempts=config.retry_max_attempts,
            retry_backoff=config.retry_backoff,
            retry_on_status=config.retry_on_status,
            rate_limit_rpm=config.rate_limit_rpm,
            rate_limit_concurrent=config.rate_limit_concurrent,
            description=config.description,
            purpose=config.purpose,
            keywords=config.keywords,
            category=config.category,
            tags=config.tags,
            use_when=config.use_when,
            do_not_use_when=config.do_not_use_when,
            requires=config.requires,
            returns=config.returns,
            parameters=[
                ToolParameterSchema(
                    name=p.name,
                    location=p.location,
                    type=p.type,
                    description=p.description,
                    required=p.required,
                    default_value=p.default_value,
                    enum_values=p.enum_values,
                )
                for p in config.parameters
            ],
            success_codes=config.success_codes,
            parse_as=config.parse_as,
            output_mapping=[
                OutputMappingSchema(field_name=m.field_name, json_path=m.json_path)
                for m in config.output_mapping
            ],
            error_mapping=[
                ErrorMappingSchema(field_name=m.field_name, json_path=m.json_path)
                for m in config.error_mapping
            ],
        )


class ApiToolSecretsResponse(BaseModel):
    """
    Response model for the secrets endpoint.
    
    Returns decrypted plaintext values for auth fields.
    """

    tool_id: str = Field(..., description="Tool ID")
    auth_secret: Optional[str] = Field(default=None, description="Decrypted token/key")
    auth_username: Optional[str] = Field(default=None, description="Decrypted username")
    auth_password: Optional[str] = Field(default=None, description="Decrypted password")


class ApiToolListResponse(BaseModel):
    """Response model for listing API tools."""

    tools: List[ApiToolResponse] = Field(..., description="List of tool configurations")
    count: int = Field(..., description="Total number of tools")