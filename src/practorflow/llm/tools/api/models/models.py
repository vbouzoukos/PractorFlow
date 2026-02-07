"""
Pydantic models for API Tool configuration.

Defines the data model for tool configurations with embedded encrypted secrets.
"""

from datetime import datetime
from typing import Any, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

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


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str = Field(..., description="Parameter name")
    location: ParamLocation = Field(..., description="Where the parameter is placed")
    type: ParamType = Field(default=ParamType.STRING, description="Parameter data type")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default_value: Optional[Any] = Field(default=None, description="Default value")
    enum_values: Optional[List[str]] = Field(default=None, description="Allowed values")


class OutputMapping(BaseModel):
    """Mapping for extracting fields from response."""
    field_name: str = Field(..., description="Output field name")
    json_path: str = Field(..., description="JSONPath expression")


class ErrorMapping(BaseModel):
    """Mapping for extracting error fields from response."""
    field_name: str = Field(..., description="Error field name")
    json_path: str = Field(..., description="JSONPath expression")


class ApiToolConfig(BaseModel):
    """Complete API tool configuration with embedded secret."""
    
    # Identity
    tool_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique tool ID")
    user_id: str = Field(..., description="Owner user ID")
    system: bool = Field(default=False, description="Whether tool is system-level (global scope)")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Basic info
    name: str = Field(..., description="Tool name identifier")
    enabled: bool = Field(default=True, description="Whether tool is active")
    
    # HTTP configuration
    base_url: str = Field(..., description="Base URL for API")
    method: HttpMethod = Field(default=HttpMethod.GET, description="HTTP method")
    path: str = Field(..., description="Endpoint path (supports templating)")
    body_content_type: BodyContentType = Field(default=BodyContentType.NONE, description="Request body type")
    
    # Authentication
    auth_type: AuthType = Field(default=AuthType.NONE, description="Authentication type")
    # Used by: API Key, Bearer - encrypted token/key value
    auth_secret: Optional[str] = Field(default=None, description="Encrypted token/key for API Key or Bearer auth")
    # Used by: Basic - encrypted username
    auth_username: Optional[str] = Field(default=None, description="Encrypted username for Basic auth")
    # Used by: Basic - encrypted password
    auth_password: Optional[str] = Field(default=None, description="Encrypted password for Basic auth")
    # Used by: All - optional expiration timestamp
    auth_secret_expires_at: Optional[datetime] = Field(default=None, description="Secret expiration")
    # Used by: API Key - where to place the key (header, query, or body)
    auth_key_location: Optional[AuthKeyLocation] = Field(default=None, description="API Key placement location")
    # Used by: API Key - parameter name (e.g., appid, X-API-Key)
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
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    
    # Response handling
    success_codes: List[int] = Field(default=[200], description="HTTP codes indicating success")
    parse_as: ParseType = Field(default=ParseType.JSON, description="Response parsing type")
    output_mapping: List[OutputMapping] = Field(default_factory=list, description="Output field extraction")
    error_mapping: List[ErrorMapping] = Field(default_factory=list, description="Error field extraction")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate tool name format."""
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        if not v.replace("_", "").isalnum():
            raise ValueError("Tool name must be alphanumeric with underscores only")
        return v.strip()

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate base URL format."""
        if not v or not v.strip():
            raise ValueError("Base URL cannot be empty")
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate endpoint path format."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        v = v.strip()
        if not v.startswith("/"):
            raise ValueError("Path must start with /")
        return v