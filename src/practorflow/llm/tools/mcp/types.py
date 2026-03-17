"""
MCP type definitions.

Pydantic models for MCP server configuration, transport configs, and tool overrides.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TransportType(str, Enum):
    """Supported MCP transport types."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


class StdioConfig(BaseModel):
    """Configuration for stdio transport."""

    command: str = Field(..., description="Command to execute")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
    allowed_commands: Optional[List[str]] = Field(
        default=None, description="Optional command allowlist"
    )


class HttpConfig(BaseModel):
    """Configuration for Streamable HTTP transport."""

    url: str = Field(..., description="Server URL")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="HTTP headers"
    )
    timeout_seconds: int = Field(default=30, description="Request timeout in seconds")


class MCPServerConfig(BaseModel):
    """Complete MCP server configuration."""

    server_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    name: str = Field(..., description="Unique server name")

    transport: TransportType = Field(..., description="Transport type")

    stdio_config: Optional[StdioConfig] = Field(
        default=None, description="stdio transport configuration"
    )
    http_config: Optional[HttpConfig] = Field(
        default=None, description="Streamable HTTP transport configuration"
    )

    enabled: bool = Field(default=True, description="Whether the server is enabled")
    purpose: str = Field(default="", description="Server purpose")
    keywords: List[str] = Field(default_factory=list, description="Search keywords")
    category: str = Field(default="", description="Server category")
    tags: List[str] = Field(default_factory=list, description="Server tags")
    use_when: List[str] = Field(
        default_factory=list, description="When to use this server"
    )
    do_not_use_when: List[str] = Field(
        default_factory=list, description="When not to use this server"
    )