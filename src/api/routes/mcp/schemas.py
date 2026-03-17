"""
MCP API request/response schemas.

Pydantic models for MCP server management endpoints.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from practorflow.llm.tools.mcp.types import (
    HttpConfig,
    MCPServerConfig,
    StdioConfig,
    TransportType,
)


# Request schemas


class MCPServerCreateRequest(BaseModel):
    """Request model for creating an MCP server."""

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


class MCPServerUpdateRequest(BaseModel):
    """Request model for updating an MCP server (partial update)."""

    name: Optional[str] = Field(default=None, description="Unique server name")
    stdio_config: Optional[StdioConfig] = Field(
        default=None, description="stdio transport configuration"
    )
    http_config: Optional[HttpConfig] = Field(
        default=None, description="Streamable HTTP transport configuration"
    )
    enabled: Optional[bool] = Field(
        default=None, description="Whether the server is enabled"
    )
    purpose: Optional[str] = Field(default=None, description="Server purpose")
    keywords: Optional[List[str]] = Field(
        default=None, description="Search keywords"
    )
    category: Optional[str] = Field(default=None, description="Server category")
    tags: Optional[List[str]] = Field(default=None, description="Server tags")
    use_when: Optional[List[str]] = Field(
        default=None, description="When to use this server"
    )
    do_not_use_when: Optional[List[str]] = Field(
        default=None, description="When not to use this server"
    )


# Response schemas


class MCPServerResponse(BaseModel):
    """Response model for MCP server details."""

    server_id: str = Field(..., description="Server ID")
    name: str = Field(..., description="Server name")
    transport: TransportType = Field(..., description="Transport type")
    stdio_config: Optional[StdioConfig] = Field(
        default=None, description="stdio transport configuration"
    )
    http_config: Optional[HttpConfig] = Field(
        default=None, description="Streamable HTTP transport configuration"
    )
    enabled: bool = Field(..., description="Whether the server is enabled")
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
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    @classmethod
    def from_config(cls, config: MCPServerConfig) -> "MCPServerResponse":
        """Create response from MCPServerConfig."""
        return cls(
            server_id=config.server_id,
            name=config.name,
            transport=config.transport,
            stdio_config=config.stdio_config,
            http_config=config.http_config,
            enabled=config.enabled,
            purpose=config.purpose,
            keywords=config.keywords,
            category=config.category,
            tags=config.tags,
            use_when=config.use_when,
            do_not_use_when=config.do_not_use_when,
            created_at=config.created_at,
            updated_at=config.updated_at,
        )


class MCPServerListResponse(BaseModel):
    """Response model for listing MCP servers."""

    servers: List[MCPServerResponse] = Field(..., description="List of MCP servers")
    count: int = Field(..., description="Total number of servers")


class MCPServerReloadResponse(BaseModel):
    """Response model for reloading MCP server."""

    server_id: str = Field(..., description="Server ID")
    reloaded: bool = Field(..., description="Reload status")
    message: str = Field(..., description="Status message")


class MCPServerDeleteResponse(BaseModel):
    """Response model for deleting MCP server."""

    server_id: str = Field(..., description="Deleted server ID")
    deleted: bool = Field(..., description="Deletion status")
    message: str = Field(..., description="Status message")
