"""
MCP API request/response schemas.

Pydantic models for MCP server management endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from practorflow.llm.tools.mcp.types import (
    HttpConfig,
    MCPServerConfig,
    StdioConfig,
    MCPToolConfig,
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
        default=None, description="SSE / Streamable HTTP transport configuration"
    )
    tools: List[MCPToolConfig] = Field(
        default_factory=list,
        description="MCP tool configurations",
    )


class MCPServerUpdateRequest(BaseModel):
    """Request model for updating an MCP server (partial update)."""

    name: Optional[str] = Field(default=None, description="Unique server name")
    stdio_config: Optional[StdioConfig] = Field(
        default=None, description="stdio transport configuration"
    )
    http_config: Optional[HttpConfig] = Field(
        default=None, description="SSE / Streamable HTTP transport configuration"
    )
    tools: Optional[List[MCPToolConfig]] = Field(
        default=None,
        description="MCP tool configurations",
    )


# Response schemas


class MCPToolInfo(BaseModel):
    """Information about a single MCP tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    input_schema: Dict = Field(..., description="Tool input schema")


class MCPServerResponse(BaseModel):
    """Response model for MCP server details."""

    server_id: str = Field(..., description="Server ID")
    name: str = Field(..., description="Server name")
    transport: TransportType = Field(..., description="Transport type")
    stdio_config: Optional[StdioConfig] = Field(
        default=None, description="stdio transport configuration"
    )
    http_config: Optional[HttpConfig] = Field(
        default=None, description="SSE / Streamable HTTP transport configuration"
    )
    tools: List[MCPToolConfig] = Field(
        default_factory=list,
        description="MCP tool configurations",
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
            tools=config.tools,
            created_at=config.created_at,
            updated_at=config.updated_at,
        )


class MCPServerListResponse(BaseModel):
    """Response model for listing MCP servers."""

    servers: List[MCPServerResponse] = Field(..., description="List of MCP servers")
    count: int = Field(..., description="Total number of servers")


class MCPServerTestResponse(BaseModel):
    """Response model for testing MCP server connection."""

    server_id: str = Field(..., description="Server ID")
    connected: bool = Field(..., description="Connection status")
    tools: List[MCPToolInfo] = Field(
        default_factory=list, description="Available tools from server"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


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
