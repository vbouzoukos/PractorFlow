"""
Tool preferences API request/response schemas.

Pydantic models for user tool preferences endpoints.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# Request/Response models


class ToolPreferenceUpdateRequest(BaseModel):
    """Request to update a single tool or MCP server preference."""

    type: str = Field(..., description="Tool type: 'builtin', 'api', or 'mcp'")
    id: str = Field(..., description="Tool identifier")
    enabled: bool = Field(..., description="Whether to enable or disable")
    server_id: Optional[str] = Field(
        default=None,
        description="MCP server ID — required when type is 'mcp' to also toggle the server",
    )


class ToolPreferenceUpdateResponse(BaseModel):
    """Response after updating a single tool preference."""

    type: str = Field(..., description="Tool type")
    id: str = Field(..., description="Tool identifier")
    enabled: bool = Field(..., description="Resulting enabled state")


class ToolInfoBase(BaseModel):
    """Base information for a tool."""

    id: str = Field(..., description="Tool identifier")
    type: str = Field(..., description="Tool type: 'builtin', 'api', or 'mcp'")
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    enabled: bool = Field(..., description="Whether the tool is enabled for the user")


class BuiltinToolInfo(ToolInfoBase):
    """Information about a built-in tool."""

    type: str = Field(default="builtin", description="Always 'builtin'")


class ApiToolInfo(ToolInfoBase):
    """Information about an API tool."""

    type: str = Field(default="api", description="Always 'api'")
    system: bool = Field(..., description="Whether the tool is a system tool")


class MCPToolInfo(ToolInfoBase):
    """Information about an MCP tool."""

    type: str = Field(default="mcp", description="Always 'mcp'")
    server_id: str = Field(..., description="Parent MCP server ID")
    server_name: str = Field(..., description="Parent MCP server name")


class ToolsListResponse(BaseModel):
    """Response model for listing all available tools."""

    builtin_tools: List[BuiltinToolInfo] = Field(
        default_factory=list, description="Built-in tools"
    )
    api_tools: List[ApiToolInfo] = Field(
        default_factory=list, description="API tools"
    )
    mcp_tools: List[MCPToolInfo] = Field(
        default_factory=list, description="MCP tools"
    )
    count: int = Field(..., description="Total number of tools")
