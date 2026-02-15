"""
MCP client implementation for transport management and tool execution.

Handles connection lifecycle, tool discovery, validation, and execution
for both stdio and HTTP-based MCP servers.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.types import Tool as MCPToolSchema

from practorflow.llm.tools.base import ToolResult
from practorflow.llm.tools.mcp.types import (
    HttpConfig,
    MCPServerConfig,
    StdioConfig,
    TransportType,
)

logger = logging.getLogger(__name__)


class MCPClient:
    """
    MCP client that manages transport lifecycle and tool execution.

    Each MCPClient instance represents a connection to one MCP server.
    Multiple MCPTool instances can share the same MCPClient connection.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client.

        Args:
            config: Server configuration including transport type and settings
        """
        self.config = config
        self._session: Optional[ClientSession] = None
        self._transport_context = None
        self._available_tools: Dict[str, MCPToolSchema] = {}
        self._connected = False

    async def connect(self) -> None:
        """
        Establish connection to MCP server.

        Creates appropriate transport based on config and initializes session.

        Raises:
            ValueError: If transport configuration is invalid
            RuntimeError: If connection fails
        """
        if self._connected:
            logger.warning(f"Client for {self.config.name} already connected")
            return

        try:
            if self.config.transport == TransportType.STDIO:
                await self._connect_stdio()
            elif self.config.transport in (
                TransportType.SSE,
                TransportType.STREAMABLE_HTTP,
            ):
                await self._connect_http()
            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")

            # List available tools after connection
            await self._refresh_tool_list()
            self._connected = True
            logger.info(
                f"Connected to MCP server '{self.config.name}' "
                f"with {len(self._available_tools)} tools"
            )

        except Exception as e:
            logger.error(f"Failed to connect to {self.config.name}: {e}")
            raise RuntimeError(f"Connection failed: {e}") from e

    async def _connect_stdio(self) -> None:
        """Establish stdio transport connection."""
        if not self.config.stdio_config:
            raise ValueError("stdio_config required for stdio transport")

        stdio_cfg: StdioConfig = self.config.stdio_config

        # Validate command against allowlist if configured
        if stdio_cfg.allowed_commands:
            if stdio_cfg.command not in stdio_cfg.allowed_commands:
                raise ValueError(
                    f"Command '{stdio_cfg.command}' not in allowed list: "
                    f"{stdio_cfg.allowed_commands}"
                )

        # Create server parameters
        server_params = StdioServerParameters(
            command=stdio_cfg.command,
            args=stdio_cfg.args,
            env=stdio_cfg.env if stdio_cfg.env else None,
        )

        # Establish stdio connection
        stdio_transport = stdio_client(server_params)
        self._transport_context = stdio_transport

        # Enter context and initialize session
        read, write = await stdio_transport.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()

    async def _connect_http(self) -> None:
        """Establish SSE/HTTP transport connection."""
        if not self.config.http_config:
            raise ValueError("http_config required for SSE/HTTP transport")

        http_cfg: HttpConfig = self.config.http_config

        # Create SSE client connection
        sse_transport = sse_client(
            url=http_cfg.url,
            headers=http_cfg.headers if http_cfg.headers else None,
            timeout=http_cfg.timeout_seconds,
        )
        self._transport_context = sse_transport

        # Enter context and initialize session
        read, write = await sse_transport.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()

    async def _refresh_tool_list(self) -> None:
        """Fetch and cache list of tools from server."""
        if not self._session:
            raise RuntimeError("Session not initialized")

        try:
            result = await self._session.list_tools()
            self._available_tools = {tool.name: tool for tool in result.tools}
            logger.debug(
                f"Refreshed tool list for {self.config.name}: "
                f"{list(self._available_tools.keys())}"
            )
        except Exception as e:
            logger.error(f"Failed to list tools from {self.config.name}: {e}")
            raise

    async def disconnect(self) -> None:
        """
        Close connection and clean up resources.

        Safe to call multiple times.
        """
        if not self._connected:
            return

        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
                self._session = None

            if self._transport_context:
                await self._transport_context.__aexit__(None, None, None)
                self._transport_context = None

            self._connected = False
            self._available_tools.clear()
            logger.info(f"Disconnected from MCP server '{self.config.name}'")

        except Exception as e:
            logger.error(f"Error during disconnect from {self.config.name}: {e}")

    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        List all tools exposed by the server.

        Used by admin to see available tools before creating them.

        Returns:
            List of tool metadata dicts with name, description, and schema

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected or not self._session:
            raise RuntimeError(f"Not connected to {self.config.name}")

        # Refresh tool list to get latest
        await self._refresh_tool_list()

        tools = []
        for tool_name, tool_schema in self._available_tools.items():
            tools.append(
                {
                    "name": tool_name,
                    "description": tool_schema.description or "",
                    "inputSchema": tool_schema.inputSchema,
                }
            )

        return tools

    async def validate_tool(self, tool_name: str) -> bool:
        """
        Verify a specific tool exists and is valid.

        Used during admin tool creation to reject invalid tools.

        Args:
            tool_name: Name of tool to validate

        Returns:
            True if tool exists and is valid, False otherwise

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected:
            raise RuntimeError(f"Not connected to {self.config.name}")

        # Refresh to ensure we have latest tool list
        await self._refresh_tool_list()

        if tool_name not in self._available_tools:
            logger.warning(
                f"Tool '{tool_name}' not found on server {self.config.name}"
            )
            return False

        # Validate schema is present
        tool_schema = self._available_tools[tool_name]
        if not tool_schema.inputSchema:
            logger.warning(
                f"Tool '{tool_name}' on {self.config.name} has no input schema"
            )
            return False

        return True

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool call via MCP protocol.

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments as dict

        Returns:
            ToolResult with execution outcome

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected or not self._session:
            return ToolResult(
                success=False,
                error=f"Not connected to server {self.config.name}",
            )

        # Validate tool exists
        if tool_name not in self._available_tools:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found on server {self.config.name}",
            )

        try:
            logger.debug(
                f"Calling tool '{tool_name}' on {self.config.name} "
                f"with args: {arguments}"
            )

            # Execute tool call via MCP session
            result = await self._session.call_tool(tool_name, arguments=arguments)

            # Extract content from MCP response
            content_parts = []
            if hasattr(result, "content") and result.content:
                for item in result.content:
                    if hasattr(item, "text"):
                        content_parts.append(item.text)

            output = "\n".join(content_parts) if content_parts else None

            # Check if call was successful
            if hasattr(result, "isError") and result.isError:
                return ToolResult(
                    success=False,
                    error=output or "Tool execution failed",
                    metadata={"server": self.config.name, "tool": tool_name},
                )

            return ToolResult(
                success=True,
                data=output,
                metadata={"server": self.config.name, "tool": tool_name},
            )

        except Exception as e:
            logger.error(
                f"Error calling tool '{tool_name}' on {self.config.name}: {e}"
            )
            return ToolResult(
                success=False,
                error=f"Tool execution error: {str(e)}",
                metadata={"server": self.config.name, "tool": tool_name},
            )

    async def health_check(self) -> bool:
        """
        Verify server connectivity.

        Returns:
            True if server is reachable and responsive, False otherwise
        """
        if not self._connected or not self._session:
            return False

        try:
            # Attempt to list tools as a health check
            await self._refresh_tool_list()
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.config.name}: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected."""
        return self._connected

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the input schema for a specific tool.

        Args:
            tool_name: Name of tool

        Returns:
            Input schema dict or None if tool not found
        """
        if tool_name not in self._available_tools:
            return None

        tool_schema = self._available_tools[tool_name]
        return tool_schema.inputSchema

