"""
MCP tool wrapper implementing AsyncBaseTool interface.

Wraps MCP server tools as AsyncBaseTool instances, enabling them to be
registered and used alongside built-in and API tools.
"""

import logging
from typing import Any, Dict, List, Optional

from practorflow.llm.tools.async_tool import AsyncBaseTool
from practorflow.llm.tools.base import ToolParameter, ToolResult
from practorflow.llm.tools.mcp.client import MCPClient
from practorflow.llm.tools.mcp.types import MCPToolConfig

logger = logging.getLogger(__name__)


class MCPTool(AsyncBaseTool):
    """
    AsyncBaseTool wrapper for MCP server tools.

    Each MCPTool instance represents one tool from an MCP server.
    Multiple MCPTool instances can share the same MCPClient connection.
    """

    def __init__(
        self,
        tool_name: str,
        server_name: str,
        server_description: str,
        input_schema: Dict[str, Any],
        client: MCPClient,
        override: Optional[MCPToolConfig] = None,
    ):
        """
        Initialize MCP tool wrapper.

        Args:
            tool_name: Name of the tool from MCP server
            server_name: Name of parent MCP server
            server_description: Server-provided description
            input_schema: JSON schema for tool parameters
            client: Shared MCPClient instance
            override: Optional admin overrides for metadata
        """
        self._name = tool_name
        self._server_name = server_name
        self._server_description = server_description
        self._input_schema = input_schema
        self._client = client
        self._override = override or MCPToolConfig()

        # Parse parameters from input schema
        self._parameters = self._parse_parameters(input_schema)

    def _parse_parameters(self, schema: Dict[str, Any]) -> List[ToolParameter]:
        """
        Convert MCP input schema to ToolParameter list.

        Args:
            schema: MCP tool input schema (JSON Schema format)

        Returns:
            List of ToolParameter instances
        """
        parameters = []
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        for param_name, param_def in properties.items():
            param = ToolParameter(
                name=param_name,
                type=param_def.get("type", "string"),
                description=param_def.get("description", ""),
                required=param_name in required_fields,
                default=param_def.get("default"),
                enum=param_def.get("enum"),
            )
            parameters.append(param)

        return parameters

    @property
    def name(self) -> str:
        """Tool name (not overridable)."""
        return self._name

    @property
    def description(self) -> str:
        """
        Tool description with enrichment.

        Uses server-provided description as base, appends admin overrides if present.
        """
        # Use override description if provided, otherwise use server description
        base_description = (
            self._override.description
            if self._override.description
            else self._server_description
        )

        # Build enriched description
        enrichments = []

        if self._override.purpose:
            enrichments.append(f"Purpose: {self._override.purpose}")

        if self._override.use_when:
            use_when_str = ", ".join(self._override.use_when)
            enrichments.append(f"Use when: {use_when_str}")

        if self._override.do_not_use_when:
            do_not_use_str = ", ".join(self._override.do_not_use_when)
            enrichments.append(f"Do not use when: {do_not_use_str}")

        if enrichments:
            return f"{base_description}\n{chr(10).join(enrichments)}"

        return base_description

    @property
    def parameters(self) -> List[ToolParameter]:
        """Tool parameters (derived from MCP schema, not overridable)."""
        return self._parameters

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the MCP tool via shared client.

        Args:
            **kwargs: Tool arguments

        Returns:
            ToolResult from MCP tool execution
        """
        try:
            logger.debug(
                f"Executing MCP tool '{self._name}' from server "
                f"'{self._server_name}' with args: {kwargs}"
            )

            # Delegate to MCPClient
            result = await self._client.call_tool(self._name, arguments=kwargs)

            return result

        except Exception as e:
            logger.error(f"Error executing MCP tool '{self._name}': {e}")
            return ToolResult(
                success=False,
                error=f"MCP tool execution failed: {str(e)}",
                metadata={
                    "server": self._server_name,
                    "tool": self._name,
                },
            )

    def get_schema(self) -> Dict[str, Any]:
        """
        Get tool schema with enriched description.

        Overrides base implementation to use enriched description.

        Returns:
            Dictionary representing the tool schema
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,  # Uses enriched description
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @property
    def server_name(self) -> str:
        """Name of parent MCP server."""
        return self._server_name

    @property
    def override(self) -> MCPToolConfig:
        """Admin override configuration."""
        return self._override

    @property
    def category(self) -> Optional[str]:
        """Tool category from admin override."""
        return self._override.category

    @property
    def tags(self) -> List[str]:
        """Tool tags from admin override."""
        return self._override.tags

    @property
    def keywords(self) -> List[str]:
        """Search keywords from admin override."""
        return self._override.keywords
