"""
MCP tool building utilities.

Provides functions to build Pydantic AI Tool objects from MCP tools.
"""

from typing import List, Optional, Set

from pydantic_ai import RunContext, Tool

from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.llm.tools.user_preferences import UserToolPreferences
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.tools.deps import ToolsDeps

logger = get_logger("agentic_tools", level=appConfiguration.LoggerConfiguration.AgentLevel)


def _create_mcp_tool_wrapper(tool_name: str):
    """
    Create an async wrapper function for an MCP tool.

    Args:
        tool_name: Name of the MCP tool to wrap.

    Returns:
        Async function that executes the tool via the registry.
    """

    async def mcp_tool_executor(ctx: RunContext[ToolsDeps], **kwargs) -> str:
        """Execute the MCP tool with provided arguments."""
        ctx.deps.tool_registry.set_document_scope(ctx.deps.document_scope)

        try:
            result = await ctx.deps.tool_registry.execute(tool_name, **kwargs)

            if result.success:
                return str(result.data) if result.data else "Tool executed successfully"
            else:
                return f"Tool error: {result.error}"
        except Exception as e:
            logger.error(f"MCP tool '{tool_name}' execution failed: {e}")
            return f"Tool execution failed: {e}"

    return mcp_tool_executor


def _build_mcp_tool_description(mcp_tool) -> str:
    """
    Build a rich description for an MCP tool including all usage guidelines.

    Combines description, purpose, use_when, do_not_use_when, category, tags,
    and keywords from the tool's override configuration into a comprehensive
    description for the LLM.

    Args:
        mcp_tool: MCPTool instance with override configuration.

    Returns:
        Rich description string for Pydantic AI Tool.
    """
    # Base description from MCPTool (already enriched)
    parts = [mcp_tool.description]

    override = mcp_tool.override

    # Add category if present
    if override.category:
        parts.append(f"\nCategory: {override.category}")

    # Add tags if present
    if override.tags:
        parts.append(f"\nTags: {', '.join(override.tags)}")

    # Add keywords if present
    if override.keywords:
        parts.append(f"\nKeywords: {', '.join(override.keywords)}")

    return "".join(parts)


def build_mcp_tools(
    tool_registry: ToolRegistry,
    user_preferences: Optional[UserToolPreferences] = None,
) -> List[Tool]:
    """
    Build Pydantic AI Tool objects for MCP tools.

    Filters MCP tools based on user preferences:
    1. Tool must be in user's enabled_tools list
    2. Tool's parent server must be in user's enabled_mcp_servers list

    Args:
        tool_registry: Registry containing loaded MCP tools.
        user_preferences: User's tool preferences for filtering.

    Returns:
        List of Tool objects for enabled MCP tools.
    """
    tools = []

    # If no preferences provided, return empty list (no MCP tools enabled by default)
    if user_preferences is None:
        logger.debug("No user preferences provided, skipping MCP tools")
        return tools

    # Build set of enabled MCP tool names for quick lookup
    enabled_mcp_tool_names: Set[str] = {
        entry.id
        for entry in user_preferences.enabled_tools
        if entry.type == "mcp"
    }

    # Build set of enabled MCP server IDs
    enabled_mcp_servers = set(user_preferences.enabled_mcp_servers)

    # Get all MCP tools from registry
    mcp_tools = tool_registry.get_mcp_tools()

    for tool_name, mcp_tool in mcp_tools.items():
        # Check if tool is in user's enabled list
        if tool_name not in enabled_mcp_tool_names:
            logger.debug(f"MCP tool '{tool_name}' not in user's enabled list, skipping")
            continue

        # Check if tool's parent server is enabled
        server_name = mcp_tool.server_name
        if server_name not in enabled_mcp_servers:
            logger.debug(
                f"MCP tool '{tool_name}' server '{server_name}' not enabled, skipping"
            )
            continue

        try:
            schema = mcp_tool.get_schema()
            func_schema = schema.get("function", {})
            description = _build_mcp_tool_description(mcp_tool)
            parameters = func_schema.get(
                "parameters",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            )

            wrapper_func = _create_mcp_tool_wrapper(tool_name)

            pydantic_tool = Tool.from_schema(
                function=wrapper_func,
                name=tool_name,
                description=description,
                json_schema=parameters,
                takes_ctx=True,
            )

            tools.append(pydantic_tool)
            logger.debug(f"Built MCP tool: {tool_name}")

        except Exception as e:
            logger.error(f"Failed to build MCP tool '{tool_name}': {e}")
            continue

    if tools:
        logger.info(f"Built {len(tools)} MCP tools for user")

    return tools
