"""
Tool registration utilities.

Provides functions to register default tools and executor-specific tools
with agent instances. Used by both AgentService and ChatService.
"""

from typing import List, Optional

from pydantic_ai import Tool

from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.llm.tools.user_preferences import UserToolPreferences
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.tools.builders import (
    build_api_tools,
    build_builtin_tools,
)

logger = get_logger("agentic_tools", level=appConfiguration.LoggerConfiguration.AgentLevel)


def register_default_tools(
    tool_registry: ToolRegistry,
    knowledge_store: KnowledgeStore,
) -> None:
    """
    Register default tools if not already present.

    Args:
        tool_registry: Registry to register tools with.
        knowledge_store: Knowledge store for search tool.
    """
    from practorflow.llm.tools.knowledge_search import KnowledgeSearchTool
    from practorflow.llm.tools.base_web_search import DuckDuckGoSearchTool
    from practorflow.llm.tools.web_fetch import WebFetchTool
    from practorflow.llm.tools.text_summarizer import TextSummarizerTool
    from practorflow.llm.tools.json_transform import JsonTransformTool
    from practorflow.llm.tools.calculator import CalculatorTool

    if "search_knowledge" not in tool_registry:
        knowledge_tool = KnowledgeSearchTool(
            knowledge_store=knowledge_store,
            default_top_k=5,
        )
        tool_registry.register(knowledge_tool)
        logger.debug("Registered search_knowledge tool")

    if "web_search" not in tool_registry:
        web_search_tool = DuckDuckGoSearchTool()
        tool_registry.register(web_search_tool)
        logger.debug("Registered web_search tool")

    if "web_fetch" not in tool_registry:
        web_fetch_tool = WebFetchTool()
        tool_registry.register(web_fetch_tool)
        logger.debug("Registered web_fetch tool")

    if "summarize_text" not in tool_registry:
        summarizer_tool = TextSummarizerTool()
        tool_registry.register(summarizer_tool)
        logger.debug("Registered summarize_text tool")

    if "json_transform" not in tool_registry:
        json_tool = JsonTransformTool()
        tool_registry.register(json_tool)
        logger.debug("Registered json_transform tool")

    if "calculator" not in tool_registry:
        calculator_tool = CalculatorTool()
        tool_registry.register(calculator_tool)
        logger.debug("Registered calculator tool")


def load_api_tools_for_user(
    tool_registry: ToolRegistry,
    user_id: str,
    user_preferences: Optional[UserToolPreferences] = None,
) -> List[Tool]:
    """
    Load user-defined API tools and build all tools for Agent constructor.

    This is the single entry point for tool preparation. It:
    1. Loads API tools from the factory into the registry
    2. Builds built-in executor tools as Tool objects
    3. Builds API tools as Tool objects using Tool.from_schema()
    4. Loads MCP toolsets into registry based on user preferences

    The returned list can be passed directly to Agent(tools=...).
    MCP toolsets are separate, retrieved via tool_registry.get_mcp_toolsets().

    Args:
        tool_registry: Registry to load tools into.
        user_id: User ID to load API tools for.
        user_preferences: User's tool preferences for filtering MCP tools.

    Returns:
        List of Tool objects for Agent constructor.
    """
    tool_registry.load_api_tools_for_user(user_id)

    tools = build_builtin_tools()

    api_tools = build_api_tools(tool_registry)
    tools.extend(api_tools)

    tool_registry.load_mcp_toolsets(user_preferences)

    logger.info(
        f"Prepared {len(tools)} tools for agent "
        f"({len(api_tools)} API tools)"
    )

    return tools


def build_tools_for_registry(
    tool_registry: ToolRegistry,
    user_preferences: Optional[UserToolPreferences] = None,
) -> List[Tool]:
    """
    Build all tools for Agent constructor from existing registry.

    Use this when API tools are already loaded into the registry
    and you just need to build Tool objects.

    Args:
        tool_registry: Registry containing tools.
        user_preferences: User's tool preferences for filtering MCP tools.

    Returns:
        List of Tool objects for Agent constructor.
    """
    tools = build_builtin_tools()

    api_tools = build_api_tools(tool_registry)
    tools.extend(api_tools)

    if user_preferences is not None:
        tool_registry.load_mcp_toolsets(user_preferences)

    return tools
