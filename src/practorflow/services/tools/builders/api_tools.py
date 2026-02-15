"""
API tool building utilities.

Provides functions to build Pydantic AI Tool objects from user-defined API tools.
"""

from typing import List

from pydantic_ai import RunContext, Tool

from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.llm.tools.api.tool import ApiTool
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.tools.deps import ToolsDeps

logger = get_logger("agentic_tools", level=appConfiguration.LoggerConfiguration.AgentLevel)


def _create_api_tool_wrapper(tool_name: str):
    """
    Create an async wrapper function for an API tool.

    Args:
        tool_name: Name of the API tool to wrap.

    Returns:
        Async function that executes the tool via the registry.
    """

    async def api_tool_executor(ctx: RunContext[ToolsDeps], **kwargs) -> str:
        """Execute the API tool with provided arguments."""
        ctx.deps.tool_registry.set_document_scope(ctx.deps.document_scope)

        try:
            result = await ctx.deps.tool_registry.execute(tool_name, **kwargs)

            if result.success:
                return str(result.data) if result.data else "Tool executed successfully"
            else:
                return f"Tool error: {result.error}"
        except Exception as e:
            logger.error(f"API tool '{tool_name}' execution failed: {e}")
            return f"Tool execution failed: {e}"

    return api_tool_executor


def _build_tool_description(tool: ApiTool) -> str:
    """
    Build a rich description for an API tool including all usage guidelines.

    Combines description, purpose, keywords, category, tags,
    use_when, do_not_use_when, requires, and returns
    from the tool's configuration into a comprehensive description for the LLM.

    Args:
        tool: ApiTool instance with configuration.

    Returns:
        Rich description string for Pydantic AI Tool.
    """
    config = tool._config
    parts = [config.description]

    if config.purpose:
        parts.append(f"\nPurpose: {config.purpose}")

    if config.keywords:
        parts.append(f"\nKeywords: {', '.join(config.keywords)}")

    if config.category:
        parts.append(f"\nCategory: {config.category}")

    if config.tags:
        parts.append(f"\nTags: {', '.join(config.tags)}")

    if config.use_when:
        use_cases = "\n".join(f"  - {case}" for case in config.use_when)
        parts.append(f"\nUse when:\n{use_cases}")

    if config.do_not_use_when:
        anti_patterns = "\n".join(f"  - {case}" for case in config.do_not_use_when)
        parts.append(f"\nDo NOT use when:\n{anti_patterns}")

    if config.requires:
        requirements = "\n".join(f"  - {req}" for req in config.requires)
        parts.append(f"\nRequires:\n{requirements}")

    if config.returns:
        parts.append(f"\nReturns: {config.returns}")

    return "".join(parts)


def build_api_tools(tool_registry: ToolRegistry) -> List[Tool]:
    """
    Build Pydantic AI Tool objects for user-defined API tools.

    Args:
        tool_registry: Registry containing loaded API tools.

    Returns:
        List of Tool objects for API tools.
    """
    tools = []

    for tool in tool_registry.get_all_tools():
        if not hasattr(tool, "user_id"):
            continue

        if not isinstance(tool, ApiTool):
            continue

        try:
            schema = tool.get_schema()
            func_schema = schema.get("function", {})
            tool_name = func_schema.get("name", tool.name)
            description = _build_tool_description(tool)
            parameters = func_schema.get(
                "parameters",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            )

            wrapper_func = _create_api_tool_wrapper(tool_name)

            pydantic_tool = Tool.from_schema(
                function=wrapper_func,
                name=tool_name,
                description=description,
                json_schema=parameters,
                takes_ctx=True,
            )

            tools.append(pydantic_tool)
            logger.debug(f"Built API tool: {tool_name}")

        except Exception as e:
            logger.error(f"Failed to build API tool '{tool.name}': {e}")
            continue

    if tools:
        logger.info(f"Built {len(tools)} API tools")

    return tools
