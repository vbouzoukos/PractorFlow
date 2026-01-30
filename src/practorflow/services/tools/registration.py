"""
Tool registration utilities.

Provides functions to register default tools and executor-specific tools
with agent instances. Used by both AgentService and ChatService.
"""

from typing import Any, Dict, List, Optional

from pydantic_ai import Agent, RunContext, Tool

from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.llm.tools.api.tool import ApiTool
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.tools.deps import ToolsDeps

logger = get_logger(
    "tools", level=appConfiguration.LoggerConfiguration.AgentLevel
)


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

    Combines description, purpose, use_when, do_not_use_when, requires,
    and returns from the tool's configuration into a comprehensive
    description for the LLM.

    Args:
        tool: ApiTool instance with configuration.

    Returns:
        Rich description string for Pydantic AI Tool.
    """
    config = tool._config
    parts = [config.description]

    if config.purpose:
        parts.append(f"\nPurpose: {config.purpose}")

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


def _build_api_tools(tool_registry: ToolRegistry) -> List[Tool]:
    """
    Build Pydantic AI Tool objects for user-defined API tools.

    Args:
        tool_registry: Registry containing loaded API tools.

    Returns:
        List of Tool objects for API tools.
    """
    tools = []

    for tool in tool_registry.get_all_tools():
        if not hasattr(tool, 'user_id'):
            continue

        if not isinstance(tool, ApiTool):
            continue

        try:
            schema = tool.get_schema()
            func_schema = schema.get("function", {})
            tool_name = func_schema.get("name", tool.name)
            description = _build_tool_description(tool)
            parameters = func_schema.get("parameters", {
                "type": "object",
                "properties": {},
                "required": [],
            })

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


def _build_builtin_tools() -> List[Tool]:
    """
    Build Pydantic AI Tool objects for built-in executor tools.

    Returns:
        List of Tool objects for built-in tools.
    """
    tools = []

    # execute_tool
    async def execute_tool(
        ctx: RunContext[ToolsDeps],
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Execute a tool by name with provided arguments.

        Args:
            tool_name: Name of the tool to execute.
            tool_args: Arguments to pass to the tool.

        Returns:
            Tool result as string.
        """
        logger.debug(f"execute_tool called: {tool_name}")

        if tool_name not in ctx.deps.tool_registry:
            return f"Error: Tool '{tool_name}' not found"

        ctx.deps.tool_registry.set_document_scope(ctx.deps.document_scope)

        try:
            args = tool_args or {}
            result = await ctx.deps.tool_registry.execute(tool_name, **args)

            if result.success:
                return str(result.data) if result.data else "Tool executed successfully"
            else:
                return f"Tool error: {result.error}"
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Tool execution failed: {e}"

    tools.append(Tool(execute_tool, takes_ctx=True))

    # search_knowledge
    async def search_knowledge(
        ctx: RunContext[ToolsDeps],
        query: str,
    ) -> str:
        """
        Search the knowledge base for relevant information from documents.

        Args:
            query: Search query text.

        Returns:
            Search results as formatted string.
        """
        logger.debug(f"search_knowledge called: {query}")

        if ctx.deps.document_scope is None:
            return ""

        results = ctx.deps.knowledge_store.search_scoped(
            query=query,
            top_k=5,
            document_ids=ctx.deps.document_scope,
        )

        if not results:
            return ""

        parts = [f"Found {len(results)} result(s):"]
        for idx, result in enumerate(results, 1):
            text = result.get("text", "")[:500]
            filename = result.get("filename", "unknown")
            parts.append(f"\n[{idx}] From {filename}:\n{text}")

        return "\n".join(parts)

    tools.append(Tool(search_knowledge, takes_ctx=True))

    # search_web
    async def search_web(
        ctx: RunContext[ToolsDeps],
        query: str,
        max_results: int = 5,
    ) -> str:
        """
        Search the web for current information.

        Args:
            query: Search query text.
            max_results: Maximum number of results to return.

        Returns:
            Search results as formatted string.
        """
        logger.debug(f"search_web called: {query}")

        result = await ctx.deps.tool_registry.execute(
            "web_search", query=query, max_results=max_results
        )

        if result.success:
            return str(result.data) if result.data else ""
        else:
            return ""

    tools.append(Tool(search_web, takes_ctx=True))

    # fetch_webpage
    async def fetch_webpage(
        ctx: RunContext[ToolsDeps],
        url: str,
        extract_mode: str = "text",
    ) -> str:
        """
        Fetch and extract content from a web page URL.

        Args:
            url: URL of the web page to fetch.
            extract_mode: Extraction mode - text, raw, or metadata.

        Returns:
            Page content as string.
        """
        logger.debug(f"fetch_webpage called: {url}")

        result = await ctx.deps.tool_registry.execute(
            "web_fetch", url=url, extract_mode=extract_mode
        )

        if result.success:
            return str(result.data) if result.data else ""
        else:
            return ""

    tools.append(Tool(fetch_webpage, takes_ctx=True))

    # summarize_text
    async def summarize_text(
        ctx: RunContext[ToolsDeps],
        text: str,
        num_sentences: int = 5,
    ) -> str:
        """
        Summarize long text by extracting important sentences.

        Args:
            text: Text content to summarize.
            num_sentences: Number of sentences to extract.

        Returns:
            Summarized text.
        """
        logger.debug(f"summarize_text called: {len(text)} chars")

        result = await ctx.deps.tool_registry.execute(
            "summarize_text",
            text=text,
            num_sentences=num_sentences,
        )

        if result.success:
            return str(result.data) if result.data else "No summary generated."
        else:
            return ""

    tools.append(Tool(summarize_text, takes_ctx=True))

    # transform_json
    async def transform_json(
        ctx: RunContext[ToolsDeps],
        json_data: str,
        operation: str,
        path: Optional[str] = None,
    ) -> str:
        """
        Parse, query, or transform JSON data.

        Args:
            json_data: JSON string to process.
            operation: Operation - parse, query, extract, flatten, keys, values.
            path: JSONPath for query/extract operations.

        Returns:
            Transformed data as string.
        """
        logger.debug(f"transform_json called: {operation}")

        result = await ctx.deps.tool_registry.execute(
            "json_transform",
            json_data=json_data,
            operation=operation,
            path=path,
        )

        if result.success:
            data = result.data
            if isinstance(data, (dict, list)):
                import json

                return json.dumps(data, indent=2)
            return str(data) if data else "No result."
        else:
            return f"JSON transform error: {result.error}"

    tools.append(Tool(transform_json, takes_ctx=True))

    # calculate
    async def calculate(
        ctx: RunContext[ToolsDeps],
        expression: Optional[str] = None,
        convert: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Perform mathematical calculations or unit conversions.

        Args:
            expression: Math expression to evaluate (e.g., 'sqrt(16)', '2**10').
            convert: Unit conversion dict with value, from, to keys.

        Returns:
            Calculation result as string.
        """
        logger.debug(f"calculate called: {expression or convert}")

        result = await ctx.deps.tool_registry.execute(
            "calculator",
            expression=expression,
            convert=convert,
        )

        if result.success:
            return str(result.data)
        else:
            return f"Calculation error: {result.error}"

    tools.append(Tool(calculate, takes_ctx=True))

    return tools


def load_api_tools_for_user(tool_registry: ToolRegistry, user_id: str) -> List[Tool]:
    """
    Load user-defined API tools and build all tools for Agent constructor.

    This is the single entry point for tool preparation. It:
    1. Loads API tools from the factory into the registry
    2. Builds built-in executor tools as Tool objects
    3. Builds API tools as Tool objects using Tool.from_schema()

    The returned list can be passed directly to Agent(tools=...).

    Args:
        tool_registry: Registry to load tools into.
        user_id: User ID to load API tools for.

    Returns:
        List of Tool objects for Agent constructor.
    """
    tool_registry.load_api_tools_for_user(user_id)

    tools = _build_builtin_tools()

    api_tools = _build_api_tools(tool_registry)
    tools.extend(api_tools)

    logger.info(f"Prepared {len(tools)} tools for agent ({len(api_tools)} API tools)")

    return tools


def build_tools_for_registry(tool_registry: ToolRegistry) -> List[Tool]:
    """
    Build all tools for Agent constructor from existing registry.

    Use this when API tools are already loaded into the registry
    and you just need to build Tool objects.

    Args:
        tool_registry: Registry containing tools.

    Returns:
        List of Tool objects for Agent constructor.
    """
    tools = _build_builtin_tools()

    api_tools = _build_api_tools(tool_registry)
    tools.extend(api_tools)

    return tools
