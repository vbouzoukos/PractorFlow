"""
Agent tool registration utilities.

Provides functions to register default tools and executor-specific tools
with agent instances.
"""

from typing import Any, Dict, Optional

from pydantic_ai import Agent, RunContext

from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.agent.deps import AgentDeps

logger = get_logger(
    "agent_tools", level=appConfiguration.LoggerConfiguration.AgentLevel
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
        logger.debug("[AgentService] Registered search_knowledge tool")

    if "web_search" not in tool_registry:
        web_search_tool = DuckDuckGoSearchTool()
        tool_registry.register(web_search_tool)
        logger.debug("[AgentService] Registered web_search tool")

    if "web_fetch" not in tool_registry:
        web_fetch_tool = WebFetchTool()
        tool_registry.register(web_fetch_tool)
        logger.debug("[AgentService] Registered web_fetch tool")

    if "summarize_text" not in tool_registry:
        summarizer_tool = TextSummarizerTool()
        tool_registry.register(summarizer_tool)
        logger.debug("[AgentService] Registered summarize_text tool")

    if "json_transform" not in tool_registry:
        json_tool = JsonTransformTool()
        tool_registry.register(json_tool)
        logger.debug("[AgentService] Registered json_transform tool")

    if "calculator" not in tool_registry:
        calculator_tool = CalculatorTool()
        tool_registry.register(calculator_tool)
        logger.debug("[AgentService] Registered calculator tool")


def register_executor_tools(agent: Agent, deps: AgentDeps) -> None:
    """
    Register tools with the executor agent.

    Args:
        agent: Agent to register tools with.
        deps: Dependencies containing tool registry.
    """

    @agent.tool
    async def execute_tool(
        ctx: RunContext[AgentDeps],
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
        logger.debug(f"[AgentService] execute_tool called: {tool_name}")

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
            logger.error(f"[AgentService] Tool execution failed: {e}")
            return f"Tool execution failed: {e}"

    @agent.tool
    async def search_knowledge(
        ctx: RunContext[AgentDeps],
        query: str,
    ) -> str:
        """
        Search the knowledge base for relevant information from documents.

        Args:
            query: Search query text.

        Returns:
            Search results as formatted string.
        """
        logger.debug(f"[AgentService] search_knowledge called: {query}")

        # Guard: No documents in scope means no search
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

    @agent.tool
    async def search_web(
        ctx: RunContext[AgentDeps],
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
        logger.debug(f"[AgentService] search_web called: {query}")

        result = await ctx.deps.tool_registry.execute(
            "web_search", query=query, max_results=max_results
        )

        if result.success:
            return str(result.data) if result.data else ""
        else:
            return ""

    @agent.tool
    async def fetch_webpage(
        ctx: RunContext[AgentDeps],
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
        logger.debug(f"[AgentService] fetch_webpage called: {url}")

        result = await ctx.deps.tool_registry.execute(
            "web_fetch", url=url, extract_mode=extract_mode
        )

        if result.success:
            return str(result.data) if result.data else ""
        else:
            return ""

    @agent.tool
    async def summarize_text(
        ctx: RunContext[AgentDeps],
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
        logger.debug(f"[AgentService] summarize_text called: {len(text)} chars")

        result = await ctx.deps.tool_registry.execute(
            "summarize_text",
            text=text,
            num_sentences=num_sentences,
        )

        if result.success:
            return str(result.data) if result.data else "No summary generated."
        else:
            return ""

    @agent.tool
    async def transform_json(
        ctx: RunContext[AgentDeps],
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
        logger.debug(f"[AgentService] transform_json called: {operation}")

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

    @agent.tool
    async def calculate(
        ctx: RunContext[AgentDeps],
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
        logger.debug(f"[AgentService] calculate called: {expression or convert}")

        result = await ctx.deps.tool_registry.execute(
            "calculator",
            expression=expression,
            convert=convert,
        )

        if result.success:
            return str(result.data)
        else:
            return f"Calculation error: {result.error}"
