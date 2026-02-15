"""
Built-in executor tools.

Provides functions to build Pydantic AI Tool objects for built-in executor tools.
"""

from typing import Any, Dict, List, Optional

from pydantic_ai import RunContext, Tool

from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.tools.deps import ToolsDeps

logger = get_logger("agentic_tools", level=appConfiguration.LoggerConfiguration.AgentLevel)


def build_builtin_tools() -> List[Tool]:
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
