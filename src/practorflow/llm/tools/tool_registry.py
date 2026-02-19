"""
Tool registry for managing available tools.

Provides centralized tool registration, lookup, and execution.
Supports dynamic loading of user-defined API tools via ApiToolFactory.
"""

from typing import Dict, List, Optional, Any
from practorflow.llm.tools.async_tool import AsyncBaseTool
from practorflow.llm.tools.base import ToolResult
from practorflow.llm.tools.mcp.store import MCPServerStore
from practorflow.llm.tools.mcp.types import TransportType
from practorflow.llm.tools.user_preferences import UserToolPreferences
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class ToolRegistry:
    """
    Registry for managing LLM tools.

    Handles tool registration, lookup, and execution with
    support for document scope management and dynamic API tool loading.
    """

    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: Dict[str, AsyncBaseTool] = {}
        self._document_scope: Optional[set] = None
        self._last_result: Optional[ToolResult] = None
        self._loaded_user_id: Optional[str] = None
        self._mcp_server_store: Optional[MCPServerStore] = None
        self._mcp_toolsets: List[Any] = []

    def load_api_tools_for_user(self, user_id: str) -> int:
        """
        Load user-defined API tools from the factory.

        Clears previously loaded API tools for different user and loads
        new tools for the specified user. Skips expired tools.

        Args:
            user_id: User ID to load tools for.

        Returns:
            Number of tools loaded.
        """
        from practorflow.llm.tools.api.factory import is_factory_initialized, get_factory

        if not is_factory_initialized():
            logger.debug("[ToolRegistry] API tool factory not initialized, skipping")
            return 0

        # Clear previously loaded user tools if switching users
        if self._loaded_user_id and self._loaded_user_id != user_id:
            self._unregister_api_tools()

        factory = get_factory()
        tools = factory.create_tools_for_user(user_id)

        registered_count = 0
        for tool in tools:
            if tool.name in self._tools:
                logger.debug(f"[ToolRegistry] Tool '{tool.name}' already registered, skipping")
                continue

            self._tools[tool.name] = tool
            registered_count += 1
            logger.info(f"[ToolRegistry] Registered API tool: {tool.name}")

        self._loaded_user_id = user_id
        logger.info(f"[ToolRegistry] Loaded {registered_count} API tools for user '{user_id}'")

        return registered_count

    def _unregister_api_tools(self) -> int:
        """
        Unregister all API tools (tools with user_id attribute).

        Returns:
            Number of tools unregistered.
        """
        to_remove = [
            name for name, tool in self._tools.items()
            if hasattr(tool, 'user_id')
        ]

        for name in to_remove:
            del self._tools[name]
            logger.debug(f"[ToolRegistry] Unregistered API tool: {name}")

        self._loaded_user_id = None

        if to_remove:
            logger.info(f"[ToolRegistry] Unregistered {len(to_remove)} API tools")

        return len(to_remove)

    def register(self, tool: AsyncBaseTool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool
        logger.info(f"[ToolRegistry] Registered tool: {tool.name}")

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"[ToolRegistry] Unregistered tool: {tool_name}")
            return True
        return False

    def get(self, tool_name: str) -> Optional[AsyncBaseTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Tool name to lookup

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def get_all_tools(self) -> List[AsyncBaseTool]:
        """Get all registered tool instances."""
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Get schemas for all registered tools.

        Returns:
            List of tool schemas in function calling format
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def set_document_scope(self, document_ids: Optional[set]) -> None:
        """
        Set document scope for knowledge-aware tools.

        Args:
            document_ids: Set of document IDs to scope searches to,
                         or None to search all documents
        """
        if document_ids is not None:
            self._document_scope = set(document_ids)
            logger.debug(
                f"[ToolRegistry] Document scope set: {len(self._document_scope)} documents"
            )
        else:
            self._document_scope = None
            logger.debug("[ToolRegistry] Document scope cleared (all documents)")

    def get_document_scope(self) -> Optional[set]:
        """Get current document scope."""
        return self._document_scope

    def clear_document_scope(self) -> None:
        """Clear document scope to search all documents."""
        self._document_scope = None
        logger.info("[ToolRegistry] Document scope cleared")

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Parameters to pass to tool

        Returns:
            ToolResult from execution
        """
        # Check built-in and API tools
        tool = self._tools.get(tool_name)

        if not tool:
            result = ToolResult(success=False, error=f"Tool not found: {tool_name}")
            self._last_result = result
            return result

        # Inject document scope for knowledge-aware tools
        if hasattr(tool, "set_document_scope"):
            tool.set_document_scope(self._document_scope)

        result = await tool(**kwargs)
        self._last_result = result

        return result

    def get_last_result(self) -> Optional[ToolResult]:
        """Get the result from the last tool execution."""
        return self._last_result

    def clear_last_result(self) -> None:
        """Clear the last tool result."""
        self._last_result = None

    def has_pending_context(self) -> bool:
        """Check if there's a successful tool result to use as context."""
        return (
            self._last_result is not None
            and self._last_result.success
            and self._last_result.data is not None
        )

    def consume_context(self) -> Optional[str]:
        """
        Get and clear pending context from last tool execution.

        Returns:
            Context string if available, None otherwise
        """
        if not self.has_pending_context():
            return None

        context = self._last_result.to_context_string()
        self._last_result = None
        return context

    def set_mcp_server_store(self, store: MCPServerStore) -> None:
        """
        Store the MCP server store reference for later use.

        Args:
            store: MCPServerStore instance.
        """
        self._mcp_server_store = store

    def load_mcp_toolsets(
        self, user_preferences: Optional[UserToolPreferences] = None
    ) -> int:
        """
        Load native Pydantic AI MCP toolsets from configured servers.

        Reads all MCPServerConfig from the store and creates native
        MCPServer* instances based on transport type. Applies user
        preference filtering and tool description enrichment via
        .filtered() and .prepared() when applicable.

        Args:
            user_preferences: Optional user preferences for filtering servers
                and tools.

        Returns:
            Number of toolsets created.
        """
        from pydantic_ai.mcp import MCPServerStdio, MCPServerSSE, MCPServerStreamableHTTP

        # Clear any existing toolsets
        self.unload_mcp_toolsets()

        if self._mcp_server_store is None:
            logger.debug("[ToolRegistry] No MCP server store set, skipping MCP toolsets")
            return 0

        server_configs = self._mcp_server_store.list()
        if not server_configs:
            logger.debug("[ToolRegistry] No MCP servers configured")
            return 0

        # Build preference lookup sets
        enabled_server_names = None
        enabled_mcp_tool_ids = None
        if user_preferences is not None:
            enabled_server_names = set(user_preferences.enabled_mcp_servers)
            enabled_mcp_tool_ids = {
                entry.id
                for entry in user_preferences.enabled_tools
                if entry.type == "mcp"
            }

        for config in server_configs:
            # Filter by enabled servers when preferences provided
            if enabled_server_names is not None:
                if config.name not in enabled_server_names:
                    logger.debug(
                        f"[ToolRegistry] MCP server '{config.name}' not in user's "
                        f"enabled list, skipping"
                    )
                    continue

            try:
                # Create native Pydantic AI MCP server instance
                if config.transport == TransportType.STDIO:
                    if not config.stdio_config:
                        logger.error(
                            f"[ToolRegistry] stdio_config missing for server '{config.name}'"
                        )
                        continue
                    server = MCPServerStdio(
                        command=config.stdio_config.command,
                        args=config.stdio_config.args,
                        env=config.stdio_config.env if config.stdio_config.env else None,
                    )
                elif config.transport == TransportType.SSE:
                    if not config.http_config:
                        logger.error(
                            f"[ToolRegistry] http_config missing for server '{config.name}'"
                        )
                        continue
                    server = MCPServerSSE(
                        url=config.http_config.url,
                        headers=config.http_config.headers if config.http_config.headers else None,
                        timeout=config.http_config.timeout_seconds,
                    )
                elif config.transport == TransportType.STREAMABLE_HTTP:
                    if not config.http_config:
                        logger.error(
                            f"[ToolRegistry] http_config missing for server '{config.name}'"
                        )
                        continue
                    server = MCPServerStreamableHTTP(
                        url=config.http_config.url,
                        headers=config.http_config.headers if config.http_config.headers else None,
                        timeout=config.http_config.timeout_seconds,
                    )
                else:
                    logger.error(
                        f"[ToolRegistry] Unsupported transport '{config.transport}' "
                        f"for server '{config.name}'"
                    )
                    continue

                # Apply tool filtering via .filtered() when preferences provided
                if enabled_mcp_tool_ids is not None:
                    server = server.filtered(
                        allowed_tools=list(enabled_mcp_tool_ids)
                    )

                # Apply description enrichment via .prepared() for tools with overrides
                if config.tools:
                    async def _prepare_tools(ctx, tools, tool_configs=config.tools):
                        for tool in tools:
                            for tool_cfg in tool_configs:
                                if tool_cfg.name != tool.name:
                                    continue
                                # Build enriched description
                                base_description = (
                                    tool_cfg.description
                                    if tool_cfg.description
                                    else tool.description
                                )
                                enrichments = []
                                if tool_cfg.purpose:
                                    enrichments.append(f"Purpose: {tool_cfg.purpose}")
                                if tool_cfg.use_when:
                                    use_when_str = ", ".join(tool_cfg.use_when)
                                    enrichments.append(f"Use when: {use_when_str}")
                                if tool_cfg.do_not_use_when:
                                    do_not_use_str = ", ".join(tool_cfg.do_not_use_when)
                                    enrichments.append(f"Do not use when: {do_not_use_str}")
                                if tool_cfg.category:
                                    enrichments.append(f"Category: {tool_cfg.category}")
                                if tool_cfg.tags:
                                    enrichments.append(f"Tags: {', '.join(tool_cfg.tags)}")
                                if tool_cfg.keywords:
                                    enrichments.append(f"Keywords: {', '.join(tool_cfg.keywords)}")
                                if enrichments:
                                    tool.description = (
                                        f"{base_description}\n"
                                        + "\n".join(enrichments)
                                    )
                                else:
                                    tool.description = base_description
                                break
                        return tools

                    server = server.prepared(_prepare_tools)

                self._mcp_toolsets.append(server)
                logger.info(f"[ToolRegistry] Loaded MCP toolset for server '{config.name}'")

            except Exception as e:
                logger.error(
                    f"[ToolRegistry] Failed to create toolset for MCP server "
                    f"'{config.name}': {e}"
                )
                continue

        logger.info(f"[ToolRegistry] Loaded {len(self._mcp_toolsets)} MCP toolsets")
        return len(self._mcp_toolsets)

    def unload_mcp_toolsets(self) -> int:
        """
        Clear all MCP toolsets.

        Returns:
            Number of toolsets removed.
        """
        count = len(self._mcp_toolsets)
        self._mcp_toolsets.clear()
        if count > 0:
            logger.info(f"[ToolRegistry] Unloaded {count} MCP toolsets")
        return count

    def get_mcp_toolsets(self) -> List[Any]:
        """
        Get all loaded MCP toolsets.

        Returns:
            List of native Pydantic AI MCPServer* instances.
        """
        return self._mcp_toolsets

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._tools

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)
