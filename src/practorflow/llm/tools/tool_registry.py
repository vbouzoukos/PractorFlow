"""
Tool registry for managing available tools.

Provides centralized tool registration, lookup, and execution.
Supports dynamic loading of user-defined API tools via ApiToolFactory.
"""

from typing import Dict, List, Optional, Any
from practorflow.llm.tools.async_tool import AsyncBaseTool
from practorflow.llm.tools.base import ToolResult
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

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._tools

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)
