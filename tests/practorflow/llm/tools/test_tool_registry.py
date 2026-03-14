"""
Unit tests for ToolRegistry.

Tests cover all ToolRegistry methods for 100% code coverage.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.llm.tools.mcp.types import (
    MCPServerConfig,
    TransportType,
    StdioConfig,
    HttpConfig,
)
from practorflow.llm.tools.user_preferences import UserToolPreferences, EnabledToolEntry
from tests.practorflow.llm.tools.tools_common import create_mock_tool


class TestToolRegistryInit:
    """Tests for ToolRegistry.__init__"""

    def test_init_creates_empty_registry(self):
        """ToolRegistry initializes with empty tools dict."""
        registry = ToolRegistry()

        assert len(registry) == 0

    def test_init_document_scope_is_none(self):
        """ToolRegistry initializes with None document scope."""
        registry = ToolRegistry()

        assert registry.get_document_scope() is None

    def test_init_last_result_is_none(self):
        """ToolRegistry initializes with None last result."""
        registry = ToolRegistry()

        assert registry.get_last_result() is None


class TestToolRegistryRegister:
    """Tests for ToolRegistry.register()"""

    def test_register_adds_tool(self):
        """register() adds tool to registry."""
        registry = ToolRegistry()
        tool = create_mock_tool(name="test_tool")

        registry.register(tool)

        assert "test_tool" in registry

    def test_register_duplicate_raises_value_error(self):
        """register() raises ValueError for duplicate tool name."""
        registry = ToolRegistry()
        tool1 = create_mock_tool(name="duplicate")
        tool2 = create_mock_tool(name="duplicate")
        registry.register(tool1)

        with pytest.raises(ValueError, match="Tool already registered: duplicate"):
            registry.register(tool2)

    def test_register_multiple_tools(self):
        """register() can add multiple different tools."""
        registry = ToolRegistry()
        tool1 = create_mock_tool(name="tool_a")
        tool2 = create_mock_tool(name="tool_b")

        registry.register(tool1)
        registry.register(tool2)

        assert len(registry) == 2
        assert "tool_a" in registry
        assert "tool_b" in registry


class TestToolRegistryUnregister:
    """Tests for ToolRegistry.unregister()"""

    def test_unregister_removes_tool(self):
        """unregister() removes existing tool and returns True."""
        registry = ToolRegistry()
        tool = create_mock_tool(name="to_remove")
        registry.register(tool)

        result = registry.unregister("to_remove")

        assert result is True
        assert "to_remove" not in registry

    def test_unregister_nonexistent_returns_false(self):
        """unregister() returns False for non-existent tool."""
        registry = ToolRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_unregister_does_not_affect_other_tools(self):
        """unregister() only removes specified tool."""
        registry = ToolRegistry()
        tool1 = create_mock_tool(name="keep")
        tool2 = create_mock_tool(name="remove")
        registry.register(tool1)
        registry.register(tool2)

        registry.unregister("remove")

        assert "keep" in registry
        assert len(registry) == 1


class TestToolRegistryGet:
    """Tests for ToolRegistry.get()"""

    def test_get_returns_registered_tool(self):
        """get() returns tool instance for registered name."""
        registry = ToolRegistry()
        tool = create_mock_tool(name="findme")
        registry.register(tool)

        result = registry.get("findme")

        assert result is tool

    def test_get_returns_none_for_nonexistent(self):
        """get() returns None for non-existent tool name."""
        registry = ToolRegistry()

        result = registry.get("nonexistent")

        assert result is None


class TestToolRegistryListTools:
    """Tests for ToolRegistry.list_tools()"""

    def test_list_tools_empty_registry(self):
        """list_tools() returns empty list for empty registry."""
        registry = ToolRegistry()

        result = registry.list_tools()

        assert result == []

    def test_list_tools_returns_tool_names(self):
        """list_tools() returns list of registered tool names."""
        registry = ToolRegistry()
        registry.register(create_mock_tool(name="alpha"))
        registry.register(create_mock_tool(name="beta"))

        result = registry.list_tools()

        assert set(result) == {"alpha", "beta"}


class TestToolRegistryGetAllTools:
    """Tests for ToolRegistry.get_all_tools()"""

    def test_get_all_tools_empty_registry(self):
        """get_all_tools() returns empty list for empty registry."""
        registry = ToolRegistry()

        result = registry.get_all_tools()

        assert result == []

    def test_get_all_tools_returns_tool_instances(self):
        """get_all_tools() returns list of tool instances."""
        registry = ToolRegistry()
        tool1 = create_mock_tool(name="first")
        tool2 = create_mock_tool(name="second")
        registry.register(tool1)
        registry.register(tool2)

        result = registry.get_all_tools()

        assert len(result) == 2
        assert tool1 in result
        assert tool2 in result


class TestToolRegistryGetSchemas:
    """Tests for ToolRegistry.get_schemas()"""

    def test_get_schemas_empty_registry(self):
        """get_schemas() returns empty list for empty registry."""
        registry = ToolRegistry()

        result = registry.get_schemas()

        assert result == []

    def test_get_schemas_returns_tool_schemas(self):
        """get_schemas() returns list of tool schemas."""
        registry = ToolRegistry()
        registry.register(create_mock_tool(name="schema_tool"))

        result = registry.get_schemas()

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "schema_tool"

    def test_get_schemas_multiple_tools(self):
        """get_schemas() returns schemas for all registered tools."""
        registry = ToolRegistry()
        registry.register(create_mock_tool(name="tool_x"))
        registry.register(create_mock_tool(name="tool_y"))

        result = registry.get_schemas()

        assert len(result) == 2
        names = {s["function"]["name"] for s in result}
        assert names == {"tool_x", "tool_y"}


class TestToolRegistryDocumentScope:
    """Tests for document scope methods."""

    def test_set_document_scope_with_set(self):
        """set_document_scope() stores document IDs as set."""
        registry = ToolRegistry()

        registry.set_document_scope({"doc1", "doc2"})

        assert registry.get_document_scope() == {"doc1", "doc2"}

    def test_set_document_scope_with_none(self):
        """set_document_scope() with None clears scope."""
        registry = ToolRegistry()
        registry.set_document_scope({"doc1"})

        registry.set_document_scope(None)

        assert registry.get_document_scope() is None

    def test_set_document_scope_copies_input(self):
        """set_document_scope() creates copy of input set."""
        registry = ToolRegistry()
        input_set = {"doc1", "doc2"}

        registry.set_document_scope(input_set)
        input_set.add("doc3")

        assert registry.get_document_scope() == {"doc1", "doc2"}

    def test_get_document_scope_returns_none_initially(self):
        """get_document_scope() returns None before any scope is set."""
        registry = ToolRegistry()

        assert registry.get_document_scope() is None

    def test_clear_document_scope(self):
        """clear_document_scope() sets scope to None."""
        registry = ToolRegistry()
        registry.set_document_scope({"doc1", "doc2"})

        registry.clear_document_scope()

        assert registry.get_document_scope() is None


class TestToolRegistryExecute:
    """Tests for ToolRegistry.execute()"""

    async def test_execute_calls_tool(self):
        """execute() calls the tool with provided kwargs."""
        registry = ToolRegistry()
        tool = create_mock_tool(name="exec_tool", data="executed")
        registry.register(tool)

        result = await registry.execute("exec_tool", query="test query")

        assert result.success is True
        assert result.data == "executed"
        assert tool.get_last_call_kwargs() == {"query": "test query"}

    async def test_execute_nonexistent_tool_returns_error(self):
        """execute() returns error result for non-existent tool."""
        registry = ToolRegistry()

        result = await registry.execute("nonexistent", query="test")

        assert result.success is False
        assert result.error == "Tool not found: nonexistent"

    async def test_execute_stores_last_result(self):
        """execute() stores result as last_result."""
        registry = ToolRegistry()
        tool = create_mock_tool(name="result_tool", data="stored")
        registry.register(tool)

        await registry.execute("result_tool", query="test")

        assert registry.get_last_result().data == "stored"

    async def test_execute_stores_error_result(self):
        """execute() stores error result for non-existent tool."""
        registry = ToolRegistry()

        await registry.execute("missing")

        last_result = registry.get_last_result()
        assert last_result.success is False
        assert "Tool not found" in last_result.error

    async def test_execute_injects_document_scope(self):
        """execute() injects document scope to tools that support it."""
        registry = ToolRegistry()
        tool = create_mock_tool(
            name="scoped_tool",
            supports_document_scope=True,
        )
        registry.register(tool)
        registry.set_document_scope({"doc_a", "doc_b"})

        await registry.execute("scoped_tool", query="test")

        assert tool.get_document_scope() == {"doc_a", "doc_b"}

    async def test_execute_no_scope_injection_without_support(self):
        """execute() does not inject scope to tools without set_document_scope."""
        registry = ToolRegistry()
        tool = create_mock_tool(
            name="no_scope_tool",
            supports_document_scope=False,
        )
        registry.register(tool)
        registry.set_document_scope({"doc1"})

        await registry.execute("no_scope_tool", query="test")

        assert tool.get_document_scope() is None


class TestToolRegistryLastResult:
    """Tests for last result methods."""

    def test_get_last_result_returns_none_initially(self):
        """get_last_result() returns None before any execution."""
        registry = ToolRegistry()

        assert registry.get_last_result() is None

    async def test_get_last_result_after_execution(self):
        """get_last_result() returns result from last execute()."""
        registry = ToolRegistry()
        tool = create_mock_tool(name="last_tool", data="last data")
        registry.register(tool)

        await registry.execute("last_tool", query="test")

        result = registry.get_last_result()
        assert result.data == "last data"

    async def test_clear_last_result(self):
        """clear_last_result() sets last_result to None."""
        registry = ToolRegistry()
        tool = create_mock_tool(name="clear_tool")
        registry.register(tool)
        await registry.execute("clear_tool", query="test")

        registry.clear_last_result()

        assert registry.get_last_result() is None


class TestToolRegistryPendingContext:
    """Tests for pending context methods."""

    def test_has_pending_context_false_when_no_result(self):
        """has_pending_context() returns False when no result exists."""
        registry = ToolRegistry()

        assert registry.has_pending_context() is False

    async def test_has_pending_context_false_when_failed(self):
        """has_pending_context() returns False when last result failed."""
        registry = ToolRegistry()
        tool = create_mock_tool(
            name="fail_tool",
            success=False,
            error="failed",
        )
        registry.register(tool)
        await registry.execute("fail_tool", query="test")

        assert registry.has_pending_context() is False

    async def test_has_pending_context_false_when_data_is_none(self):
        """has_pending_context() returns False when data is None."""
        registry = ToolRegistry()
        tool = create_mock_tool(
            name="none_tool",
            success=True,
            data=None,
        )
        registry.register(tool)
        await registry.execute("none_tool", query="test")

        assert registry.has_pending_context() is False

    async def test_has_pending_context_true_when_success_with_data(self):
        """has_pending_context() returns True when success with data."""
        registry = ToolRegistry()
        tool = create_mock_tool(
            name="context_tool",
            success=True,
            data="context data",
        )
        registry.register(tool)
        await registry.execute("context_tool", query="test")

        assert registry.has_pending_context() is True

    async def test_consume_context_returns_context_string(self):
        """consume_context() returns context string from result."""
        registry = ToolRegistry()
        tool = create_mock_tool(
            name="consume_tool",
            success=True,
            data="consumable context",
        )
        registry.register(tool)
        await registry.execute("consume_tool", query="test")

        context = registry.consume_context()

        assert context == "consumable context"

    async def test_consume_context_clears_last_result(self):
        """consume_context() sets last_result to None."""
        registry = ToolRegistry()
        tool = create_mock_tool(name="clear_on_consume", data="data")
        registry.register(tool)
        await registry.execute("clear_on_consume", query="test")

        registry.consume_context()

        assert registry.get_last_result() is None

    def test_consume_context_returns_none_when_no_pending(self):
        """consume_context() returns None when no pending context."""
        registry = ToolRegistry()

        result = registry.consume_context()

        assert result is None

    async def test_consume_context_returns_none_when_failed(self):
        """consume_context() returns None when last result failed."""
        registry = ToolRegistry()
        tool = create_mock_tool(name="fail", success=False, error="err")
        registry.register(tool)
        await registry.execute("fail", query="test")

        result = registry.consume_context()

        assert result is None


class TestToolRegistryDunderMethods:
    """Tests for ToolRegistry dunder methods."""

    def test_contains_returns_true_for_registered(self):
        """__contains__ returns True for registered tool."""
        registry = ToolRegistry()
        registry.register(create_mock_tool(name="exists"))

        assert ("exists" in registry) is True

    def test_contains_returns_false_for_unregistered(self):
        """__contains__ returns False for unregistered tool."""
        registry = ToolRegistry()

        assert ("missing" in registry) is False

    def test_len_empty_registry(self):
        """__len__ returns 0 for empty registry."""
        registry = ToolRegistry()

        assert len(registry) == 0

    def test_len_with_tools(self):
        """__len__ returns count of registered tools."""
        registry = ToolRegistry()
        registry.register(create_mock_tool(name="one"))
        registry.register(create_mock_tool(name="two"))
        registry.register(create_mock_tool(name="three"))

        assert len(registry) == 3


class TestLoadApiToolsForUser:
    """Tests for ToolRegistry.load_api_tools_for_user()"""

    def test_returns_zero_when_factory_not_initialized(self):
        """load_api_tools_for_user() returns 0 when factory is not initialized."""
        registry = ToolRegistry()

        with patch(
            "practorflow.llm.tools.tool_registry.ToolRegistry.load_api_tools_for_user",
            wraps=None,
        ):
            pass

        with patch(
            "practorflow.llm.tools.api.factory.is_factory_initialized",
            return_value=False,
        ):
            result = registry.load_api_tools_for_user("user-1")

        assert result == 0

    def test_loads_tools_for_user(self):
        """load_api_tools_for_user() loads and registers tools from factory."""
        registry = ToolRegistry()
        mock_tool = create_mock_tool(name="api_tool_1")
        mock_factory = MagicMock()
        mock_factory.create_tools_for_user.return_value = [mock_tool]

        with (
            patch(
                "practorflow.llm.tools.api.factory.is_factory_initialized",
                return_value=True,
            ),
            patch(
                "practorflow.llm.tools.api.factory.get_factory",
                return_value=mock_factory,
            ),
        ):
            result = registry.load_api_tools_for_user("user-1")

        assert result == 1
        assert "api_tool_1" in registry

    def test_skips_already_registered_tool(self):
        """load_api_tools_for_user() skips tools already in registry."""
        registry = ToolRegistry()
        existing_tool = create_mock_tool(name="existing")
        registry.register(existing_tool)

        duplicate_tool = create_mock_tool(name="existing")
        mock_factory = MagicMock()
        mock_factory.create_tools_for_user.return_value = [duplicate_tool]

        with (
            patch(
                "practorflow.llm.tools.api.factory.is_factory_initialized",
                return_value=True,
            ),
            patch(
                "practorflow.llm.tools.api.factory.get_factory",
                return_value=mock_factory,
            ),
        ):
            result = registry.load_api_tools_for_user("user-1")

        assert result == 0
        assert len(registry) == 1

    def test_unregisters_previous_user_tools_when_switching_users(self):
        """load_api_tools_for_user() clears tools when switching user."""
        registry = ToolRegistry()

        old_tool = MagicMock()
        old_tool.name = "old_user_tool"
        old_tool.user_id = "user-old"
        registry._tools["old_user_tool"] = old_tool
        registry._loaded_user_id = "user-old"

        new_tool = create_mock_tool(name="new_user_tool")
        mock_factory = MagicMock()
        mock_factory.create_tools_for_user.return_value = [new_tool]

        with (
            patch(
                "practorflow.llm.tools.api.factory.is_factory_initialized",
                return_value=True,
            ),
            patch(
                "practorflow.llm.tools.api.factory.get_factory",
                return_value=mock_factory,
            ),
        ):
            result = registry.load_api_tools_for_user("user-new")

        assert "old_user_tool" not in registry
        assert "new_user_tool" in registry
        assert result == 1


class TestUnregisterApiTools:
    """Tests for ToolRegistry._unregister_api_tools()"""

    def test_removes_tools_with_user_id(self):
        """_unregister_api_tools() removes tools that have a user_id attribute."""
        registry = ToolRegistry()

        api_tool = MagicMock()
        api_tool.name = "api_tool"
        api_tool.user_id = "user-1"
        registry._tools["api_tool"] = api_tool

        regular_tool = create_mock_tool(name="regular")
        registry.register(regular_tool)

        count = registry._unregister_api_tools()

        assert count == 1
        assert "api_tool" not in registry
        assert "regular" in registry

    def test_returns_zero_when_no_api_tools(self):
        """_unregister_api_tools() returns 0 when no api tools registered."""
        registry = ToolRegistry()
        registry.register(create_mock_tool(name="builtin"))

        count = registry._unregister_api_tools()

        assert count == 0
        assert "builtin" in registry

    def test_clears_loaded_user_id(self):
        """_unregister_api_tools() clears _loaded_user_id."""
        registry = ToolRegistry()
        registry._loaded_user_id = "user-1"

        registry._unregister_api_tools()

        assert registry._loaded_user_id is None


class TestMCPServerStore:
    """Tests for set_mcp_server_store()."""

    def test_set_mcp_server_store(self):
        """set_mcp_server_store() stores the provided store."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        registry.set_mcp_server_store(mock_store)

        assert registry._mcp_server_store is mock_store


class TestUnloadMCPToolsets:
    """Tests for unload_mcp_toolsets()."""

    def test_unload_clears_toolsets(self):
        """unload_mcp_toolsets() removes all toolsets and returns count."""
        registry = ToolRegistry()
        registry._mcp_toolsets = [MagicMock(), MagicMock()]

        count = registry.unload_mcp_toolsets()

        assert count == 2
        assert registry._mcp_toolsets == []

    def test_unload_returns_zero_when_empty(self):
        """unload_mcp_toolsets() returns 0 when no toolsets loaded."""
        registry = ToolRegistry()

        count = registry.unload_mcp_toolsets()

        assert count == 0


class TestGetMCPToolsets:
    """Tests for get_mcp_toolsets()."""

    def test_returns_empty_list_initially(self):
        """get_mcp_toolsets() returns empty list when no toolsets loaded."""
        registry = ToolRegistry()

        result = registry.get_mcp_toolsets()

        assert result == []

    def test_returns_loaded_toolsets(self):
        """get_mcp_toolsets() returns the list of loaded toolsets."""
        registry = ToolRegistry()
        mock_toolset = MagicMock()
        registry._mcp_toolsets = [mock_toolset]

        result = registry.get_mcp_toolsets()

        assert result == [mock_toolset]


class TestLoadMCPToolsets:
    """Tests for load_mcp_toolsets()."""

    def test_returns_zero_when_no_store(self):
        """load_mcp_toolsets() returns 0 when mcp_server_store is None."""
        registry = ToolRegistry()

        with patch(
            "pydantic_ai.mcp.MCPServerStdio",
            MagicMock(),
        ):
            result = registry.load_mcp_toolsets()

        assert result == 0

    def test_returns_zero_when_no_servers_configured(self):
        """load_mcp_toolsets() returns 0 when store has no servers."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_store.list.return_value = []
        registry.set_mcp_server_store(mock_store)

        with patch("pydantic_ai.mcp.MCPServerStdio", MagicMock()):
            result = registry.load_mcp_toolsets()

        assert result == 0

    def test_loads_stdio_server(self):
        """load_mcp_toolsets() creates MCPServerStdio for stdio transport."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        config = MCPServerConfig(
            name="stdio-server",
            transport=TransportType.STDIO,
            stdio_config=StdioConfig(command="my-cmd", args=["--arg"]),
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        mock_server = MagicMock()
        mock_stdio_cls = MagicMock(return_value=mock_server)

        with patch("pydantic_ai.mcp.MCPServerStdio", mock_stdio_cls):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    result = registry.load_mcp_toolsets()

        assert result == 1
        mock_stdio_cls.assert_called_once_with(
            command="my-cmd",
            args=["--arg"],
            env=None,
        )

    def test_loads_sse_server(self):
        """load_mcp_toolsets() creates MCPServerSSE for sse transport."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        config = MCPServerConfig(
            name="sse-server",
            transport=TransportType.SSE,
            http_config=HttpConfig(url="http://localhost:8080"),
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        mock_server = MagicMock()
        mock_sse_cls = MagicMock(return_value=mock_server)

        with patch("pydantic_ai.mcp.MCPServerStdio", MagicMock()):
            with patch("pydantic_ai.mcp.MCPServerSSE", mock_sse_cls):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    result = registry.load_mcp_toolsets()

        assert result == 1
        mock_sse_cls.assert_called_once()

    def test_loads_streamable_http_server(self):
        """load_mcp_toolsets() creates MCPServerStreamableHTTP for streamable_http transport."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        config = MCPServerConfig(
            name="http-server",
            transport=TransportType.STREAMABLE_HTTP,
            http_config=HttpConfig(url="http://localhost:9090"),
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        mock_server = MagicMock()
        mock_http_cls = MagicMock(return_value=mock_server)

        with patch("pydantic_ai.mcp.MCPServerStdio", MagicMock()):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", mock_http_cls):
                    result = registry.load_mcp_toolsets()

        assert result == 1
        mock_http_cls.assert_called_once()

    def test_skips_server_not_in_enabled_list(self):
        """load_mcp_toolsets() skips servers not in user enabled list."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        config = MCPServerConfig(
            name="disabled-server",
            transport=TransportType.STDIO,
            stdio_config=StdioConfig(command="cmd"),
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        prefs = UserToolPreferences(user_id="u1", enabled_mcp_servers=["other-server"])

        with patch("pydantic_ai.mcp.MCPServerStdio", MagicMock()):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    result = registry.load_mcp_toolsets(user_preferences=prefs)

        assert result == 0

    def test_skips_stdio_server_missing_stdio_config(self):
        """load_mcp_toolsets() skips stdio server when stdio_config is missing."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        config = MCPServerConfig(
            name="bad-stdio",
            transport=TransportType.STDIO,
            stdio_config=None,
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        with patch("pydantic_ai.mcp.MCPServerStdio", MagicMock()):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    result = registry.load_mcp_toolsets()

        assert result == 0

    def test_skips_sse_server_missing_http_config(self):
        """load_mcp_toolsets() skips sse server when http_config is missing."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        config = MCPServerConfig(
            name="bad-sse",
            transport=TransportType.SSE,
            http_config=None,
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        with patch("pydantic_ai.mcp.MCPServerStdio", MagicMock()):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    result = registry.load_mcp_toolsets()

        assert result == 0

    def test_applies_tool_filtering_when_preferences_provided(self):
        """load_mcp_toolsets() calls server.filtered() when user preferences specify mcp tools."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        config = MCPServerConfig(
            name="filtered-server",
            transport=TransportType.STDIO,
            stdio_config=StdioConfig(command="cmd"),
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        mock_server = MagicMock()
        mock_server.filtered.return_value = mock_server
        mock_stdio_cls = MagicMock(return_value=mock_server)

        prefs = UserToolPreferences(
            user_id="u1",
            enabled_mcp_servers=["filtered-server"],
            enabled_tools=[EnabledToolEntry(type="mcp", id="tool-abc")],
        )

        with patch("pydantic_ai.mcp.MCPServerStdio", mock_stdio_cls):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    result = registry.load_mcp_toolsets(user_preferences=prefs)

        assert result == 1
        mock_server.filtered.assert_called_once_with(allowed_tools=["tool-abc"])

    def test_handles_exception_during_toolset_creation(self):
        """load_mcp_toolsets() skips server and continues when creation fails."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        config = MCPServerConfig(
            name="error-server",
            transport=TransportType.STDIO,
            stdio_config=StdioConfig(command="cmd"),
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        mock_stdio_cls = MagicMock(side_effect=RuntimeError("connection failed"))

        with patch("pydantic_ai.mcp.MCPServerStdio", mock_stdio_cls):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    result = registry.load_mcp_toolsets()

        assert result == 0

    def test_skips_streamable_http_server_missing_http_config(self):
        """load_mcp_toolsets() skips streamable_http server when http_config is missing."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        config = MCPServerConfig(
            name="bad-streamable",
            transport=TransportType.STREAMABLE_HTTP,
            http_config=None,
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        with patch("pydantic_ai.mcp.MCPServerStdio", MagicMock()):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    result = registry.load_mcp_toolsets()

        assert result == 0

    def test_skips_unsupported_transport(self):
        """load_mcp_toolsets() skips server with unsupported transport type."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        config = MCPServerConfig(
            name="unsupported-server",
            transport=TransportType.STDIO,
            stdio_config=StdioConfig(command="cmd"),
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        # Patch the transport check to simulate an unsupported transport
        with patch("pydantic_ai.mcp.MCPServerStdio", MagicMock()):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    with patch.object(config, "transport", "unknown_transport"):
                        result = registry.load_mcp_toolsets()

        assert result == 0

    def test_applies_tool_description_enrichment(self):
        """load_mcp_toolsets() calls server.prepared() when tools have config."""
        from practorflow.llm.tools.mcp.types import MCPToolConfig

        registry = ToolRegistry()
        mock_store = MagicMock()

        tool_cfg = MCPToolConfig(
            name="my-tool",
            description="base desc",
            purpose="do something",
            use_when=["condition A"],
            do_not_use_when=["condition B"],
            category="util",
            tags=["tag1"],
            keywords=["kw1"],
        )
        config = MCPServerConfig(
            name="enriched-server",
            transport=TransportType.STDIO,
            stdio_config=StdioConfig(command="cmd"),
            tools=[tool_cfg],
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        mock_server = MagicMock()
        mock_server.prepared.return_value = mock_server
        mock_stdio_cls = MagicMock(return_value=mock_server)

        with patch("pydantic_ai.mcp.MCPServerStdio", mock_stdio_cls):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    result = registry.load_mcp_toolsets()

        assert result == 1
        mock_server.prepared.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_tools_callback_enriches_description(self):
        """_prepare_tools callback enriches matching tool descriptions."""
        from practorflow.llm.tools.mcp.types import MCPToolConfig

        registry = ToolRegistry()
        mock_store = MagicMock()

        tool_cfg = MCPToolConfig(
            name="my-tool",
            description="base desc",
            purpose="do something",
            use_when=["condition A"],
            do_not_use_when=["condition B"],
            category="util",
            tags=["tag1"],
            keywords=["kw1"],
        )
        config = MCPServerConfig(
            name="enriched-server",
            transport=TransportType.STDIO,
            stdio_config=StdioConfig(command="cmd"),
            tools=[tool_cfg],
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        captured_callback = None

        def capture_prepared(cb):
            nonlocal captured_callback
            captured_callback = cb
            return mock_server

        mock_server = MagicMock()
        mock_server.filtered.return_value = mock_server
        mock_server.prepared.side_effect = capture_prepared
        mock_stdio_cls = MagicMock(return_value=mock_server)

        with patch("pydantic_ai.mcp.MCPServerStdio", mock_stdio_cls):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    registry.load_mcp_toolsets()

        assert captured_callback is not None

        mock_tool = MagicMock()
        mock_tool.name = "my-tool"
        mock_tool.description = "original"

        result = await captured_callback(None, [mock_tool])

        assert result == [mock_tool]
        assert "base desc" in mock_tool.description
        assert "Purpose: do something" in mock_tool.description
        assert "Use when: condition A" in mock_tool.description
        assert "Do not use when: condition B" in mock_tool.description
        assert "Category: util" in mock_tool.description
        assert "Tags: tag1" in mock_tool.description
        assert "Keywords: kw1" in mock_tool.description

    @pytest.mark.asyncio
    async def test_prepare_tools_callback_skips_nonmatching_tool(self):
        """_prepare_tools callback leaves non-matching tools unchanged."""
        from practorflow.llm.tools.mcp.types import MCPToolConfig

        registry = ToolRegistry()
        mock_store = MagicMock()

        tool_cfg = MCPToolConfig(
            name="my-tool",
            description="base desc",
        )
        config = MCPServerConfig(
            name="server",
            transport=TransportType.STDIO,
            stdio_config=StdioConfig(command="cmd"),
            tools=[tool_cfg],
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        captured_callback = None

        def capture_prepared(cb):
            nonlocal captured_callback
            captured_callback = cb
            return mock_server

        mock_server = MagicMock()
        mock_server.filtered.return_value = mock_server
        mock_server.prepared.side_effect = capture_prepared
        mock_stdio_cls = MagicMock(return_value=mock_server)

        with patch("pydantic_ai.mcp.MCPServerStdio", mock_stdio_cls):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    registry.load_mcp_toolsets()

        mock_tool = MagicMock()
        mock_tool.name = "other-tool"
        mock_tool.description = "unchanged"

        result = await captured_callback(None, [mock_tool])

        assert result == [mock_tool]
        assert mock_tool.description == "unchanged"

    @pytest.mark.asyncio
    async def test_prepare_tools_callback_no_enrichments_uses_base_description(self):
        """_prepare_tools sets base description when no enrichment fields are set."""
        from practorflow.llm.tools.mcp.types import MCPToolConfig

        registry = ToolRegistry()
        mock_store = MagicMock()

        tool_cfg = MCPToolConfig(
            name="plain-tool",
            description="plain desc",
        )
        config = MCPServerConfig(
            name="plain-server",
            transport=TransportType.STDIO,
            stdio_config=StdioConfig(command="cmd"),
            tools=[tool_cfg],
        )
        mock_store.list.return_value = [config]
        registry.set_mcp_server_store(mock_store)

        captured_callback = None

        def capture_prepared(cb):
            nonlocal captured_callback
            captured_callback = cb
            return mock_server

        mock_server = MagicMock()
        mock_server.filtered.return_value = mock_server
        mock_server.prepared.side_effect = capture_prepared
        mock_stdio_cls = MagicMock(return_value=mock_server)

        with patch("pydantic_ai.mcp.MCPServerStdio", mock_stdio_cls):
            with patch("pydantic_ai.mcp.MCPServerSSE", MagicMock()):
                with patch("pydantic_ai.mcp.MCPServerStreamableHTTP", MagicMock()):
                    registry.load_mcp_toolsets()

        mock_tool = MagicMock()
        mock_tool.name = "plain-tool"
        mock_tool.description = "original"

        result = await captured_callback(None, [mock_tool])

        assert result == [mock_tool]
        assert mock_tool.description == "plain desc"
