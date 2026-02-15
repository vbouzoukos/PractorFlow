"""
Unit tests for ToolRegistry.

Tests cover all ToolRegistry methods for 100% code coverage.
"""

import pytest

from practorflow.llm.tools.tool_registry import ToolRegistry
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
