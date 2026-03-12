"""
Tests for builtin_tools builder.

Tests the inner functions produced by build_builtin_tools() by
capturing them from the Tool constructor and calling them with mock contexts.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from practorflow.services.tools.builders.builtin_tools import build_builtin_tools


def _capture_inner_functions():
    """Run build_builtin_tools() with Tool mocked to capture inner functions."""
    captured = []

    def capture(f, takes_ctx=False):
        captured.append(f)
        return MagicMock()

    with patch("practorflow.services.tools.builders.builtin_tools.Tool", side_effect=capture):
        build_builtin_tools()

    return captured


def _make_ctx(tool_registry, document_scope=None, knowledge_store=None):
    ctx = MagicMock()
    ctx.deps.tool_registry = tool_registry
    ctx.deps.document_scope = document_scope
    ctx.deps.knowledge_store = knowledge_store or MagicMock()
    return ctx


class TestBuildBuiltinTools:
    def test_returns_seven_tools(self):
        """build_builtin_tools() returns exactly 7 Tool objects."""
        tools = build_builtin_tools()

        assert len(tools) == 7

    def test_all_items_are_tool_objects(self):
        """build_builtin_tools() returns pydantic_ai Tool instances."""
        from pydantic_ai import Tool

        tools = build_builtin_tools()

        for tool in tools:
            assert isinstance(tool, Tool)


class TestExecuteToolInnerFunction:
    """Tests for the execute_tool inner function (index 0)."""

    @pytest.mark.asyncio
    async def test_tool_not_found_returns_error(self):
        captured = _capture_inner_functions()
        execute_tool = captured[0]

        registry = MagicMock()
        registry.__contains__ = MagicMock(return_value=False)
        ctx = _make_ctx(registry)

        result = await execute_tool(ctx, tool_name="missing", tool_args=None)

        assert "not found" in result

    @pytest.mark.asyncio
    async def test_success_returns_data_string(self):
        captured = _capture_inner_functions()
        execute_tool = captured[0]

        registry = MagicMock()
        registry.__contains__ = MagicMock(return_value=True)
        registry.execute = AsyncMock(return_value=MagicMock(success=True, data="got it"))
        ctx = _make_ctx(registry)

        result = await execute_tool(ctx, tool_name="my_tool", tool_args={"q": "test"})

        assert result == "got it"

    @pytest.mark.asyncio
    async def test_success_with_none_data_returns_default(self):
        captured = _capture_inner_functions()
        execute_tool = captured[0]

        registry = MagicMock()
        registry.__contains__ = MagicMock(return_value=True)
        registry.execute = AsyncMock(return_value=MagicMock(success=True, data=None))
        ctx = _make_ctx(registry)

        result = await execute_tool(ctx, tool_name="tool", tool_args=None)

        assert result == "Tool executed successfully"

    @pytest.mark.asyncio
    async def test_failure_returns_error_string(self):
        captured = _capture_inner_functions()
        execute_tool = captured[0]

        registry = MagicMock()
        registry.__contains__ = MagicMock(return_value=True)
        registry.execute = AsyncMock(return_value=MagicMock(success=False, error="fail msg"))
        ctx = _make_ctx(registry)

        result = await execute_tool(ctx, tool_name="tool", tool_args=None)

        assert "fail msg" in result

    @pytest.mark.asyncio
    async def test_exception_returns_error_string(self):
        captured = _capture_inner_functions()
        execute_tool = captured[0]

        registry = MagicMock()
        registry.__contains__ = MagicMock(return_value=True)
        registry.execute = AsyncMock(side_effect=RuntimeError("boom"))
        ctx = _make_ctx(registry)

        result = await execute_tool(ctx, tool_name="tool", tool_args=None)

        assert "failed" in result


class TestSearchKnowledgeInnerFunction:
    """Tests for the search_knowledge inner function (index 1)."""

    @pytest.mark.asyncio
    async def test_no_document_scope_returns_empty(self):
        captured = _capture_inner_functions()
        search_knowledge = captured[1]

        ctx = _make_ctx(MagicMock(), document_scope=None)

        result = await search_knowledge(ctx, query="anything")

        assert result == ""

    @pytest.mark.asyncio
    async def test_no_results_returns_empty(self):
        captured = _capture_inner_functions()
        search_knowledge = captured[1]

        knowledge_store = MagicMock()
        knowledge_store.search_scoped.return_value = []
        ctx = _make_ctx(MagicMock(), document_scope={"doc1"}, knowledge_store=knowledge_store)

        result = await search_knowledge(ctx, query="query")

        assert result == ""

    @pytest.mark.asyncio
    async def test_results_returns_formatted_string(self):
        captured = _capture_inner_functions()
        search_knowledge = captured[1]

        knowledge_store = MagicMock()
        knowledge_store.search_scoped.return_value = [
            {"text": "relevant content", "filename": "doc.txt"}
        ]
        ctx = _make_ctx(MagicMock(), document_scope={"doc1"}, knowledge_store=knowledge_store)

        result = await search_knowledge(ctx, query="query")

        assert "relevant content" in result
        assert "doc.txt" in result


class TestSearchWebInnerFunction:
    """Tests for the search_web inner function (index 2)."""

    @pytest.mark.asyncio
    async def test_success_returns_data(self):
        captured = _capture_inner_functions()
        search_web = captured[2]

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=True, data="web results"))
        ctx = _make_ctx(registry)

        result = await search_web(ctx, query="python")

        assert result == "web results"

    @pytest.mark.asyncio
    async def test_failure_returns_empty(self):
        captured = _capture_inner_functions()
        search_web = captured[2]

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=False, error="err"))
        ctx = _make_ctx(registry)

        result = await search_web(ctx, query="python")

        assert result == ""


class TestFetchWebpageInnerFunction:
    """Tests for the fetch_webpage inner function (index 3)."""

    @pytest.mark.asyncio
    async def test_success_returns_content(self):
        captured = _capture_inner_functions()
        fetch_webpage = captured[3]

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=True, data="page content"))
        ctx = _make_ctx(registry)

        result = await fetch_webpage(ctx, url="http://example.com")

        assert result == "page content"

    @pytest.mark.asyncio
    async def test_failure_returns_empty(self):
        captured = _capture_inner_functions()
        fetch_webpage = captured[3]

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=False, error="404"))
        ctx = _make_ctx(registry)

        result = await fetch_webpage(ctx, url="http://bad.url")

        assert result == ""


class TestSummarizeTextInnerFunction:
    """Tests for the summarize_text inner function (index 4)."""

    @pytest.mark.asyncio
    async def test_success_returns_summary(self):
        captured = _capture_inner_functions()
        summarize_text = captured[4]

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=True, data="summary text"))
        ctx = _make_ctx(registry)

        result = await summarize_text(ctx, text="long text", num_sentences=3)

        assert result == "summary text"

    @pytest.mark.asyncio
    async def test_failure_returns_empty(self):
        captured = _capture_inner_functions()
        summarize_text = captured[4]

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=False, error="err"))
        ctx = _make_ctx(registry)

        result = await summarize_text(ctx, text="text")

        assert result == ""

    @pytest.mark.asyncio
    async def test_success_with_none_data_returns_default(self):
        captured = _capture_inner_functions()
        summarize_text = captured[4]

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=True, data=None))
        ctx = _make_ctx(registry)

        result = await summarize_text(ctx, text="text")

        assert result == "No summary generated."


class TestTransformJsonInnerFunction:
    """Tests for the transform_json inner function (index 5)."""

    @pytest.mark.asyncio
    async def test_success_with_dict_returns_json_string(self):
        captured = _capture_inner_functions()
        transform_json = captured[5]

        registry = MagicMock()
        registry.execute = AsyncMock(
            return_value=MagicMock(success=True, data={"key": "val"})
        )
        ctx = _make_ctx(registry)

        result = await transform_json(ctx, json_data='{"key":"val"}', operation="parse")

        assert '"key"' in result
        assert '"val"' in result

    @pytest.mark.asyncio
    async def test_success_with_string_data_returns_string(self):
        captured = _capture_inner_functions()
        transform_json = captured[5]

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=True, data="scalar"))
        ctx = _make_ctx(registry)

        result = await transform_json(ctx, json_data="{}", operation="keys")

        assert result == "scalar"

    @pytest.mark.asyncio
    async def test_failure_returns_error_message(self):
        captured = _capture_inner_functions()
        transform_json = captured[5]

        registry = MagicMock()
        registry.execute = AsyncMock(
            return_value=MagicMock(success=False, error="parse error")
        )
        ctx = _make_ctx(registry)

        result = await transform_json(ctx, json_data="bad", operation="parse")

        assert "parse error" in result


class TestCalculateInnerFunction:
    """Tests for the calculate inner function (index 6)."""

    @pytest.mark.asyncio
    async def test_success_returns_result(self):
        captured = _capture_inner_functions()
        calculate = captured[6]

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=True, data="42"))
        ctx = _make_ctx(registry)

        result = await calculate(ctx, expression="6 * 7")

        assert result == "42"

    @pytest.mark.asyncio
    async def test_failure_returns_error_message(self):
        captured = _capture_inner_functions()
        calculate = captured[6]

        registry = MagicMock()
        registry.execute = AsyncMock(
            return_value=MagicMock(success=False, error="division by zero")
        )
        ctx = _make_ctx(registry)

        result = await calculate(ctx, expression="1/0")

        assert "division by zero" in result
