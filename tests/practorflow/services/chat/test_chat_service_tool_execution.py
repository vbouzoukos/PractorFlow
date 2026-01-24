"""
Unit tests for ChatService tool execution.

Tests:
- search_knowledge tool execution
- search_web tool execution
"""

from unittest.mock import MagicMock

import pytest

from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.services.chat.common_chat_service import (
    chat_service,
    mock_model_config,
    mock_model_pool,
    mock_session_store,
    mock_web_search_tool,
    sample_session,
)


def _capture_tools(chat_service):
    """Helper to capture registered tool functions from chat_service."""
    mock_agent = MagicMock()
    captured_tools = {}

    def capture_tool(func):
        captured_tools[func.__name__] = func
        return func

    mock_agent.tool = capture_tool
    chat_service._register_tools(mock_agent)
    return captured_tools


def _create_mock_run_context(knowledge_store, web_search_tool, document_scope=None):
    """Helper to create mock RunContext with ChatDeps."""
    mock_ctx = MagicMock()
    mock_ctx.deps = MagicMock()
    mock_ctx.deps.knowledge_store = knowledge_store
    mock_ctx.deps.document_scope = document_scope
    mock_ctx.deps.web_search_tool = web_search_tool
    return mock_ctx


# ============================================================================
# Test search_knowledge tool execution
# ============================================================================


@pytest.mark.asyncio
async def test_search_knowledge_returns_formatted_results(
    chat_service,
    mock_knowledge_store,
    mock_web_search_tool,
):
    """Test search_knowledge returns formatted results from knowledge store."""
    mock_knowledge_store.search_scoped.return_value = [
        {
            "text": "First result text",
            "metadata": {"filename": "doc1.txt"},
            "similarity": 0.95,
        },
        {
            "text": "Second result text",
            "metadata": {"filename": "doc2.pdf"},
            "similarity": 0.85,
        },
    ]

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(
        mock_knowledge_store,
        mock_web_search_tool,
        document_scope={"doc-1", "doc-2"},
    )

    result = await tools["search_knowledge"](ctx, "test query")

    assert "--- Section 1 (Source: doc1.txt, Relevance: 0.95) ---" in result
    assert "First result text" in result
    assert "--- Section 2 (Source: doc2.pdf, Relevance: 0.85) ---" in result
    assert "Second result text" in result
    assert 'Search results for: "test query"' in result
    assert "Found 2 relevant section(s):" in result
    mock_knowledge_store.search_scoped.assert_called_once_with(
        query="test query",
        top_k=10,
        document_ids={"doc-1", "doc-2"},
    )


@pytest.mark.asyncio
async def test_search_knowledge_no_results(
    chat_service,
    mock_knowledge_store,
    mock_web_search_tool,
):
    """Test search_knowledge returns message when no results found."""
    mock_knowledge_store.search_scoped.return_value = []

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(mock_knowledge_store, mock_web_search_tool)

    result = await tools["search_knowledge"](ctx, "unknown query")

    assert "" in result


@pytest.mark.asyncio
async def test_search_knowledge_missing_metadata(
    chat_service,
    mock_knowledge_store,
    mock_web_search_tool,
):
    """Test search_knowledge handles missing metadata gracefully."""
    mock_knowledge_store.search_scoped.return_value = [
        {"text": "Result without metadata"},
        {"text": "Result with empty metadata", "metadata": {}},
    ]

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(mock_knowledge_store, mock_web_search_tool)

    result = await tools["search_knowledge"](ctx, "query")

    assert "(Source: unknown, Relevance: 0.00)" in result
    assert "Result without metadata" in result
    assert "Result with empty metadata" in result


@pytest.mark.asyncio
async def test_search_knowledge_with_none_document_scope(
    chat_service,
    mock_knowledge_store,
    mock_web_search_tool,
):
    """Test search_knowledge passes None document scope correctly."""
    mock_knowledge_store.search_scoped.return_value = [
        {"text": "Global result", "metadata": {"filename": "global.txt"}},
    ]

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(
        mock_knowledge_store,
        mock_web_search_tool,
        document_scope=None,
    )

    result = await tools["search_knowledge"](ctx, "global search")

    mock_knowledge_store.search_scoped.assert_called_once_with(
        query="global search",
        top_k=10,
        document_ids=None,
    )
    assert "Global result" in result


# ============================================================================
# Test search_web tool execution
# ============================================================================


class FakeWebSearchResult:
    def __init__(self, success=True, data=""):
        self.success = success
        self.data = data
        self.error = None


@pytest.mark.asyncio
async def test_search_web_returns_results(
    chat_service,
    mock_knowledge_store,
    mock_web_search_tool,
):
    formatted = (
        'Web search results for: "latest news"\n\n'
        "1. Result 1\nSnippet 1\nhttps://example.com/1\n\n"
        "2. Result 2\nSnippet 2\nhttps://example.com/2"
    )

    mock_web_search_tool.execute.return_value = FakeWebSearchResult(
        success=True,
        data=formatted,
    )

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(mock_knowledge_store, mock_web_search_tool)

    result = await tools["search_web"](ctx, "latest news")

    assert 'Web search results for: "latest news"' in result
    assert "1. Result 1" in result
    assert "Snippet 1" in result
    assert "https://example.com/1" in result
    assert "2. Result 2" in result
    mock_web_search_tool.execute.assert_called_once_with(query="latest news")


@pytest.mark.asyncio
async def test_search_web_no_results(
    chat_service,
    mock_knowledge_store,
    mock_web_search_tool,
):
    """Test search_web returns message when no results found."""
    mock_web_search_tool.execute.return_value = []

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(mock_knowledge_store, mock_web_search_tool)

    result = await tools["search_web"](ctx, "obscure query")

    assert "" in result


@pytest.mark.asyncio
async def test_search_web_success_false(
    chat_service,
    mock_knowledge_store,
    mock_web_search_tool,
):
    """Test search_web returns message when no results found."""

    mock_web_search_tool.execute.return_value = FakeWebSearchResult(
        success=False,
    )

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(mock_knowledge_store, mock_web_search_tool)

    result = await tools["search_web"](ctx, "obscure query")

    assert "" in result


@pytest.mark.asyncio
async def test_search_web_empty_data(
    chat_service,
    mock_knowledge_store,
    mock_web_search_tool,
):
    """Test search_web returns message when no results found."""

    mock_web_search_tool.execute.return_value = FakeWebSearchResult(
        success=True, data=None
    )

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(mock_knowledge_store, mock_web_search_tool)

    result = await tools["search_web"](ctx, "obscure query")

    assert "" in result


@pytest.mark.asyncio
async def test_search_web_no_tool_available(
    chat_service,
    mock_knowledge_store,
):
    """Test search_web returns message when no web search tool available."""
    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(mock_knowledge_store, web_search_tool=None)

    result = await tools["search_web"](ctx, "search query")

    assert "Web search is not available" in result


@pytest.mark.asyncio
async def test_search_web_handles_exception(
    chat_service,
    mock_knowledge_store,
    mock_web_search_tool,
):
    """Test search_web handles exceptions gracefully."""
    mock_web_search_tool.execute.side_effect = Exception("Network error")

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(mock_knowledge_store, mock_web_search_tool)

    result = await tools["search_web"](ctx, "failing query")

    assert "" in result


@pytest.mark.asyncio
async def test_search_web_limits_results_to_five(
    chat_service,
    mock_knowledge_store,
    mock_web_search_tool,
):
    formatted = "\n".join(f"{i+1}. Result {i}" for i in range(5))

    mock_web_search_tool.execute.return_value = FakeWebSearchResult(
        success=True,
        data=formatted,
    )

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(mock_knowledge_store, mock_web_search_tool)

    result = await tools["search_web"](ctx, "many results")

    assert "5. Result 4" in result
    assert "6. Result 5" not in result
