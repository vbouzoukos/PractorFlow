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
        {"text": "First result text", "metadata": {"filename": "doc1.txt"}},
        {"text": "Second result text", "metadata": {"filename": "doc2.pdf"}},
    ]

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(
        mock_knowledge_store,
        mock_web_search_tool,
        document_scope={"doc-1", "doc-2"},
    )

    result = await tools["search_knowledge"](ctx, "test query")

    assert "[Source: doc1.txt]" in result
    assert "First result text" in result
    assert "[Source: doc2.pdf]" in result
    assert "Second result text" in result
    mock_knowledge_store.search_scoped.assert_called_once_with(
        query="test query",
        top_k=5,
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

    assert "No relevant information found" in result


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

    assert "[Source: Unknown]" in result
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
        top_k=5,
        document_ids=None,
    )
    assert "Global result" in result


# ============================================================================
# Test search_web tool execution
# ============================================================================


@pytest.mark.asyncio
async def test_search_web_returns_results(
    chat_service,
    mock_knowledge_store,
    mock_web_search_tool,
):
    """Test search_web returns results from web search tool."""
    mock_web_search_tool.search.return_value = "Web search results for query"

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(mock_knowledge_store, mock_web_search_tool)

    result = await tools["search_web"](ctx, "latest news")

    assert result == "Web search results for query"
    mock_web_search_tool.search.assert_called_once_with("latest news")


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
    mock_web_search_tool.search.side_effect = Exception("Network error")

    tools = _capture_tools(chat_service)
    ctx = _create_mock_run_context(mock_knowledge_store, mock_web_search_tool)

    result = await tools["search_web"](ctx, "failing query")

    assert "Web search failed" in result
    assert "Network error" in result