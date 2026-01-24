import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.practorflow.common.fixtures import mock_knowledge_store, mock_llm_config
from practorflow.llm.pyai.tools import (
    KnowledgeDeps,
    search_knowledge,
    search_knowledge_generic,
    format_search_results,
    register_knowledge_tools,
)


@pytest.mark.asyncio
async def test_search_knowledge_success(mock_knowledge_store):
    deps = KnowledgeDeps(knowledge_store=mock_knowledge_store, document_scope=None)
    ctx = SimpleNamespace(deps=deps)

    mock_knowledge_store.search_scoped.return_value = [
        {"text": "Section 1", "metadata": {"filename": "a.txt"}, "similarity": 0.9},
        {"text": "Section 2", "metadata": {"filename": "b.txt"}, "similarity": 0.8},
    ]

    result = await search_knowledge(ctx, "example query", 5)

    assert 'Search results for: "example query"' in result
    assert "Found 2 relevant section(s):" in result
    assert "Section 1" in result
    assert "Section 2" in result


@pytest.mark.asyncio
async def test_search_knowledge_empty_results(mock_knowledge_store):
    deps = KnowledgeDeps(knowledge_store=mock_knowledge_store, document_scope=None)
    ctx = SimpleNamespace(deps=deps)

    mock_knowledge_store.search_scoped.return_value = []

    result = await search_knowledge(ctx, "no results")

    assert result == ""


@pytest.mark.asyncio
async def test_search_knowledge_with_document_scope(mock_knowledge_store):
    deps = KnowledgeDeps(
        knowledge_store=mock_knowledge_store,
        document_scope={"doc-1", "doc-2"},
    )
    ctx = SimpleNamespace(deps=deps)

    mock_knowledge_store.search_scoped.return_value = [
        {"text": "Scoped text", "metadata": {}, "similarity": 0.5}
    ]

    result = await search_knowledge(ctx, "scoped query", 1)

    mock_knowledge_store.search_scoped.assert_called_once()
    assert "Scoped text" in result


@pytest.mark.asyncio
async def test_search_knowledge_exception(mock_knowledge_store):
    deps = KnowledgeDeps(knowledge_store=mock_knowledge_store, document_scope=None)
    ctx = SimpleNamespace(deps=deps)

    mock_knowledge_store.search_scoped.side_effect = RuntimeError("boom")

    result = await search_knowledge(ctx, "query")

    assert result.startswith("Search failed:")


def test_format_search_results_with_missing_fields():
    results = [
        {"text": "Text only"},
        {"metadata": {"filename": "file.txt"}, "similarity": 0.42},
    ]

    formatted = format_search_results(results, "query")

    assert 'Search results for: "query"' in formatted
    assert "Section 1" in formatted
    assert "Section 2" in formatted
    assert "unknown" in formatted
    assert "0.00" in formatted or "0.42" in formatted


@pytest.mark.asyncio
async def test_search_knowledge_generic_success(mock_knowledge_store):
    deps = SimpleNamespace(
        knowledge_store=mock_knowledge_store,
        document_scope=None,
    )
    ctx = SimpleNamespace(deps=deps)

    mock_knowledge_store.search_scoped.return_value = [
        {"text": "Generic text", "metadata": {}, "similarity": 1.0}
    ]

    result = await search_knowledge_generic(ctx, "generic query", 3)

    assert "Generic text" in result


@pytest.mark.asyncio
async def test_search_knowledge_generic_missing_knowledge_store():
    deps = SimpleNamespace()
    ctx = SimpleNamespace(deps=deps)

    result = await search_knowledge_generic(ctx, "query")

    assert result == "Error: Dependencies do not include knowledge_store."


@pytest.mark.asyncio
async def test_search_knowledge_generic_exception(mock_knowledge_store):
    deps = SimpleNamespace(knowledge_store=mock_knowledge_store)
    ctx = SimpleNamespace(deps=deps)

    mock_knowledge_store.search_scoped.side_effect = Exception("fail")

    result = await search_knowledge_generic(ctx, "query")

    assert result.startswith("Search failed:")


def test_register_knowledge_tools_default():
    agent = MagicMock()

    returned = register_knowledge_tools(agent)

    agent.tool.assert_called_once()
    assert returned is agent


def test_register_knowledge_tools_generic():
    agent = MagicMock()

    returned = register_knowledge_tools(agent, use_generic=True)

    agent.tool.assert_called_once()
    assert returned is agent
