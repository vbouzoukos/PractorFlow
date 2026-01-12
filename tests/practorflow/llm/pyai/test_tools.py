import pytest
from types import SimpleNamespace

from tests.practorflow.common.fixtures import mock_knowledge_store, mock_llm_config
from practorflow.llm.pyai.tools import KnowledgeDeps, search_knowledge


@pytest.mark.asyncio
async def test_search_knowledge(mock_knowledge_store, mock_llm_config):
    # Build deps exactly as expected
    deps = KnowledgeDeps(
        knowledge_store=mock_knowledge_store,
        document_scope=None,
    )

    # Minimal ctx mock with only what the tool uses
    ctx = SimpleNamespace(deps=deps)

    # Test case 1: Successful search with non-empty results
    mock_knowledge_store.search_scoped.return_value = [
        {"text": "Section 1", "metadata": {}, "similarity": 0.9},
        {"text": "Section 2", "metadata": {}, "similarity": 0.8},
        {"text": "Section 3", "metadata": {}, "similarity": 0.7},
    ]

    result = await search_knowledge(ctx, "example query", 5)

    assert 'Search results for: "example query"' in result
    assert "Found 3 relevant section(s):" in result
    assert "Section 1" in result
    assert "Section 2" in result
    assert "Section 3" in result

    # Test case 2: Empty results
    mock_knowledge_store.search_scoped.return_value = []

    result = await search_knowledge(ctx, "nonexistent query")

    assert result == "No relevant documents found for the query."
