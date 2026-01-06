"""
Unit tests for KnowledgeSearchTool.

Tests cover all KnowledgeSearchTool methods for 100% code coverage.
"""

import pytest
from unittest.mock import MagicMock

from practorflow.llm.tools.knowledge_search import KnowledgeSearchTool
from tests.practorflow.common.fixtures import mock_knowledge_store


class TestKnowledgeSearchToolInit:
    """Tests for KnowledgeSearchTool.__init__"""

    def test_init_stores_knowledge_store(self, mock_knowledge_store):
        """__init__ stores knowledge_store reference."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        assert tool._knowledge_store is mock_knowledge_store

    def test_init_default_top_k(self, mock_knowledge_store):
        """__init__ uses default top_k of 5."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        assert tool._default_top_k == 5

    def test_init_custom_top_k(self, mock_knowledge_store):
        """__init__ accepts custom default_top_k."""
        tool = KnowledgeSearchTool(
            knowledge_store=mock_knowledge_store,
            default_top_k=10,
        )

        assert tool._default_top_k == 10

    def test_init_document_scope_is_none(self, mock_knowledge_store):
        """__init__ sets document_scope to None."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        assert tool._document_scope is None


class TestKnowledgeSearchToolProperties:
    """Tests for KnowledgeSearchTool properties."""

    def test_name_returns_knowledge_search(self, mock_knowledge_store):
        """name property returns 'knowledge_search'."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        assert tool.name == "knowledge_search"

    def test_description_is_non_empty(self, mock_knowledge_store):
        """description property returns non-empty string."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        assert len(tool.description) > 0
        assert "knowledge base" in tool.description.lower()

    def test_parameters_contains_query(self, mock_knowledge_store):
        """parameters includes required query parameter."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        params = tool.parameters
        query_param = next((p for p in params if p.name == "query"), None)

        assert query_param is not None
        assert query_param.type == "string"
        assert query_param.required is True

    def test_parameters_contains_top_k(self, mock_knowledge_store):
        """parameters includes optional top_k parameter."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        params = tool.parameters
        top_k_param = next((p for p in params if p.name == "top_k"), None)

        assert top_k_param is not None
        assert top_k_param.type == "integer"
        assert top_k_param.required is False

    def test_parameters_top_k_default_matches_init(self, mock_knowledge_store):
        """top_k parameter default matches default_top_k from init."""
        tool = KnowledgeSearchTool(
            knowledge_store=mock_knowledge_store,
            default_top_k=8,
        )

        params = tool.parameters
        top_k_param = next((p for p in params if p.name == "top_k"), None)

        assert top_k_param.default == 8


class TestKnowledgeSearchToolSetDocumentScope:
    """Tests for KnowledgeSearchTool.set_document_scope()"""

    def test_set_document_scope_with_set(self, mock_knowledge_store):
        """set_document_scope stores document IDs."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        tool.set_document_scope({"doc1", "doc2"})

        assert tool._document_scope == {"doc1", "doc2"}

    def test_set_document_scope_with_none(self, mock_knowledge_store):
        """set_document_scope with None clears scope."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)
        tool.set_document_scope({"doc1"})

        tool.set_document_scope(None)

        assert tool._document_scope is None


class TestKnowledgeSearchToolExecute:
    """Tests for KnowledgeSearchTool.execute()"""

    def test_execute_missing_query_returns_error(self, mock_knowledge_store):
        """execute() returns error when query is missing."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        result = tool.execute()

        assert result.success is False
        assert "Query parameter is required" in result.error

    def test_execute_empty_query_returns_error(self, mock_knowledge_store):
        """execute() returns error when query is empty string."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        result = tool.execute(query="")

        assert result.success is False
        assert "Query parameter is required" in result.error

    def test_execute_no_results_returns_success_with_none_data(
        self, mock_knowledge_store
    ):
        """execute() returns success with None data when no results found."""
        mock_knowledge_store.search_scoped.return_value = []
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        result = tool.execute(query="test query")

        assert result.success is True
        assert result.data is None
        assert result.metadata["results_count"] == 0
        assert result.metadata["query"] == "test query"

    def test_execute_with_results_returns_formatted_data(self, mock_knowledge_store):
        """execute() returns formatted results when search succeeds."""
        mock_knowledge_store.search_scoped.return_value = [
            {
                "text": "First result text",
                "metadata": {"filename": "doc1.txt"},
                "similarity": 0.95,
                "document_id": "doc-1",
            },
            {
                "text": "Second result text",
                "metadata": {"filename": "doc2.pdf"},
                "similarity": 0.85,
                "document_id": "doc-2",
            },
        ]
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        result = tool.execute(query="search term")

        assert result.success is True
        assert "First result text" in result.data
        assert "Second result text" in result.data
        assert result.metadata["results_count"] == 2
        assert result.metadata["query"] == "search term"

    def test_execute_uses_default_top_k(self, mock_knowledge_store):
        """execute() uses default_top_k when not provided."""
        mock_knowledge_store.search_scoped.return_value = []
        tool = KnowledgeSearchTool(
            knowledge_store=mock_knowledge_store,
            default_top_k=7,
        )

        tool.execute(query="test")

        mock_knowledge_store.search_scoped.assert_called_once_with(
            query="test",
            top_k=7,
            document_ids=None,
        )

    def test_execute_uses_provided_top_k(self, mock_knowledge_store):
        """execute() uses provided top_k over default."""
        mock_knowledge_store.search_scoped.return_value = []
        tool = KnowledgeSearchTool(
            knowledge_store=mock_knowledge_store,
            default_top_k=5,
        )

        tool.execute(query="test", top_k=3)

        mock_knowledge_store.search_scoped.assert_called_once_with(
            query="test",
            top_k=3,
            document_ids=None,
        )

    def test_execute_passes_document_scope(self, mock_knowledge_store):
        """execute() passes document scope to search_scoped."""
        mock_knowledge_store.search_scoped.return_value = []
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)
        tool.set_document_scope({"doc-a", "doc-b"})

        tool.execute(query="test")

        mock_knowledge_store.search_scoped.assert_called_once_with(
            query="test",
            top_k=5,
            document_ids={"doc-a", "doc-b"},
        )

    def test_execute_exception_returns_error(self, mock_knowledge_store):
        """execute() returns error result when exception occurs."""
        mock_knowledge_store.search_scoped.side_effect = RuntimeError("DB error")
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        result = tool.execute(query="test")

        assert result.success is False
        assert "Search failed" in result.error
        assert "DB error" in result.error

    def test_execute_metadata_contains_document_ids(self, mock_knowledge_store):
        """execute() metadata includes unique document IDs from results."""
        mock_knowledge_store.search_scoped.return_value = [
            {"text": "text1", "metadata": {}, "document_id": "doc-1"},
            {"text": "text2", "metadata": {}, "document_id": "doc-2"},
            {"text": "text3", "metadata": {}, "document_id": "doc-1"},
        ]
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        result = tool.execute(query="test")

        assert set(result.metadata["document_ids"]) == {"doc-1", "doc-2"}

    def test_execute_handles_missing_document_id(self, mock_knowledge_store):
        """execute() handles results without document_id gracefully."""
        mock_knowledge_store.search_scoped.return_value = [
            {"text": "text1", "metadata": {}},
            {"text": "text2", "metadata": {}, "document_id": "doc-1"},
        ]
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        result = tool.execute(query="test")

        assert result.success is True
        assert result.metadata["document_ids"] == ["doc-1"]


class TestKnowledgeSearchToolFormatResults:
    """Tests for KnowledgeSearchTool._format_results()"""

    def test_format_results_includes_section_numbers(self, mock_knowledge_store):
        """_format_results includes section numbers in output."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)
        results = [
            {"text": "First", "metadata": {"filename": "a.txt"}, "similarity": 0.9},
            {"text": "Second", "metadata": {"filename": "b.txt"}, "similarity": 0.8},
        ]

        formatted = tool._format_results(results)

        assert "Section 1" in formatted
        assert "Section 2" in formatted

    def test_format_results_includes_filename(self, mock_knowledge_store):
        """_format_results includes source filename."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)
        results = [
            {"text": "Content", "metadata": {"filename": "report.pdf"}, "similarity": 0.9},
        ]

        formatted = tool._format_results(results)

        assert "report.pdf" in formatted

    def test_format_results_includes_similarity(self, mock_knowledge_store):
        """_format_results includes relevance score."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)
        results = [
            {"text": "Content", "metadata": {"filename": "doc.txt"}, "similarity": 0.87},
        ]

        formatted = tool._format_results(results)

        assert "0.87" in formatted

    def test_format_results_includes_text(self, mock_knowledge_store):
        """_format_results includes result text content."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)
        results = [
            {
                "text": "This is the actual content from the document.",
                "metadata": {"filename": "doc.txt"},
                "similarity": 0.9,
            },
        ]

        formatted = tool._format_results(results)

        assert "This is the actual content from the document." in formatted

    def test_format_results_handles_missing_filename(self, mock_knowledge_store):
        """_format_results uses 'unknown' when filename missing."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)
        results = [
            {"text": "Content", "metadata": {}, "similarity": 0.9},
        ]

        formatted = tool._format_results(results)

        assert "unknown" in formatted

    def test_format_results_handles_missing_similarity(self, mock_knowledge_store):
        """_format_results handles missing similarity with default 0.0."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)
        results = [
            {"text": "Content", "metadata": {"filename": "doc.txt"}},
        ]

        formatted = tool._format_results(results)

        assert "0.00" in formatted

    def test_format_results_handles_empty_text(self, mock_knowledge_store):
        """_format_results handles missing text gracefully."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)
        results = [
            {"metadata": {"filename": "doc.txt"}, "similarity": 0.9},
        ]

        formatted = tool._format_results(results)

        assert "Section 1" in formatted
        assert "doc.txt" in formatted

    def test_format_results_separates_sections(self, mock_knowledge_store):
        """_format_results separates sections with blank lines."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)
        results = [
            {"text": "First", "metadata": {"filename": "a.txt"}, "similarity": 0.9},
            {"text": "Second", "metadata": {"filename": "b.txt"}, "similarity": 0.8},
        ]

        formatted = tool._format_results(results)

        assert "\n\n" in formatted


class TestKnowledgeSearchToolCallable:
    """Tests for KnowledgeSearchTool.__call__ (inherited from BaseTool)."""

    def test_tool_is_callable(self, mock_knowledge_store):
        """Tool can be called directly via __call__."""
        mock_knowledge_store.search_scoped.return_value = []
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        result = tool(query="direct call")

        assert result.success is True
        mock_knowledge_store.search_scoped.assert_called_once()


class TestKnowledgeSearchToolSchema:
    """Tests for KnowledgeSearchTool.get_schema() (inherited from BaseTool)."""

    def test_get_schema_returns_function_format(self, mock_knowledge_store):
        """get_schema returns OpenAI function calling format."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "knowledge_search"
        assert "parameters" in schema["function"]

    def test_get_schema_includes_query_as_required(self, mock_knowledge_store):
        """get_schema marks query as required parameter."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        schema = tool.get_schema()

        assert "query" in schema["function"]["parameters"]["required"]

    def test_get_schema_includes_top_k_as_optional(self, mock_knowledge_store):
        """get_schema includes top_k but not as required."""
        tool = KnowledgeSearchTool(knowledge_store=mock_knowledge_store)

        schema = tool.get_schema()

        properties = schema["function"]["parameters"]["properties"]
        required = schema["function"]["parameters"]["required"]

        assert "top_k" in properties
        assert "top_k" not in required