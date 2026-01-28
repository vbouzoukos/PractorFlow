"""
Unit tests for DuckDuckGoSearchTool.

Tests cover all DuckDuckGoSearchTool methods for 100% code coverage.
"""

import pytest
from unittest.mock import MagicMock, patch

from practorflow.llm.tools.base_web_search import DuckDuckGoSearchTool


class TestDuckDuckGoSearchToolInit:
    """Tests for DuckDuckGoSearchTool.__init__"""

    def test_init_default_values(self):
        """__init__ sets default values correctly."""
        tool = DuckDuckGoSearchTool()

        assert tool._default_max_results == 5
        assert tool._region == "wt-wt"
        assert tool._safesearch == "moderate"
        assert tool._ddgs is None

    def test_init_custom_max_results(self):
        """__init__ accepts custom default_max_results."""
        tool = DuckDuckGoSearchTool(default_max_results=10)

        assert tool._default_max_results == 10

    def test_init_custom_region(self):
        """__init__ accepts custom region."""
        tool = DuckDuckGoSearchTool(region="us-en")

        assert tool._region == "us-en"

    def test_init_custom_safesearch(self):
        """__init__ accepts custom safesearch setting."""
        tool = DuckDuckGoSearchTool(safesearch="off")

        assert tool._safesearch == "off"


class TestDuckDuckGoSearchToolProperties:
    """Tests for DuckDuckGoSearchTool properties."""

    def test_name_returns_web_search(self):
        """name property returns 'web_search'."""
        tool = DuckDuckGoSearchTool()

        assert tool.name == "web_search"

    def test_description_is_non_empty(self):
        """description property returns non-empty string."""
        tool = DuckDuckGoSearchTool()

        assert len(tool.description) > 0
        assert "web" in tool.description.lower()

    def test_parameters_contains_query(self):
        """parameters includes required query parameter."""
        tool = DuckDuckGoSearchTool()

        params = tool.parameters
        query_param = next((p for p in params if p.name == "query"), None)

        assert query_param is not None
        assert query_param.type == "string"
        assert query_param.required is True

    def test_parameters_contains_max_results(self):
        """parameters includes optional max_results parameter."""
        tool = DuckDuckGoSearchTool()

        params = tool.parameters
        max_results_param = next((p for p in params if p.name == "max_results"), None)

        assert max_results_param is not None
        assert max_results_param.type == "integer"
        assert max_results_param.required is False

    def test_parameters_max_results_default_matches_init(self):
        """max_results parameter default matches default_max_results from init."""
        tool = DuckDuckGoSearchTool(default_max_results=8)

        params = tool.parameters
        max_results_param = next((p for p in params if p.name == "max_results"), None)

        assert max_results_param.default == 8


class TestDuckDuckGoSearchToolGetClient:
    """Tests for DuckDuckGoSearchTool._get_client()"""

    def test_get_client_lazy_initialization(self):
        """_get_client initializes client on first call."""
        tool = DuckDuckGoSearchTool()

        mock_ddgs_instance = MagicMock()
        mock_ddgs_class = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_module = MagicMock()
        mock_ddgs_module.DDGS = mock_ddgs_class

        with patch.dict("sys.modules", {"ddgs": mock_ddgs_module}):
            client = tool._get_client()

            assert client is mock_ddgs_instance
            mock_ddgs_class.assert_called_once()

    def test_get_client_reuses_instance(self):
        """_get_client returns same instance on subsequent calls."""
        tool = DuckDuckGoSearchTool()

        mock_ddgs_instance = MagicMock()
        mock_ddgs_class = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_module = MagicMock()
        mock_ddgs_module.DDGS = mock_ddgs_class

        with patch.dict("sys.modules", {"ddgs": mock_ddgs_module}):
            client1 = tool._get_client()
            client2 = tool._get_client()

            assert client1 is client2
            mock_ddgs_class.assert_called_once()

    def test_get_client_import_error(self):
        """_get_client raises ImportError when ddgs not installed."""
        tool = DuckDuckGoSearchTool()

        with patch.dict("sys.modules", {"ddgs": None}):
            with pytest.raises(ImportError, match="ddgs is required"):
                tool._get_client()


class TestDuckDuckGoSearchToolExecute:
    """Tests for DuckDuckGoSearchTool.execute()"""

    async def test_execute_missing_query_returns_error(self):
        """execute() returns error when query is missing."""
        tool = DuckDuckGoSearchTool()

        result = await tool.execute()

        assert result.success is False
        assert "Query parameter is required" in result.error

    async def test_execute_empty_query_returns_error(self):
        """execute() returns error when query is empty string."""
        tool = DuckDuckGoSearchTool()

        result = await tool.execute(query="")

        assert result.success is False
        assert "Query parameter is required" in result.error

    async def test_execute_no_results_returns_success_with_none_data(self):
        """execute() returns success with None data when no results found."""
        tool = DuckDuckGoSearchTool()

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.text.return_value = []
            mock_get_client.return_value = mock_client

            result = await tool.execute(query="nonexistent query")

            assert result.success is True
            assert result.data is None
            assert result.metadata["results_count"] == 0
            assert result.metadata["provider"] == "duckduckgo"

    async def test_execute_with_results_returns_formatted_data(self):
        """execute() returns formatted results when search succeeds."""
        tool = DuckDuckGoSearchTool()

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.text.return_value = [
                {"title": "Result 1", "href": "http://example.com/1", "body": "First result"},
                {"title": "Result 2", "href": "http://example.com/2", "body": "Second result"},
            ]
            mock_get_client.return_value = mock_client

            result = await tool.execute(query="test query")

            assert result.success is True
            assert "Result 1" in result.data
            assert "Result 2" in result.data
            assert result.metadata["results_count"] == 2
            assert result.metadata["query"] == "test query"

    async def test_execute_uses_default_max_results(self):
        """execute() uses default_max_results when not provided."""
        tool = DuckDuckGoSearchTool(default_max_results=7)

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.text.return_value = []
            mock_get_client.return_value = mock_client

            await tool.execute(query="test")

            mock_client.text.assert_called_once_with(
                "test",
                region="wt-wt",
                safesearch="moderate",
                max_results=7,
            )

    async def test_execute_uses_provided_max_results(self):
        """execute() uses provided max_results over default."""
        tool = DuckDuckGoSearchTool(default_max_results=5)

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.text.return_value = []
            mock_get_client.return_value = mock_client

            await tool.execute(query="test", max_results=3)

            mock_client.text.assert_called_once_with(
                "test",
                region="wt-wt",
                safesearch="moderate",
                max_results=3,
            )

    async def test_execute_passes_region_and_safesearch(self):
        """execute() passes region and safesearch to client."""
        tool = DuckDuckGoSearchTool(region="uk-en", safesearch="off")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.text.return_value = []
            mock_get_client.return_value = mock_client

            await tool.execute(query="test")

            mock_client.text.assert_called_once_with(
                "test",
                region="uk-en",
                safesearch="off",
                max_results=5,
            )

    async def test_execute_exception_returns_error(self):
        """execute() returns error result when exception occurs."""
        tool = DuckDuckGoSearchTool()

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.text.side_effect = RuntimeError("Network error")
            mock_get_client.return_value = mock_client

            result = await tool.execute(query="test")

            assert result.success is False
            assert "Web search failed" in result.error
            assert "Network error" in result.error

class TestDuckDuckGoSearchToolFormatResults:
    """Tests for DuckDuckGoSearchTool._format_results()"""

    def test_format_results_includes_result_numbers(self):
        """_format_results includes result numbers in output."""
        tool = DuckDuckGoSearchTool()
        results = [
            {"title": "First", "url": "http://a.com", "snippet": "A"},
            {"title": "Second", "url": "http://b.com", "snippet": "B"},
        ]

        formatted = tool._format_results(results)

        assert "Result 1" in formatted
        assert "Result 2" in formatted

    def test_format_results_includes_title(self):
        """_format_results includes title in output."""
        tool = DuckDuckGoSearchTool()
        results = [{"title": "Test Title", "url": "http://test.com", "snippet": "Content"}]

        formatted = tool._format_results(results)

        assert "Test Title" in formatted

    def test_format_results_includes_url(self):
        """_format_results includes URL in output."""
        tool = DuckDuckGoSearchTool()
        results = [{"title": "Test", "url": "http://example.org/page", "snippet": "Content"}]

        formatted = tool._format_results(results)

        assert "http://example.org/page" in formatted

    def test_format_results_includes_snippet(self):
        """_format_results includes snippet in output."""
        tool = DuckDuckGoSearchTool()
        results = [{"title": "Test", "url": "http://test.com", "snippet": "This is the snippet text."}]

        formatted = tool._format_results(results)

        assert "This is the snippet text." in formatted

    def test_format_results_handles_missing_title(self):
        """_format_results uses 'No title' when title missing."""
        tool = DuckDuckGoSearchTool()
        results = [{"url": "http://test.com", "snippet": "Content"}]

        formatted = tool._format_results(results)

        assert "No title" in formatted

    def test_format_results_separates_results(self):
        """_format_results separates results with blank lines."""
        tool = DuckDuckGoSearchTool()
        results = [
            {"title": "First", "url": "http://a.com", "snippet": "A"},
            {"title": "Second", "url": "http://b.com", "snippet": "B"},
        ]

        formatted = tool._format_results(results)

        assert "\n\n" in formatted


class TestDuckDuckGoSearchToolFormatNewsResults:
    """Tests for DuckDuckGoSearchTool._format_news_results()"""

    def test_format_news_results_includes_news_numbers(self):
        """_format_news_results includes news numbers in output."""
        tool = DuckDuckGoSearchTool()
        results = [
            {"title": "News A", "url": "http://a.com", "snippet": "A", "date": "", "source": ""},
            {"title": "News B", "url": "http://b.com", "snippet": "B", "date": "", "source": ""},
        ]

        formatted = tool._format_news_results(results)

        assert "News 1" in formatted
        assert "News 2" in formatted

    def test_format_news_results_includes_source_and_date(self):
        """_format_news_results includes source and date when present."""
        tool = DuckDuckGoSearchTool()
        results = [
            {
                "title": "Breaking",
                "url": "http://news.com",
                "snippet": "Content",
                "date": "2024-01-15",
                "source": "CNN",
            }
        ]

        formatted = tool._format_news_results(results)

        assert "CNN" in formatted
        assert "2024-01-15" in formatted

    def test_format_news_results_omits_meta_when_empty(self):
        """_format_news_results omits meta line when source and date are empty."""
        tool = DuckDuckGoSearchTool()
        results = [
            {"title": "News", "url": "http://news.com", "snippet": "Content", "date": "", "source": ""}
        ]

        formatted = tool._format_news_results(results)

        assert "Source:" not in formatted
        assert "Date:" not in formatted


class TestDuckDuckGoSearchToolCallable:
    """Tests for DuckDuckGoSearchTool.__call__ (inherited from AsyncBaseTool)."""

    async def test_tool_is_callable(self):
        """Tool can be called directly via __call__."""
        tool = DuckDuckGoSearchTool()

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.text.return_value = []
            mock_get_client.return_value = mock_client

            result = await tool(query="direct call")

            assert result.success is True


class TestDuckDuckGoSearchToolSchema:
    """Tests for DuckDuckGoSearchTool.get_schema() (inherited from AsyncBaseTool)."""

    def test_get_schema_returns_function_format(self):
        """get_schema returns OpenAI function calling format."""
        tool = DuckDuckGoSearchTool()

        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "web_search"
        assert "parameters" in schema["function"]

    def test_get_schema_includes_query_as_required(self):
        """get_schema marks query as required parameter."""
        tool = DuckDuckGoSearchTool()

        schema = tool.get_schema()

        assert "query" in schema["function"]["parameters"]["required"]