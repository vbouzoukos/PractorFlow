"""
Unit tests for SerpAPISearchTool.

Tests cover all SerpAPISearchTool methods for 100% code coverage.
"""

import pytest
from unittest.mock import MagicMock, patch

from practorflow.llm.tools.serpapi_web_search import SerpAPISearchTool


class TestSerpAPISearchToolInit:
    """Tests for SerpAPISearchTool.__init__"""

    def test_init_stores_api_key(self):
        """__init__ stores API key."""
        tool = SerpAPISearchTool(api_key="test-key")

        assert tool._api_key == "test-key"

    def test_init_default_values(self):
        """__init__ sets default values correctly."""
        tool = SerpAPISearchTool(api_key="test-key")

        assert tool._default_max_results == 5
        assert tool._engine == "google"
        assert tool._country == "us"
        assert tool._language == "en"
        assert tool._safe_search is True
        assert tool._client is None

    def test_init_custom_max_results(self):
        """__init__ accepts custom default_max_results."""
        tool = SerpAPISearchTool(api_key="key", default_max_results=10)

        assert tool._default_max_results == 10

    def test_init_custom_engine(self):
        """__init__ accepts custom engine."""
        tool = SerpAPISearchTool(api_key="key", engine="bing")

        assert tool._engine == "bing"

    def test_init_custom_country(self):
        """__init__ accepts custom country."""
        tool = SerpAPISearchTool(api_key="key", country="uk")

        assert tool._country == "uk"

    def test_init_custom_language(self):
        """__init__ accepts custom language."""
        tool = SerpAPISearchTool(api_key="key", language="de")

        assert tool._language == "de"

    def test_init_custom_safe_search(self):
        """__init__ accepts custom safe_search."""
        tool = SerpAPISearchTool(api_key="key", safe_search=False)

        assert tool._safe_search is False


class TestSerpAPISearchToolProperties:
    """Tests for SerpAPISearchTool properties."""

    def test_name_returns_web_search(self):
        """name property returns 'web_search'."""
        tool = SerpAPISearchTool(api_key="key")

        assert tool.name == "web_search"

    def test_description_is_non_empty(self):
        """description property returns non-empty string."""
        tool = SerpAPISearchTool(api_key="key")

        assert len(tool.description) > 0
        assert "web" in tool.description.lower()

    def test_parameters_contains_query(self):
        """parameters includes required query parameter."""
        tool = SerpAPISearchTool(api_key="key")

        params = tool.parameters
        query_param = next((p for p in params if p.name == "query"), None)

        assert query_param is not None
        assert query_param.type == "string"
        assert query_param.required is True

    def test_parameters_contains_max_results(self):
        """parameters includes optional max_results parameter."""
        tool = SerpAPISearchTool(api_key="key")

        params = tool.parameters
        max_results_param = next((p for p in params if p.name == "max_results"), None)

        assert max_results_param is not None
        assert max_results_param.type == "integer"
        assert max_results_param.required is False


class TestSerpAPISearchToolGetClient:
    """Tests for SerpAPISearchTool._get_client()"""

    def test_get_client_lazy_initialization(self):
        """_get_client initializes client on first call."""
        tool = SerpAPISearchTool(api_key="key")

        mock_client_class = MagicMock()
        mock_serpapi_module = MagicMock()
        mock_serpapi_module.Client = mock_client_class

        with patch.dict("sys.modules", {"serpapi": mock_serpapi_module}):
            client = tool._get_client()

            assert client is mock_client_class

    def test_get_client_reuses_instance(self):
        """_get_client returns same class on subsequent calls."""
        tool = SerpAPISearchTool(api_key="key")

        mock_client_class = MagicMock()
        mock_serpapi_module = MagicMock()
        mock_serpapi_module.Client = mock_client_class

        with patch.dict("sys.modules", {"serpapi": mock_serpapi_module}):
            client1 = tool._get_client()
            client2 = tool._get_client()

            assert client1 is client2

    def test_get_client_import_error(self):
        """_get_client raises ImportError when serpapi not installed."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.dict("sys.modules", {"serpapi": None}):
            with pytest.raises(ImportError, match="serpapi is required"):
                tool._get_client()


class TestSerpAPISearchToolBuildParams:
    """Tests for SerpAPISearchTool._build_params()"""

    def test_build_params_google(self):
        """_build_params returns correct params for google engine."""
        tool = SerpAPISearchTool(
            api_key="key",
            engine="google",
            country="us",
            language="en",
            safe_search=True,
        )

        params = tool._build_params("test query", 5)

        assert params["q"] == "test query"
        assert params["num"] == 5
        assert params["engine"] == "google"
        assert params["gl"] == "us"
        assert params["hl"] == "en"
        assert params["safe"] == "active"

    def test_build_params_google_safe_search_off(self):
        """_build_params sets safe=off when safe_search is False."""
        tool = SerpAPISearchTool(api_key="key", engine="google", safe_search=False)

        params = tool._build_params("test", 5)

        assert params["safe"] == "off"

    def test_build_params_bing(self):
        """_build_params returns correct params for bing engine."""
        tool = SerpAPISearchTool(
            api_key="key",
            engine="bing",
            country="uk",
            language="en",
            safe_search=True,
        )

        params = tool._build_params("test", 5)

        assert params["engine"] == "bing"
        assert params["cc"] == "UK"
        assert params["setLang"] == "en"
        assert params["safeSearch"] == "Strict"

    def test_build_params_bing_safe_search_off(self):
        """_build_params sets safeSearch=Off for bing when safe_search is False."""
        tool = SerpAPISearchTool(api_key="key", engine="bing", safe_search=False)

        params = tool._build_params("test", 5)

        assert params["safeSearch"] == "Off"

    def test_build_params_yahoo(self):
        """_build_params returns correct params for yahoo engine."""
        tool = SerpAPISearchTool(api_key="key", engine="yahoo", language="es")

        params = tool._build_params("test", 5)

        assert params["engine"] == "yahoo"
        assert params["vl"] == "lang_es"

    def test_build_params_duckduckgo(self):
        """_build_params returns correct params for duckduckgo engine."""
        tool = SerpAPISearchTool(api_key="key", engine="duckduckgo", country="de", language="de")

        params = tool._build_params("test", 5)

        assert params["engine"] == "duckduckgo"
        assert params["kl"] == "de-de"

    def test_build_params_baidu(self):
        """_build_params returns correct params for baidu engine."""
        tool = SerpAPISearchTool(api_key="key", engine="baidu")

        params = tool._build_params("test", 5)

        assert params["engine"] == "baidu"

    def test_build_params_yandex(self):
        """_build_params returns correct params for yandex engine."""
        tool = SerpAPISearchTool(api_key="key", engine="yandex", language="ru")

        params = tool._build_params("test", 5)

        assert params["engine"] == "yandex"
        assert params["lang"] == "ru"

    def test_build_params_unknown_engine(self):
        """_build_params handles unknown engine by setting engine param."""
        tool = SerpAPISearchTool(api_key="key", engine="custom_engine")

        params = tool._build_params("test", 5)

        assert params["engine"] == "custom_engine"


class TestSerpAPISearchToolExtractResults:
    """Tests for SerpAPISearchTool._extract_results()"""

    def test_extract_results_organic(self):
        """_extract_results extracts organic results."""
        tool = SerpAPISearchTool(api_key="key")
        response = {
            "organic_results": [
                {
                    "title": "Result 1",
                    "link": "http://example.com/1",
                    "snippet": "First result",
                    "position": 1,
                    "displayed_link": "example.com",
                },
            ]
        }

        results = tool._extract_results(response, 5)

        assert len(results) == 1
        assert results[0]["title"] == "Result 1"
        assert results[0]["url"] == "http://example.com/1"
        assert results[0]["snippet"] == "First result"
        assert results[0]["position"] == 1

    def test_extract_results_limits_to_max_results(self):
        """_extract_results limits results to max_results."""
        tool = SerpAPISearchTool(api_key="key")
        response = {
            "organic_results": [
                {"title": f"Result {i}", "link": f"http://example.com/{i}", "snippet": "", "position": i}
                for i in range(10)
            ]
        }

        results = tool._extract_results(response, 3)

        assert len(results) == 3

    def test_extract_results_fallback_to_web_results(self):
        """_extract_results falls back to web_results when no organic_results."""
        tool = SerpAPISearchTool(api_key="key")
        response = {
            "web_results": [
                {
                    "title": "Web Result",
                    "url": "http://web.com",
                    "description": "Web description",
                    "position": 1,
                },
            ]
        }

        results = tool._extract_results(response, 5)

        assert len(results) == 1
        assert results[0]["title"] == "Web Result"
        assert results[0]["url"] == "http://web.com"
        assert results[0]["snippet"] == "Web description"

    def test_extract_results_handles_missing_fields(self):
        """_extract_results handles missing fields with defaults."""
        tool = SerpAPISearchTool(api_key="key")
        response = {"organic_results": [{}]}

        results = tool._extract_results(response, 5)

        assert len(results) == 1
        assert results[0]["title"] == ""
        assert results[0]["url"] == ""


class TestSerpAPISearchToolExecute:
    """Tests for SerpAPISearchTool.execute()"""

    def test_execute_missing_query_returns_error(self):
        """execute() returns error when query is missing."""
        tool = SerpAPISearchTool(api_key="key")

        result = tool.execute()

        assert result.success is False
        assert "Query parameter is required" in result.error

    def test_execute_empty_query_returns_error(self):
        """execute() returns error when query is empty string."""
        tool = SerpAPISearchTool(api_key="key")

        result = tool.execute(query="")

        assert result.success is False
        assert "Query parameter is required" in result.error

    def test_execute_api_error_in_response(self):
        """execute() returns error when API returns error in response."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client = MagicMock()
            mock_client.search.return_value = {"error": "Invalid API key"}
            mock_client_class.return_value = mock_client
            mock_get_client.return_value = mock_client_class

            result = tool.execute(query="test")

            assert result.success is False
            assert "SerpAPI error" in result.error

    def test_execute_no_results_returns_success_with_none_data(self):
        """execute() returns success with None data when no results found."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client = MagicMock()
            mock_client.search.return_value = {"organic_results": []}
            mock_client_class.return_value = mock_client
            mock_get_client.return_value = mock_client_class

            result = tool.execute(query="nonexistent")

            assert result.success is True
            assert result.data is None
            assert result.metadata["results_count"] == 0
            assert result.metadata["provider"] == "serpapi"

    def test_execute_with_results_returns_formatted_data(self):
        """execute() returns formatted results when search succeeds."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "organic_results": [
                    {"title": "Result 1", "link": "http://a.com", "snippet": "First", "position": 1},
                    {"title": "Result 2", "link": "http://b.com", "snippet": "Second", "position": 2},
                ],
                "search_metadata": {"id": "search123"},
                "search_information": {"total_results": 1000, "time_taken_displayed": "0.5s"},
            }
            mock_client_class.return_value = mock_client
            mock_get_client.return_value = mock_client_class

            result = tool.execute(query="test query")

            assert result.success is True
            assert "Result 1" in result.data
            assert "Result 2" in result.data
            assert result.metadata["results_count"] == 2
            assert result.metadata["search_id"] == "search123"

    def test_execute_uses_default_max_results(self):
        """execute() uses default_max_results when not provided."""
        tool = SerpAPISearchTool(api_key="key", default_max_results=7)

        with patch.object(tool, "_get_client") as mock_get_client:
            with patch.object(tool, "_build_params") as mock_build_params:
                mock_build_params.return_value = {"q": "test"}
                mock_client_class = MagicMock()
                mock_client = MagicMock()
                mock_client.search.return_value = {"organic_results": []}
                mock_client_class.return_value = mock_client
                mock_get_client.return_value = mock_client_class

                tool.execute(query="test")

                mock_build_params.assert_called_once_with("test", 7)

    def test_execute_exception_returns_error(self):
        """execute() returns error result when exception occurs."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_get_client.side_effect = RuntimeError("Connection error")

            result = tool.execute(query="test")

            assert result.success is False
            assert "Web search failed" in result.error


class TestSerpAPISearchToolSearchNews:
    """Tests for SerpAPISearchTool.search_news()"""

    def test_search_news_returns_formatted_results(self):
        """search_news() returns formatted news results."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "news_results": [
                    {
                        "title": "News 1",
                        "link": "http://news.com/1",
                        "snippet": "First news",
                        "date": "1 hour ago",
                        "source": {"name": "CNN"},
                    },
                ]
            }
            mock_client_class.return_value = mock_client
            mock_get_client.return_value = mock_client_class

            result = tool.search_news("breaking news")

            assert result.success is True
            assert "News 1" in result.data
            assert result.metadata["type"] == "news"
            assert result.metadata["provider"] == "serpapi"

    def test_search_news_api_error(self):
        """search_news() returns error when API returns error."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client = MagicMock()
            mock_client.search.return_value = {"error": "Rate limit exceeded"}
            mock_client_class.return_value = mock_client
            mock_get_client.return_value = mock_client_class

            result = tool.search_news("test")

            assert result.success is False
            assert "SerpAPI error" in result.error

    def test_search_news_no_results(self):
        """search_news() returns success with None data when no results."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client = MagicMock()
            mock_client.search.return_value = {"news_results": []}
            mock_client_class.return_value = mock_client
            mock_get_client.return_value = mock_client_class

            result = tool.search_news("nonexistent")

            assert result.success is True
            assert result.data is None
            assert result.metadata["results_count"] == 0

    def test_search_news_exception_returns_error(self):
        """search_news() returns error result when exception occurs."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_get_client.side_effect = RuntimeError("Network error")

            result = tool.search_news("test")

            assert result.success is False
            assert "News search failed" in result.error


class TestSerpAPISearchToolSearchImages:
    """Tests for SerpAPISearchTool.search_images()"""

    def test_search_images_returns_formatted_results(self):
        """search_images() returns formatted image results."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "images_results": [
                    {
                        "title": "Image 1",
                        "link": "http://site.com/page",
                        "original": "http://site.com/image.jpg",
                        "source": "Example Site",
                        "original_width": 800,
                        "original_height": 600,
                    },
                ]
            }
            mock_client_class.return_value = mock_client
            mock_get_client.return_value = mock_client_class

            result = tool.search_images("cats")

            assert result.success is True
            assert "Image 1" in result.data
            assert result.metadata["type"] == "images"

    def test_search_images_api_error(self):
        """search_images() returns error when API returns error."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client = MagicMock()
            mock_client.search.return_value = {"error": "Invalid request"}
            mock_client_class.return_value = mock_client
            mock_get_client.return_value = mock_client_class

            result = tool.search_images("test")

            assert result.success is False
            assert "SerpAPI error" in result.error

    def test_search_images_no_results(self):
        """search_images() returns success with None data when no results."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client = MagicMock()
            mock_client.search.return_value = {"images_results": []}
            mock_client_class.return_value = mock_client
            mock_get_client.return_value = mock_client_class

            result = tool.search_images("nonexistent")

            assert result.success is True
            assert result.data is None

    def test_search_images_exception_returns_error(self):
        """search_images() returns error result when exception occurs."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_get_client.side_effect = RuntimeError("Timeout")

            result = tool.search_images("test")

            assert result.success is False
            assert "Image search failed" in result.error


class TestSerpAPISearchToolFormatResults:
    """Tests for SerpAPISearchTool._format_results()"""

    def test_format_results_includes_position(self):
        """_format_results uses position in header."""
        tool = SerpAPISearchTool(api_key="key")
        results = [{"title": "Test", "url": "http://test.com", "snippet": "Content", "position": 3}]

        formatted = tool._format_results(results)

        assert "Result 3" in formatted

    def test_format_results_fallback_position(self):
        """_format_results uses index when position missing."""
        tool = SerpAPISearchTool(api_key="key")
        results = [{"title": "Test", "url": "http://test.com", "snippet": "Content"}]

        formatted = tool._format_results(results)

        assert "Result 1" in formatted

    def test_format_results_includes_url_and_snippet(self):
        """_format_results includes URL and snippet."""
        tool = SerpAPISearchTool(api_key="key")
        results = [{"title": "Test", "url": "http://example.org", "snippet": "This is content", "position": 1}]

        formatted = tool._format_results(results)

        assert "http://example.org" in formatted
        assert "This is content" in formatted


class TestSerpAPISearchToolFormatNewsResults:
    """Tests for SerpAPISearchTool._format_news_results()"""

    def test_format_news_results_with_source_and_date(self):
        """_format_news_results includes source and date when present."""
        tool = SerpAPISearchTool(api_key="key")
        results = [
            {
                "title": "Breaking News",
                "url": "http://news.com",
                "snippet": "Content",
                "date": "2 hours ago",
                "source": "BBC",
            }
        ]

        formatted = tool._format_news_results(results)

        assert "BBC" in formatted
        assert "2 hours ago" in formatted

    def test_format_news_results_without_meta(self):
        """_format_news_results omits meta when source and date empty."""
        tool = SerpAPISearchTool(api_key="key")
        results = [{"title": "News", "url": "http://news.com", "snippet": "Content", "date": "", "source": ""}]

        formatted = tool._format_news_results(results)

        assert "Source:" not in formatted


class TestSerpAPISearchToolFormatImageResults:
    """Tests for SerpAPISearchTool._format_image_results()"""

    def test_format_image_results_with_all_fields(self):
        """_format_image_results includes all fields when present."""
        tool = SerpAPISearchTool(api_key="key")
        results = [
            {
                "title": "Cat Photo",
                "url": "http://site.com/page",
                "image_url": "http://site.com/cat.jpg",
                "source": "Cat Site",
                "width": 1920,
                "height": 1080,
            }
        ]

        formatted = tool._format_image_results(results)

        assert "Cat Photo" in formatted
        assert "http://site.com/page" in formatted
        assert "http://site.com/cat.jpg" in formatted
        assert "Cat Site" in formatted
        assert "1920x1080" in formatted

    def test_format_image_results_without_optional_fields(self):
        """_format_image_results handles missing optional fields."""
        tool = SerpAPISearchTool(api_key="key")
        results = [{"title": "Image", "url": "http://site.com", "image_url": "http://site.com/img.png"}]

        formatted = tool._format_image_results(results)

        assert "Image" in formatted
        assert "Source:" not in formatted
        assert "Size:" not in formatted


class TestSerpAPISearchToolSetEngine:
    """Tests for SerpAPISearchTool.set_engine()"""

    def test_set_engine_changes_engine(self):
        """set_engine() changes the search engine."""
        tool = SerpAPISearchTool(api_key="key", engine="google")

        tool.set_engine("bing")

        assert tool._engine == "bing"


class TestSerpAPISearchToolCallable:
    """Tests for SerpAPISearchTool.__call__ (inherited from BaseTool)."""

    def test_tool_is_callable(self):
        """Tool can be called directly via __call__."""
        tool = SerpAPISearchTool(api_key="key")

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client = MagicMock()
            mock_client.search.return_value = {"organic_results": []}
            mock_client_class.return_value = mock_client
            mock_get_client.return_value = mock_client_class

            result = tool(query="direct call")

            assert result.success is True


class TestSerpAPISearchToolSchema:
    """Tests for SerpAPISearchTool.get_schema() (inherited from BaseTool)."""

    def test_get_schema_returns_function_format(self):
        """get_schema returns OpenAI function calling format."""
        tool = SerpAPISearchTool(api_key="key")

        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "web_search"
        assert "parameters" in schema["function"]