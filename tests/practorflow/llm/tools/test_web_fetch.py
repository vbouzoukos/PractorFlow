import pytest
from types import SimpleNamespace

from practorflow.llm.tools.web_fetch import WebFetchTool


class TestWebFetchTool:
    def test_init_name_description_and_parameters(self):
        tool = WebFetchTool()

        assert tool._default_timeout == 30
        assert tool._max_content_length == 50000
        assert "Mozilla" in tool._user_agent

        assert tool.name == "web_fetch"
        assert isinstance(tool.description, str)

        params = tool.parameters
        assert params[0].name == "url"
        assert params[1].name == "extract_mode"
        assert params[1].default == "text"
        assert params[1].enum == ["text", "raw", "metadata"]

    async def test_execute_missing_url_and_invalid_scheme(self):
        tool = WebFetchTool()

        result = await tool.execute()
        assert result.success is False
        assert "URL parameter is required" in result.error

        result = await tool.execute(url="ftp://example.com")
        assert result.success is False
        assert "URL must start with http" in result.error

    async def test_execute_http_error_status(self, monkeypatch):
        tool = WebFetchTool()

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, headers):
                return SimpleNamespace(
                    status_code=404,
                    headers={"content-type": "text/html"},
                    text="<html></html>",
                )

        monkeypatch.setattr("httpx.AsyncClient", lambda **_: FakeClient())

        result = await tool.execute(url="https://example.com")

        assert result.success is False
        assert "HTTP error 404" in result.error
        assert result.metadata["status_code"] == 404


    async def test_execute_unsupported_content_type(self, monkeypatch):
        tool = WebFetchTool()

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, headers):
                return SimpleNamespace(
                    status_code=200,
                    headers={"content-type": "application/json"},
                    text="{}",
                )

        monkeypatch.setattr("httpx.AsyncClient", lambda **_: FakeClient())

        result = await tool.execute(url="https://example.com")

        assert result.success is False
        assert "Unsupported content type" in result.error

    async def test_execute_raw_mode_with_truncation(self, monkeypatch):
        tool = WebFetchTool(max_content_length=10)

        html = "<html>" + ("x" * 20) + "</html>"

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, headers):
                return SimpleNamespace(
                    status_code=200,
                    headers={"content-type": "text/html"},
                    text=html,
                )

        monkeypatch.setattr("httpx.AsyncClient", lambda **_: FakeClient())

        result = await tool.execute(url="https://example.com", extract_mode="raw")

        assert result.success is True
        assert result.data.endswith("(truncated)")
        assert result.metadata["mode"] == "raw"
        assert result.metadata["truncated"] is True

    async def test_execute_metadata_mode_full_coverage(self, monkeypatch):
        tool = WebFetchTool()

        html = """
        <html>
            <title>Test Page</title>
            <meta name="description" content="desc">
            <meta property="og:title" content="og title">
            <meta property="og:description" content="og desc">
            <h1>Main</h1>
            <h2>Sub</h2>
            <a href="https://a.com">Link A</a>
        </html>
        """

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, headers):
                return SimpleNamespace(
                    status_code=200,
                    headers={"content-type": "text/html"},
                    text=html,
                )

        monkeypatch.setattr("httpx.AsyncClient", lambda **_: FakeClient())

        result = await tool.execute(url="https://example.com", extract_mode="metadata")

        assert result.success is True
        assert result.data["title"] == "Test Page"
        assert result.data["description"] == "desc"
        assert result.data["og_title"] == "og title"
        assert result.data["og_description"] == "og desc"
        assert result.data["links"][0]["url"] == "https://a.com"
        assert result.data["headings"][0]["level"] == 1

    async def test_execute_text_mode_with_title_and_truncation(self, monkeypatch):
        tool = WebFetchTool(max_content_length=50)

        html = """
        <html>
            <title>My Title</title>
            <p>Hello world this is a test. The quick brown fox jumps over the lazy dog.</p>
        </html>
        """

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, headers):
                return SimpleNamespace(
                    status_code=200,
                    headers={"content-type": "text/html"},
                    text=html,
                )

        monkeypatch.setattr("httpx.AsyncClient", lambda **_: FakeClient())

        result = await tool.execute(url="https://example.com")

        assert result.success is True
        assert "Hello world" in result.data
        assert result.metadata["mode"] == "text"
        assert result.metadata["title"] == "My Title"
        assert result.metadata["truncated"] is True

    def test_extract_text_removes_tags_and_whitespace(self):
        tool = WebFetchTool()

        html = """
        <html>
            <script>alert(1)</script>
            <style>body{}</style>
            <p> Hello   world </p>
            <div>Test<br>Line</div>
        </html>
        """

        text = tool._extract_text(html)

        assert "alert" not in text
        assert "Hello world" in text
        assert "Test" in text
        assert "Line" in text

    async def test_execute_exception_path(self, monkeypatch):
        tool = WebFetchTool()

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        class FakeClient:
            async def __aenter__(self):
                raise RuntimeError("boom")

            async def __aexit__(self, *args):
                pass

        monkeypatch.setattr("httpx.AsyncClient", lambda **_: FakeClient())

        result = await tool.execute(url="https://example.com")

        assert result.success is False
        assert result.error.startswith("Failed to fetch URL:")
        assert "boom" in result.error

    async def test_extract_metadata_description_fallback_pattern(self, monkeypatch):
        tool = WebFetchTool()

        html = """
        <html>
            <meta content="Fallback description" name="description">
        </html>
        """

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, headers):
                return SimpleNamespace(
                    status_code=200,
                    headers={"content-type": "text/html"},
                    text=html,
                )

        monkeypatch.setattr("httpx.AsyncClient", lambda **_: FakeClient())

        result = await tool.execute(url="https://example.com", extract_mode="metadata")

        assert result.success is True
        assert result.data["description"] == "Fallback description"

    async def test_extract_metadata_links_limit_break(self, monkeypatch):
        tool = WebFetchTool()

        # 21 valid links; if break triggers at 20 appended, the 21st must not appear
        links_html = "".join(
            f'<a href="https://example.com/{i}">Link {i}</a>' for i in range(21)
        )
        html = f"<html>{links_html}</html>"

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, headers):
                return SimpleNamespace(
                    status_code=200,
                    headers={"content-type": "text/html"},
                    text=html,
                )

        monkeypatch.setattr("httpx.AsyncClient", lambda **_: FakeClient())

        result = await tool.execute(url="https://example.com", extract_mode="metadata")

        assert result.success is True

        links = result.data["links"]
        assert len(links) == 20

        urls = [l["url"] for l in links]
        assert "https://example.com/19" in urls
        # The 21st link would be /20, but break should prevent it from being added
        assert "https://example.com/20" not in urls

    async def test_extract_metadata_headings_limit_break(self, monkeypatch):
        tool = WebFetchTool()

        headings_html = "".join(f"<h1>Heading {i}</h1>" for i in range(20))

        html = f"<html>{headings_html}</html>"

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, headers):
                return SimpleNamespace(
                    status_code=200,
                    headers={"content-type": "text/html"},
                    text=html,
                )

        monkeypatch.setattr("httpx.AsyncClient", lambda **_: FakeClient())

        result = await tool.execute(url="https://example.com", extract_mode="metadata")

        assert result.success is True
        assert len(result.data["headings"]) == 15
