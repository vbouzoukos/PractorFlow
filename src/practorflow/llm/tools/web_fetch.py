"""
Web fetch tool for retrieving web page content.

Fetches and extracts readable content from web pages.
Useful for retrieving information from URLs provided by search results.
"""

import re
from typing import List, Optional

import httpx

from practorflow.llm.tools.async_tool import AsyncBaseTool
from practorflow.llm.tools.base import ToolParameter, ToolResult
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class WebFetchTool(AsyncBaseTool):
    """
    Web fetch tool for retrieving web page content.

    Fetches a URL and extracts readable text content,
    stripping HTML tags, scripts, and styles.
    """

    def __init__(
        self,
        default_timeout: int = 30,
        max_content_length: int = 50000,
        user_agent: Optional[str] = None,
    ):
        """
        Initialize web fetch tool.

        Args:
            default_timeout: Default request timeout in seconds.
            max_content_length: Maximum content length to return in characters.
            user_agent: Custom User-Agent header.
        """
        self._default_timeout = default_timeout
        self._max_content_length = max_content_length
        self._user_agent = user_agent or (
            "Mozilla/5.0 (compatible; PractorFlow/1.0; +https://github.com/practorflow)"
        )

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch and extract readable content from a web page URL. "
            "Use this tool to retrieve the text content of a specific web page "
            "when you need more details than a search snippet provides."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="The URL of the web page to fetch",
                required=True,
            ),
            ToolParameter(
                name="extract_mode",
                type="string",
                description="Content extraction mode: text (readable text), raw (raw HTML), metadata (title, description, links)",
                required=False,
                default="text",
                enum=["text", "raw", "metadata"],
            ),
        ]

    def _extract_text(self, html: str) -> str:
        """
        Extract readable text from HTML content.

        Args:
            html: Raw HTML content.

        Returns:
            Extracted text content.
        """
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</(p|div|h[1-6]|li|tr)>', '\n', text, flags=re.IGNORECASE)

        text = re.sub(r'<[^>]+>', '', text)

        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'&\w+;', '', text)

        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                line = re.sub(r'\s+', ' ', line)
                lines.append(line)

        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _extract_metadata(self, html: str, url: str) -> dict:
        """
        Extract metadata from HTML content.

        Args:
            html: Raw HTML content.
            url: Source URL.

        Returns:
            Dictionary with metadata.
        """
        metadata = {"url": url}

        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        desc_match = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']',
            html, re.IGNORECASE
        )
        if not desc_match:
            desc_match = re.search(
                r'<meta[^>]*content=["\'](.*?)["\'][^>]*name=["\']description["\']',
                html, re.IGNORECASE
            )
        if desc_match:
            metadata["description"] = desc_match.group(1).strip()

        og_title = re.search(
            r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\'](.*?)["\']',
            html, re.IGNORECASE
        )
        if og_title:
            metadata["og_title"] = og_title.group(1).strip()

        og_desc = re.search(
            r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\'](.*?)["\']',
            html, re.IGNORECASE
        )
        if og_desc:
            metadata["og_description"] = og_desc.group(1).strip()

        links = []
        for link_match in re.finditer(r'<a[^>]*href=["\'](https?://[^"\']+)["\'][^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL):
            href = link_match.group(1)
            text = re.sub(r'<[^>]+>', '', link_match.group(2)).strip()
            if text and len(text) < 200:
                links.append({"url": href, "text": text})
            if len(links) >= 20:
                break
        metadata["links"] = links

        headings = []
        for h_match in re.finditer(r'<h([1-3])[^>]*>(.*?)</h\1>', html, re.IGNORECASE | re.DOTALL):
            level = h_match.group(1)
            text = re.sub(r'<[^>]+>', '', h_match.group(2)).strip()
            if text:
                headings.append({"level": int(level), "text": text})
            if len(headings) >= 15:
                break
        metadata["headings"] = headings

        return metadata

    async def execute(self, **kwargs) -> ToolResult:
        """
        Fetch and extract content from a web page.

        Args:
            url: URL to fetch.
            extract_mode: Content extraction mode (text, raw, metadata).

        Returns:
            ToolResult with page content or error.
        """
        url = kwargs.get("url")
        extract_mode = kwargs.get("extract_mode", "text")

        if not url:
            return ToolResult(success=False, error="URL parameter is required")

        if not url.startswith(("http://", "https://")):
            return ToolResult(success=False, error="URL must start with http:// or https://")

        try:
            logger.debug(f"[WebFetch] Fetching: {url}")

            headers = {
                "User-Agent": self._user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }

            async with httpx.AsyncClient(timeout=self._default_timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)

            if response.status_code >= 400:
                return ToolResult(
                    success=False,
                    error=f"HTTP error {response.status_code}",
                    metadata={"url": url, "status_code": response.status_code},
                )

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return ToolResult(
                    success=False,
                    error=f"Unsupported content type: {content_type}",
                    metadata={"url": url, "content_type": content_type},
                )

            html = response.text

            if extract_mode == "raw":
                content = html[:self._max_content_length]
                if len(html) > self._max_content_length:
                    content += "\n... (truncated)"

                return ToolResult(
                    success=True,
                    data=content,
                    metadata={
                        "url": url,
                        "mode": "raw",
                        "status_code": response.status_code,
                        "content_length": len(html),
                        "truncated": len(html) > self._max_content_length,
                    },
                )

            elif extract_mode == "metadata":
                metadata = self._extract_metadata(html, url)
                metadata["status_code"] = response.status_code

                return ToolResult(
                    success=True,
                    data=metadata,
                    metadata={"url": url, "mode": "metadata"},
                )

            else:
                text = self._extract_text(html)

                if len(text) > self._max_content_length:
                    text = text[:self._max_content_length] + "\n... (truncated)"

                title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else None

                return ToolResult(
                    success=True,
                    data=text,
                    metadata={
                        "url": url,
                        "mode": "text",
                        "title": title,
                        "status_code": response.status_code,
                        "content_length": len(text),
                        "truncated": len(text) >= self._max_content_length,
                    },
                )

        except Exception as e:
            logger.error(f"[WebFetch] Error: {e}")
            return ToolResult(success=False, error=f"Failed to fetch URL: {str(e)}")