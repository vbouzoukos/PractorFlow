"""Tests for MCP API schemas."""

import pytest
from api.routes.mcp.schemas import (
    MCPServerCreateRequest,
    MCPServerUpdateRequest,
    MCPServerResponse,
    MCPServerListResponse,
    MCPServerToolsResponse,
    MCPServerReloadResponse,
    MCPServerDeleteResponse,
    MCPToolInfo,
)
from practorflow.llm.tools.mcp.types import (
    MCPServerConfig,
    TransportType,
    StdioConfig,
    HttpConfig,
    MCPToolConfig,
)


def _make_server_config(**kwargs):
    defaults = {
        "name": "test-server",
        "transport": TransportType.STDIO,
        "stdio_config": StdioConfig(command="npx"),
    }
    defaults.update(kwargs)
    return MCPServerConfig(**defaults)


# ---------------------------------------------------------------------------
# MCPServerCreateRequest
# ---------------------------------------------------------------------------


def test_create_request_stdio():
    req = MCPServerCreateRequest(
        name="my-server",
        transport=TransportType.STDIO,
        stdio_config=StdioConfig(command="npx", args=["mcp"]),
    )
    assert req.name == "my-server"
    assert req.stdio_config.command == "npx"



def test_create_request_tools_default_empty():
    req = MCPServerCreateRequest(name="s", transport=TransportType.STDIO)
    assert req.tools == []


# ---------------------------------------------------------------------------
# MCPServerUpdateRequest
# ---------------------------------------------------------------------------


def test_update_request_all_optional():
    req = MCPServerUpdateRequest()
    assert req.name is None
    assert req.stdio_config is None


def test_update_request_with_name():
    req = MCPServerUpdateRequest(name="new-name")
    assert req.name == "new-name"


# ---------------------------------------------------------------------------
# MCPServerResponse.from_config
# ---------------------------------------------------------------------------


def test_from_config_basic():
    config = _make_server_config()
    resp = MCPServerResponse.from_config(config)
    assert resp.server_id == config.server_id
    assert resp.name == "test-server"
    assert resp.transport == TransportType.STDIO


def test_from_config_preserves_stdio_config():
    config = _make_server_config(
        stdio_config=StdioConfig(command="python", args=["-m", "mcp"])
    )
    resp = MCPServerResponse.from_config(config)
    assert resp.stdio_config.command == "python"


def test_from_config_preserves_tools():
    tool = MCPToolConfig(name="my-tool")
    config = _make_server_config(tools=[tool])
    resp = MCPServerResponse.from_config(config)
    assert len(resp.tools) == 1
    assert resp.tools[0].name == "my-tool"


# ---------------------------------------------------------------------------
# MCPServerListResponse
# ---------------------------------------------------------------------------


def test_list_response():
    config = _make_server_config()
    server_resp = MCPServerResponse.from_config(config)
    resp = MCPServerListResponse(servers=[server_resp], count=1)
    assert resp.count == 1
    assert len(resp.servers) == 1


# ---------------------------------------------------------------------------
# MCPServerToolsResponse
# ---------------------------------------------------------------------------


def test_tools_response_connected():
    tool = MCPToolInfo(name="tool1", description="d", input_schema={})
    resp = MCPServerToolsResponse(
        server_id="s1",
        connected=True,
        tools=[tool],
        error=None,
    )
    assert resp.connected is True
    assert len(resp.tools) == 1


def test_tools_response_error():
    resp = MCPServerToolsResponse(
        server_id="s1",
        connected=False,
        error="connection refused",
    )
    assert resp.connected is False
    assert resp.error == "connection refused"


# ---------------------------------------------------------------------------
# MCPServerDeleteResponse / MCPServerReloadResponse
# ---------------------------------------------------------------------------


def test_delete_response():
    resp = MCPServerDeleteResponse(server_id="s1", deleted=True, message="ok")
    assert resp.deleted is True


def test_reload_response():
    resp = MCPServerReloadResponse(server_id="s1", reloaded=True, message="done")
    assert resp.reloaded is True
