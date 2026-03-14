"""Tests for tools route schemas."""

import pytest
from api.routes.tools.schemas import (
    ToolPreferenceUpdateRequest,
    ToolPreferenceUpdateResponse,
    ToolInfoBase,
    BuiltinToolInfo,
    ApiToolInfo,
    MCPToolInfo,
    ToolsListResponse,
)


def test_tool_preference_update_request():
    req = ToolPreferenceUpdateRequest(type="builtin", id="search", enabled=True)
    assert req.type == "builtin"
    assert req.enabled is True
    assert req.server_id is None


def test_tool_preference_update_request_with_server_id():
    req = ToolPreferenceUpdateRequest(
        type="mcp", id="tool1", enabled=True, server_id="server1"
    )
    assert req.server_id == "server1"


def test_tool_preference_update_response():
    resp = ToolPreferenceUpdateResponse(type="api", id="t1", enabled=False)
    assert resp.enabled is False


def test_builtin_tool_info():
    info = BuiltinToolInfo(
        id="search", type="builtin", name="search", description="Search", enabled=True
    )
    assert info.type == "builtin"
    assert info.enabled is True


def test_api_tool_info():
    info = ApiToolInfo(
        id="t1", type="api", name="my_tool", description="A tool", enabled=False, system=False
    )
    assert info.system is False


def test_mcp_tool_info():
    info = MCPToolInfo(
        id="t1",
        type="mcp",
        name="tool",
        description="d",
        enabled=True,
        server_id="s1",
        server_name="my-server",
    )
    assert info.server_id == "s1"
    assert info.server_name == "my-server"


def test_tools_list_response_count():
    b = BuiltinToolInfo(id="b1", type="builtin", name="b", description="d", enabled=True)
    a = ApiToolInfo(id="a1", type="api", name="a", description="d", enabled=False, system=False)
    resp = ToolsListResponse(builtin_tools=[b], api_tools=[a], mcp_tools=[], count=2)
    assert resp.count == 2
    assert len(resp.builtin_tools) == 1
    assert len(resp.api_tools) == 1
