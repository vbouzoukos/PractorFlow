"""Tests for tools_options router endpoints."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.auth.dependencies import get_current_user
from api.dependencies import (
    get_api_tool_store,
    get_mcp_server_store,
    get_tool_preferences_store,
)
from api.routes.tools.tools_options import router
from practorflow.llm.tools.user_preferences import UserToolPreferences, EnabledToolEntry
from practorflow.llm.tools.mcp.types import MCPServerConfig, TransportType, StdioConfig, MCPToolConfig
from practorflow.llm.tools.api.models.models import ApiToolConfig

from tests.api.common_api_fixtures import app_base, override_auth, mock_current_user


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_prefs_store():
    return MagicMock()


@pytest.fixture
def mock_api_store():
    return MagicMock()


@pytest.fixture
def mock_mcp_store():
    return MagicMock()


@pytest.fixture
def app(app_base, override_auth, mock_prefs_store, mock_api_store, mock_mcp_store):
    app_base.include_router(router)
    app_base.dependency_overrides[get_current_user] = override_auth
    app_base.dependency_overrides[get_tool_preferences_store] = lambda: mock_prefs_store
    app_base.dependency_overrides[get_api_tool_store] = lambda: mock_api_store
    app_base.dependency_overrides[get_mcp_server_store] = lambda: mock_mcp_store
    return app_base


@pytest.fixture
def client(app):
    return TestClient(app)


def _make_prefs(user_id="test-user", enabled_tools=None, enabled_mcp_servers=None):
    return UserToolPreferences(
        user_id=user_id,
        enabled_tools=enabled_tools or [],
        enabled_mcp_servers=enabled_mcp_servers or [],
    )


def _make_api_config(name="my_tool", user_id="test-user", system=False):
    return ApiToolConfig(
        user_id=user_id,
        name=name,
        base_url="https://example.com",
        path="/data",
        description="A tool",
        keywords=["kw"],
        system=system,
    )


# ---------------------------------------------------------------------------
# GET / — list_tools
# ---------------------------------------------------------------------------


def test_list_tools_with_existing_prefs(client, mock_prefs_store, mock_api_store, mock_mcp_store):
    prefs = _make_prefs(enabled_tools=[EnabledToolEntry(type="builtin", id="search_knowledge")])
    mock_prefs_store.get.return_value = prefs
    mock_api_store.list.return_value = []
    mock_mcp_store.list.return_value = []

    resp = client.get("/tools")

    assert resp.status_code == 200
    data = resp.json()
    assert "builtin_tools" in data
    assert "api_tools" in data
    assert "mcp_tools" in data
    # search_knowledge should be enabled
    enabled = [t for t in data["builtin_tools"] if t["id"] == "search_knowledge"]
    assert enabled[0]["enabled"] is True


def test_list_tools_creates_default_prefs_when_none(
    client, mock_prefs_store, mock_api_store, mock_mcp_store
):
    mock_prefs_store.get.return_value = None
    mock_prefs_store.create_default.return_value = _make_prefs()
    mock_api_store.list.return_value = []
    mock_mcp_store.list.return_value = []

    resp = client.get("/tools")

    assert resp.status_code == 200
    mock_prefs_store.create_default.assert_called_once()


def test_list_tools_includes_api_tools(
    client, mock_prefs_store, mock_api_store, mock_mcp_store, mock_current_user
):
    prefs = _make_prefs()
    mock_prefs_store.get.return_value = prefs
    api_tool = _make_api_config(user_id=mock_current_user.user_id)
    mock_api_store.list.return_value = [api_tool]
    mock_mcp_store.list.return_value = []

    resp = client.get("/tools")

    assert resp.status_code == 200
    assert len(resp.json()["api_tools"]) > 0


def test_list_tools_includes_mcp_tools(
    client, mock_prefs_store, mock_api_store, mock_mcp_store
):
    prefs = _make_prefs()
    mock_prefs_store.get.return_value = prefs
    mock_api_store.list.return_value = []

    tool_cfg = MCPToolConfig(name="mcp-tool", description="desc")
    server = MCPServerConfig(
        name="s1",
        transport=TransportType.STDIO,
        stdio_config=StdioConfig(command="npx"),
        tools=[tool_cfg],
    )
    mock_mcp_store.list.return_value = [server]

    resp = client.get("/tools")

    assert resp.status_code == 200
    assert len(resp.json()["mcp_tools"]) == 1


def test_list_tools_mcp_tool_enabled_when_both_in_prefs(
    client, mock_prefs_store, mock_api_store, mock_mcp_store
):
    tool_cfg = MCPToolConfig(name="mcp-tool", description="desc")
    server = MCPServerConfig(
        name="s1",
        transport=TransportType.STDIO,
        stdio_config=StdioConfig(command="npx"),
        tools=[tool_cfg],
    )
    prefs = _make_prefs(
        enabled_tools=[EnabledToolEntry(type="mcp", id=tool_cfg.tool_id)],
        enabled_mcp_servers=[server.server_id],
    )
    mock_prefs_store.get.return_value = prefs
    mock_api_store.list.return_value = []
    mock_mcp_store.list.return_value = [server]

    resp = client.get("/tools")

    mcp_tools = resp.json()["mcp_tools"]
    assert mcp_tools[0]["enabled"] is True


# ---------------------------------------------------------------------------
# PUT /preferences — update_preference
# ---------------------------------------------------------------------------


def test_update_preference_enable_new_tool(client, mock_prefs_store):
    prefs = _make_prefs()
    mock_prefs_store.get.return_value = prefs

    resp = client.put(
        "/tools/preferences",
        json={"type": "builtin", "id": "calculator", "enabled": True},
    )

    assert resp.status_code == 200
    assert resp.json()["enabled"] is True
    mock_prefs_store.update.assert_called_once()


def test_update_preference_disable_existing_tool(client, mock_prefs_store):
    prefs = _make_prefs(enabled_tools=[EnabledToolEntry(type="builtin", id="calculator")])
    mock_prefs_store.get.return_value = prefs

    resp = client.put(
        "/tools/preferences",
        json={"type": "builtin", "id": "calculator", "enabled": False},
    )

    assert resp.status_code == 200
    assert resp.json()["enabled"] is False


def test_update_preference_enable_already_present_does_not_duplicate(client, mock_prefs_store):
    prefs = _make_prefs(enabled_tools=[EnabledToolEntry(type="builtin", id="calculator")])
    mock_prefs_store.get.return_value = prefs

    resp = client.put(
        "/tools/preferences",
        json={"type": "builtin", "id": "calculator", "enabled": True},
    )

    assert resp.status_code == 200
    # Should NOT append a duplicate
    assert len(prefs.enabled_tools) == 1


def test_update_preference_creates_default_when_none(client, mock_prefs_store):
    mock_prefs_store.get.return_value = None
    mock_prefs_store.create_default.return_value = _make_prefs()

    resp = client.put(
        "/tools/preferences",
        json={"type": "builtin", "id": "calculator", "enabled": True},
    )

    assert resp.status_code == 200
    mock_prefs_store.create_default.assert_called_once()


def test_update_preference_mcp_enable_adds_server(client, mock_prefs_store):
    prefs = _make_prefs()
    mock_prefs_store.get.return_value = prefs

    resp = client.put(
        "/tools/preferences",
        json={"type": "mcp", "id": "tool1", "enabled": True, "server_id": "srv1"},
    )

    assert resp.status_code == 200
    assert "srv1" in prefs.enabled_mcp_servers


def test_update_preference_mcp_disable_removes_server_when_no_other_tools(
    client, mock_prefs_store
):
    prefs = _make_prefs(
        enabled_tools=[EnabledToolEntry(type="mcp", id="tool1")],
        enabled_mcp_servers=["srv1"],
    )
    mock_prefs_store.get.return_value = prefs

    resp = client.put(
        "/tools/preferences",
        json={"type": "mcp", "id": "tool1", "enabled": False, "server_id": "srv1"},
    )

    assert resp.status_code == 200
    assert "srv1" not in prefs.enabled_mcp_servers


def test_update_preference_mcp_disable_keeps_server_when_other_tools_remain(
    client, mock_prefs_store
):
    prefs = _make_prefs(
        enabled_tools=[
            EnabledToolEntry(type="mcp", id="tool1"),
            EnabledToolEntry(type="mcp", id="tool2"),
        ],
        enabled_mcp_servers=["srv1"],
    )
    mock_prefs_store.get.return_value = prefs

    resp = client.put(
        "/tools/preferences",
        json={"type": "mcp", "id": "tool1", "enabled": False, "server_id": "srv1"},
    )

    assert resp.status_code == 200
    assert "srv1" in prefs.enabled_mcp_servers
