"""Tests for MCP server router endpoints."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.auth.dependencies import get_current_user, require_llm_admin
from api.dependencies import get_mcp_server_store
from api.routes.mcp.mcp import router
from practorflow.llm.tools.mcp.types import (
    MCPServerConfig,
    TransportType,
    StdioConfig,
    HttpConfig,
)

from tests.api.common_api_fixtures import app_base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_user():
    from api.auth.schemas import UserContext
    return UserContext(user_id="admin", is_authenticated=True, permissions=["llm_admin"])


@pytest.fixture
def override_admin(admin_user):
    async def _override():
        return admin_user
    return _override


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def app(app_base, override_admin, mock_store):
    app_base.include_router(router)
    app_base.dependency_overrides[get_current_user] = override_admin
    app_base.dependency_overrides[require_llm_admin] = override_admin
    app_base.dependency_overrides[get_mcp_server_store] = lambda: mock_store
    return app_base


@pytest.fixture
def client(app):
    return TestClient(app)


def _make_server(**kwargs):
    defaults = {
        "name": "test-server",
        "transport": TransportType.STDIO,
        "stdio_config": StdioConfig(command="npx", args=["mcp"]),
    }
    defaults.update(kwargs)
    return MCPServerConfig(**defaults)


# ---------------------------------------------------------------------------
# POST / — create_server
# ---------------------------------------------------------------------------


def test_create_server_stdio_success(client, mock_store):
    created = _make_server()
    mock_store.list.return_value = []
    mock_store.create.return_value = created

    resp = client.post(
        "/mcp/servers",
        json={
            "name": "test-server",
            "transport": "stdio",
            "stdio_config": {"command": "npx", "args": ["mcp"]},
        },
    )

    assert resp.status_code == 201
    assert resp.json()["name"] == "test-server"


def test_create_server_missing_stdio_config_returns_400(client, mock_store):
    mock_store.list.return_value = []

    resp = client.post(
        "/mcp/servers",
        json={"name": "bad-server", "transport": "stdio"},
    )

    assert resp.status_code == 400


def test_create_server_missing_http_config_returns_400(client, mock_store):
    mock_store.list.return_value = []

    resp = client.post(
        "/mcp/servers",
        json={"name": "bad-server", "transport": "streamable_http"},
    )

    assert resp.status_code == 400


def test_create_server_duplicate_name_returns_409(client, mock_store):
    existing = _make_server(name="existing-server")
    mock_store.list.return_value = [existing]

    resp = client.post(
        "/mcp/servers",
        json={
            "name": "existing-server",
            "transport": "stdio",
            "stdio_config": {"command": "npx"},
        },
    )

    assert resp.status_code == 409



# ---------------------------------------------------------------------------
# GET / — list_servers
# ---------------------------------------------------------------------------


def test_list_servers_returns_all(client, mock_store):
    s1 = _make_server(name="s1")
    s2 = _make_server(name="s2")
    mock_store.list.return_value = [s1, s2]

    resp = client.get("/mcp/servers")

    assert resp.status_code == 200
    assert resp.json()["count"] == 2


def test_list_servers_empty(client, mock_store):
    mock_store.list.return_value = []

    resp = client.get("/mcp/servers")

    assert resp.status_code == 200
    assert resp.json()["count"] == 0


# ---------------------------------------------------------------------------
# GET /{server_id} — get_server
# ---------------------------------------------------------------------------


def test_get_server_found(client, mock_store):
    server = _make_server()
    mock_store.get.return_value = server

    resp = client.get(f"/mcp/servers/{server.server_id}")

    assert resp.status_code == 200
    assert resp.json()["name"] == "test-server"


def test_get_server_not_found_returns_404(client, mock_store):
    mock_store.get.return_value = None

    resp = client.get("/mcp/servers/nonexistent-id")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /{server_id} — update_server
# ---------------------------------------------------------------------------


def test_update_server_success(client, mock_store):
    server = _make_server()
    updated = _make_server(name="updated-server")
    mock_store.get.return_value = server
    mock_store.list.return_value = []
    mock_store.update.return_value = updated

    resp = client.put(
        f"/mcp/servers/{server.server_id}",
        json={"name": "updated-server"},
    )

    assert resp.status_code == 200
    assert resp.json()["name"] == "updated-server"


def test_update_server_not_found_returns_404(client, mock_store):
    mock_store.get.return_value = None

    resp = client.put(
        "/mcp/servers/nonexistent",
        json={"name": "new-name"},
    )

    assert resp.status_code == 404


def test_update_server_name_conflict_returns_409(client, mock_store):
    server = _make_server(name="original")
    other = _make_server(name="taken")
    mock_store.get.return_value = server
    mock_store.list.return_value = [other]

    resp = client.put(
        f"/mcp/servers/{server.server_id}",
        json={"name": "taken"},
    )

    assert resp.status_code == 409


def test_update_server_store_failure_returns_500(client, mock_store):
    server = _make_server()
    mock_store.get.return_value = server
    mock_store.list.return_value = []
    mock_store.update.return_value = None

    resp = client.put(
        f"/mcp/servers/{server.server_id}",
        json={"name": "fail"},
    )

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# DELETE /{server_id} — delete_server
# ---------------------------------------------------------------------------


def test_delete_server_success(client, mock_store):
    server = _make_server()
    mock_store.get.return_value = server
    mock_store.delete.return_value = True

    resp = client.delete(f"/mcp/servers/{server.server_id}")

    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


def test_delete_server_not_found_returns_404(client, mock_store):
    mock_store.get.return_value = None

    resp = client.delete("/mcp/servers/nonexistent")

    assert resp.status_code == 404


def test_delete_server_store_failure_returns_500(client, mock_store):
    server = _make_server()
    mock_store.get.return_value = server
    mock_store.delete.return_value = False

    resp = client.delete(f"/mcp/servers/{server.server_id}")

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /{server_id}/reload — reload_server
# ---------------------------------------------------------------------------


def test_reload_server_not_found_returns_404(client, mock_store):
    mock_store.get.return_value = None

    resp = client.post("/mcp/servers/nonexistent/reload")

    assert resp.status_code == 404


def test_reload_server_returns_501(client, mock_store):
    server = _make_server()
    mock_store.get.return_value = server

    resp = client.post(f"/mcp/servers/{server.server_id}/reload")

    assert resp.status_code == 501
