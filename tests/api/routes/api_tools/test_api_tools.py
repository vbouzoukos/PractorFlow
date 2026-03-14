"""Tests for api_tools router endpoints."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.auth.schemas import UserContext
from api.auth.dependencies import get_current_user, require_llm_admin
from api.dependencies import (
    get_api_tool_store,
    get_encryption_service,
    resolve_tool,
)
from api.routes.api_tools.api_tools import router
from practorflow.llm.tools.api.models.models import ApiToolConfig

from tests.api.common_api_fixtures import (
    app_base,
    override_auth,
    mock_current_user,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_user():
    return UserContext(user_id="admin-user", is_authenticated=True, permissions=["llm_admin"])


@pytest.fixture
def override_admin(admin_user):
    async def _override():
        return admin_user
    return _override


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def mock_encryption():
    enc = MagicMock()
    enc.encrypt.side_effect = lambda v: f"enc:{v}"
    enc.decrypt.side_effect = lambda v: v.replace("enc:", "")
    return enc


def _make_config(**kwargs):
    defaults = {
        "user_id": "test-user",
        "name": "my_tool",
        "base_url": "https://example.com",
        "path": "/api/data",
        "description": "A test tool",
        "keywords": ["test"],
    }
    defaults.update(kwargs)
    return ApiToolConfig(**defaults)


@pytest.fixture
def sample_tool():
    return _make_config()


@pytest.fixture
def app(app_base, override_auth, mock_store, mock_encryption):
    app_base.include_router(router)
    app_base.dependency_overrides[get_current_user] = override_auth
    app_base.dependency_overrides[get_api_tool_store] = lambda: mock_store
    app_base.dependency_overrides[get_encryption_service] = lambda: mock_encryption
    return app_base


@pytest.fixture
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET / — list_tools
# ---------------------------------------------------------------------------


def test_list_tools_returns_combined(client, app, mock_store, mock_current_user):
    user_tool = _make_config(user_id=mock_current_user.user_id, name="user_tool")
    system_tool = _make_config(user_id="", name="sys_tool", system=True)
    mock_store.list.side_effect = [
        [user_tool],   # user tools call
        [system_tool], # system tools call
    ]

    resp = client.get("/api-tools")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["tools"]) == 2


def test_list_tools_empty(client, mock_store):
    mock_store.list.return_value = []

    resp = client.get("/api-tools")

    assert resp.status_code == 200
    assert resp.json()["count"] == 0


# ---------------------------------------------------------------------------
# POST / — create_tool
# ---------------------------------------------------------------------------


def _create_payload(**kwargs):
    defaults = {
        "name": "new_tool",
        "base_url": "https://api.example.com",
        "path": "/data",
        "description": "A tool",
        "keywords": ["kw"],
    }
    defaults.update(kwargs)
    return defaults


def test_create_tool_success(client, app, mock_store, mock_encryption):
    created = _make_config(name="new_tool")
    mock_store.create.return_value = created

    resp = client.post("/api-tools", json=_create_payload())

    assert resp.status_code == 201
    assert resp.json()["name"] == "new_tool"
    mock_store.create.assert_called_once()


def test_create_tool_encrypts_secret(client, app, mock_store, mock_encryption):
    created = _make_config(name="secret_tool")
    mock_store.create.return_value = created

    resp = client.post(
        "/api-tools",
        json=_create_payload(
            name="secret_tool",
            auth_type="bearer",
            auth_secret="my-secret",
        ),
    )

    assert resp.status_code == 201
    mock_encryption.encrypt.assert_called()


def test_create_tool_encrypts_basic_creds(client, app, mock_store, mock_encryption):
    created = _make_config(name="basic_tool")
    mock_store.create.return_value = created

    resp = client.post(
        "/api-tools",
        json=_create_payload(
            name="basic_tool",
            auth_type="basic",
            auth_username="user",
            auth_password="pass",
        ),
    )

    assert resp.status_code == 201
    assert mock_encryption.encrypt.call_count >= 2


def test_create_system_tool_forbidden_without_admin(client, app):
    resp = client.post("/api-tools", json=_create_payload(system=True))

    assert resp.status_code == 403


def test_create_system_tool_allowed_with_admin(app_base, override_admin, mock_store, mock_encryption):
    app_base.include_router(router)
    app_base.dependency_overrides[get_current_user] = override_admin
    app_base.dependency_overrides[get_api_tool_store] = lambda: mock_store
    app_base.dependency_overrides[get_encryption_service] = lambda: mock_encryption

    created = _make_config(name="sys_tool", system=True, user_id="")
    mock_store.create.return_value = created

    with TestClient(app_base) as c:
        resp = c.post("/api-tools", json=_create_payload(name="sys_tool", system=True))

    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# GET /{tool_id} — get_tool
# ---------------------------------------------------------------------------


def test_get_tool_success(client, app, sample_tool, mock_current_user):
    app.dependency_overrides[resolve_tool] = lambda: sample_tool

    resp = client.get(f"/api-tools/{sample_tool.tool_id}")

    assert resp.status_code == 200
    assert resp.json()["name"] == sample_tool.name


# ---------------------------------------------------------------------------
# PUT /{tool_id} — update_tool
# ---------------------------------------------------------------------------


def test_update_tool_success(client, app, sample_tool, mock_store, mock_encryption):
    updated = _make_config(name="updated_tool")
    mock_store.update.return_value = updated
    app.dependency_overrides[resolve_tool] = lambda: sample_tool

    with MagicMock() as _:
        from unittest.mock import patch
        with patch("api.routes.api_tools.api_tools.is_factory_initialized", return_value=False):
            resp = client.put(
                f"/api-tools/{sample_tool.tool_id}",
                json={"name": "updated_tool"},
            )

    assert resp.status_code == 200
    assert resp.json()["name"] == "updated_tool"


def test_update_tool_store_failure_returns_500(client, app, sample_tool, mock_store):
    mock_store.update.return_value = None
    app.dependency_overrides[resolve_tool] = lambda: sample_tool

    from unittest.mock import patch
    with patch("api.routes.api_tools.api_tools.is_factory_initialized", return_value=False):
        resp = client.put(
            f"/api-tools/{sample_tool.tool_id}",
            json={"name": "fail"},
        )

    assert resp.status_code == 500


def test_update_tool_invalidates_rate_limiter_when_factory_initialized(
    client, app, sample_tool, mock_store, mock_encryption
):
    updated = _make_config(name="updated_tool")
    mock_store.update.return_value = updated
    app.dependency_overrides[resolve_tool] = lambda: sample_tool

    mock_factory = MagicMock()

    from unittest.mock import patch
    with patch("api.routes.api_tools.api_tools.is_factory_initialized", return_value=True):
        with patch("api.routes.api_tools.api_tools.get_factory", return_value=mock_factory):
            resp = client.put(
                f"/api-tools/{sample_tool.tool_id}",
                json={"name": "updated_tool"},
            )

    assert resp.status_code == 200
    mock_factory.invalidate_rate_limiter.assert_called_once_with(sample_tool.tool_id)


def test_update_tool_encrypts_secret(client, app, sample_tool, mock_store, mock_encryption):
    updated = _make_config(name="my_tool")
    mock_store.update.return_value = updated
    app.dependency_overrides[resolve_tool] = lambda: sample_tool

    from unittest.mock import patch
    with patch("api.routes.api_tools.api_tools.is_factory_initialized", return_value=False):
        resp = client.put(
            f"/api-tools/{sample_tool.tool_id}",
            json={"auth_secret": "new-secret"},
        )

    assert resp.status_code == 200
    mock_encryption.encrypt.assert_called()


# ---------------------------------------------------------------------------
# DELETE /{tool_id} — delete_tool
# ---------------------------------------------------------------------------


def test_delete_tool_success(client, app, sample_tool, mock_store):
    mock_store.delete.return_value = True
    app.dependency_overrides[resolve_tool] = lambda: sample_tool

    from unittest.mock import patch
    with patch("api.routes.api_tools.api_tools.is_factory_initialized", return_value=False):
        resp = client.delete(f"/api-tools/{sample_tool.tool_id}")

    assert resp.status_code == 204


def test_delete_tool_store_failure_returns_500(client, app, sample_tool, mock_store):
    mock_store.delete.return_value = False
    app.dependency_overrides[resolve_tool] = lambda: sample_tool

    from unittest.mock import patch
    with patch("api.routes.api_tools.api_tools.is_factory_initialized", return_value=False):
        resp = client.delete(f"/api-tools/{sample_tool.tool_id}")

    assert resp.status_code == 500


def test_delete_tool_invalidates_rate_limiter_when_factory_initialized(
    client, app, sample_tool, mock_store
):
    mock_store.delete.return_value = True
    app.dependency_overrides[resolve_tool] = lambda: sample_tool
    mock_factory = MagicMock()

    from unittest.mock import patch
    with patch("api.routes.api_tools.api_tools.is_factory_initialized", return_value=True):
        with patch("api.routes.api_tools.api_tools.get_factory", return_value=mock_factory):
            resp = client.delete(f"/api-tools/{sample_tool.tool_id}")

    assert resp.status_code == 204
    mock_factory.invalidate_rate_limiter.assert_called_once_with(sample_tool.tool_id)


# ---------------------------------------------------------------------------
# GET /{tool_id}/secrets — get_tool_secrets
# ---------------------------------------------------------------------------


def test_get_tool_secrets_success(client, app, mock_encryption):
    tool = _make_config(auth_secret="enc:tok", auth_username=None, auth_password=None)
    app.dependency_overrides[resolve_tool] = lambda: tool

    mock_encryption.decrypt.return_value = "tok"

    resp = client.get(f"/api-tools/{tool.tool_id}/secrets")

    assert resp.status_code == 200
    data = resp.json()
    assert data["auth_secret"] == "tok"
    assert data["auth_username"] is None
    assert data["auth_password"] is None


def test_get_tool_secrets_decrypt_failure_returns_500(client, app, mock_encryption):
    tool = _make_config(auth_secret="enc:bad")
    app.dependency_overrides[resolve_tool] = lambda: tool

    mock_encryption.decrypt.side_effect = Exception("bad key")

    resp = client.get(f"/api-tools/{tool.tool_id}/secrets")

    assert resp.status_code == 500


def test_get_tool_secrets_no_secrets(client, app, mock_encryption):
    tool = _make_config()  # auth_secret=None by default
    app.dependency_overrides[resolve_tool] = lambda: tool

    resp = client.get(f"/api-tools/{tool.tool_id}/secrets")

    assert resp.status_code == 200
    data = resp.json()
    assert data["auth_secret"] is None
    assert data["auth_username"] is None
    assert data["auth_password"] is None
