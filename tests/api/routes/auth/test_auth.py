import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock

from api.routes.auth import router
from api.auth.schemas import TokenResponse, UserContext
from api.auth.service import AuthenticationError


@pytest.fixture()
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def mock_auth_service():
    service = Mock()
    service.authenticate = AsyncMock()
    service.provider_name = "mock-provider"
    service.requires_credentials = True
    service.is_open_mode = False
    return service


@pytest.fixture(autouse=True)
def override_dependencies(app, mock_auth_service):
    from api.auth import dependencies

    app.dependency_overrides[dependencies.get_auth_service] = lambda: mock_auth_service
    app.dependency_overrides[dependencies.get_current_user] = (
        lambda: UserContext(user_id="user-1", is_authenticated=True)
    )
    yield
    app.dependency_overrides.clear()


# ------------------------------------------------------------------
# /auth/token
# ------------------------------------------------------------------

def test_obtain_token_success(client, mock_auth_service):
    mock_auth_service.authenticate.return_value = TokenResponse(
        access_token="token",
        token_type="bearer",
        expires_in=3600,
    )

    response = client.post(
        "/auth/token",
        json={"app_secret": "secret"},
    )

    assert response.status_code == 200
    assert response.json()["access_token"] == "token"


def test_obtain_token_authentication_error(client, mock_auth_service):
    mock_auth_service.authenticate.side_effect = AuthenticationError(
        error="invalid_credentials",
        description="Bad credentials",
    )

    response = client.post(
        "/auth/token",
        json={"app_secret": "wrong"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["detail"]["error"] == "invalid_credentials"
    assert response.headers["www-authenticate"] == 'Bearer error="invalid_credentials"'


def test_obtain_token_server_error(client, mock_auth_service):
    mock_auth_service.authenticate.side_effect = Exception("boom")

    response = client.post(
        "/auth/token",
        json={"app_secret": "secret"},
    )

    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["error"] == "server_error"
    assert "boom" in body["detail"]["error_description"]


# ------------------------------------------------------------------
# /auth/me
# ------------------------------------------------------------------

def test_get_me(client):
    response = client.get("/auth/me")

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "user-1"
    assert body["is_authenticated"] is True


# ------------------------------------------------------------------
# /auth/status
# ------------------------------------------------------------------

def test_get_auth_status(client, mock_auth_service):
    response = client.get("/auth/status")

    assert response.status_code == 200
    assert response.json() == {
        "provider": "mock-provider",
        "requires_credentials": True,
        "is_open_mode": False,
    }
