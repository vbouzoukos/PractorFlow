import pytest
from unittest.mock import AsyncMock, Mock, patch

from api.auth.service import AuthService, AuthenticationError
from api.auth.schemas import TokenRequest, TokenResponse, UserContext
from api.auth.jwt_handler import JWTError


@pytest.fixture()
def config_local():
    cfg = Mock()
    cfg.is_oidc_mode = False
    cfg.is_open_mode = False
    cfg.app_secret = "secret"
    cfg.jwt = Mock()
    return cfg


@pytest.fixture()
def config_oidc():
    cfg = Mock()
    cfg.is_oidc_mode = True
    cfg.is_open_mode = False
    cfg.oidc = Mock()
    cfg.jwt = Mock()
    return cfg


@pytest.fixture()
def jwt_handler():
    handler = Mock()
    handler.create_token.return_value = "jwt-token"
    handler.expiry_seconds = 3600
    handler.validate_token.return_value = UserContext(
        user_id="user-1",
        is_authenticated=True,
    )
    return handler


@pytest.fixture()
def auth_result_success():
    result = Mock()
    result.success = True
    result.user_id = "user-1"
    result.error = None
    result.error_description = None
    return result


@pytest.fixture()
def auth_result_failure():
    result = Mock()
    result.success = False
    result.user_id = None
    result.error = "invalid"
    result.error_description = "bad credentials"
    return result


def _build_service_with_mocks(config, jwt_handler, provider):
    """
    Create AuthService while mocking imported collaborators in api.auth.service:
    - JWTHandler
    - LocalAuthProvider / OIDCAuthProvider (via _create_provider)
    """
    with patch("api.auth.service.JWTHandler", return_value=jwt_handler):
        # Let __init__ run, then overwrite provider if needed.
        service = AuthService(config)
    service._provider = provider
    service._jwt_handler = jwt_handler
    return service


# ------------------------------------------------------------------
# AuthenticationError helpers
# ------------------------------------------------------------------

def test_authentication_error_to_auth_error():
    err = AuthenticationError(error="e1", description="d1")
    auth_err = err.to_auth_error()

    assert auth_err.error == "e1"
    assert auth_err.error_description == "d1"


# ------------------------------------------------------------------
# Provider selection
# ------------------------------------------------------------------

def test_create_local_provider(config_local, jwt_handler):
    with patch("api.auth.service.JWTHandler", return_value=jwt_handler):
        with patch("api.auth.service.LocalAuthProvider") as provider_cls:
            AuthService(config_local)
            provider_cls.assert_called_once_with("secret")


def test_create_oidc_provider(config_oidc, jwt_handler):
    with patch("api.auth.service.JWTHandler", return_value=jwt_handler):
        with patch("api.auth.service.OIDCAuthProvider") as provider_cls:
            AuthService(config_oidc)
            provider_cls.assert_called_once_with(config_oidc.oidc)


# ------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------

def test_properties(config_local, jwt_handler):
    provider = Mock()
    provider.provider_name = "local"
    provider.requires_credentials = True

    service = _build_service_with_mocks(config_local, jwt_handler, provider)

    assert service.provider_name == "local"
    assert service.requires_credentials is True
    assert service.is_open_mode is False


# ------------------------------------------------------------------
# authenticate()
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_authenticate_success_local(config_local, jwt_handler, auth_result_success):
    provider = Mock()
    provider.validate = AsyncMock(return_value=auth_result_success)

    service = _build_service_with_mocks(config_local, jwt_handler, provider)

    request = TokenRequest(app_secret="secret", username="user")

    response = await service.authenticate(request)

    assert isinstance(response, TokenResponse)
    assert response.access_token == "jwt-token"
    assert response.token_type == "bearer"
    assert response.expires_in == 3600


@pytest.mark.asyncio
async def test_authenticate_success_oidc(config_oidc, jwt_handler, auth_result_success):
    provider = Mock()
    provider.validate = AsyncMock(return_value=auth_result_success)

    service = _build_service_with_mocks(config_oidc, jwt_handler, provider)

    request = TokenRequest(identity_token="id-token")

    response = await service.authenticate(request)

    assert response.access_token == "jwt-token"


@pytest.mark.asyncio
async def test_authenticate_provider_failure_with_error(config_local, jwt_handler, auth_result_failure):
    provider = Mock()
    provider.validate = AsyncMock(return_value=auth_result_failure)

    service = _build_service_with_mocks(config_local, jwt_handler, provider)

    request = TokenRequest(app_secret="bad")

    with pytest.raises(AuthenticationError) as exc:
        await service.authenticate(request)

    assert exc.value.error == "invalid"
    assert exc.value.description == "bad credentials"


@pytest.mark.asyncio
async def test_authenticate_provider_failure_default_error(config_local, jwt_handler):
    # Covers: result.error is None -> uses "authentication_failed"
    result = Mock()
    result.success = False
    result.user_id = None
    result.error = None
    result.error_description = "nope"

    provider = Mock()
    provider.validate = AsyncMock(return_value=result)

    service = _build_service_with_mocks(config_local, jwt_handler, provider)

    request = TokenRequest(app_secret="bad")

    with pytest.raises(AuthenticationError) as exc:
        await service.authenticate(request)

    assert exc.value.error == "authentication_failed"
    assert exc.value.description == "nope"


@pytest.mark.asyncio
async def test_authenticate_jwt_error(config_local, auth_result_success):
    provider = Mock()
    provider.validate = AsyncMock(return_value=auth_result_success)

    jwt_handler = Mock()
    jwt_handler.create_token.side_effect = JWTError(
        error="jwt_failed",
        description="jwt error",
    )

    service = _build_service_with_mocks(config_local, jwt_handler, provider)

    request = TokenRequest(app_secret="secret")

    with pytest.raises(AuthenticationError) as exc:
        await service.authenticate(request)

    assert exc.value.error == "jwt_failed"
    assert exc.value.description == "jwt error"


# ------------------------------------------------------------------
# validate_token()
# ------------------------------------------------------------------

def test_validate_token_success(config_local, jwt_handler):
    provider = Mock()
    service = _build_service_with_mocks(config_local, jwt_handler, provider)

    result = service.validate_token("token")

    assert result.user_id == "user-1"
    assert result.is_authenticated is True


def test_validate_token_failure(config_local):
    provider = Mock()

    jwt_handler = Mock()
    jwt_handler.validate_token.side_effect = JWTError(
        error="invalid",
        description="bad token",
    )

    service = _build_service_with_mocks(config_local, jwt_handler, provider)

    with pytest.raises(AuthenticationError) as exc:
        service.validate_token("bad")

    assert exc.value.error == "invalid"
    assert exc.value.description == "bad token"


# ------------------------------------------------------------------
# get_open_mode_context()
# ------------------------------------------------------------------

def test_get_open_mode_context_with_username(config_local, jwt_handler):
    provider = Mock()
    service = _build_service_with_mocks(config_local, jwt_handler, provider)

    ctx = service.get_open_mode_context(username="user-x")

    assert ctx.user_id == "user-x"
    assert ctx.is_authenticated is False


def test_get_open_mode_context_without_username(config_local, jwt_handler):
    provider = Mock()
    service = _build_service_with_mocks(config_local, jwt_handler, provider)

    ctx = service.get_open_mode_context()

    assert ctx.user_id.startswith("anonymous_")
    assert ctx.is_authenticated is False
