import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from unittest.mock import Mock

from api.auth.dependencies import (
    set_auth_service,
    get_auth_service,
    get_current_user,
    require_authenticated_user,
    get_optional_user,
)
from api.auth.service import AuthenticationError
from api.auth.schemas import UserContext


# ------------------------------------------------------------------
# set_auth_service / get_auth_service
# ------------------------------------------------------------------

def test_get_auth_service_not_initialized():
    with pytest.raises(RuntimeError):
        get_auth_service()


def test_set_and_get_auth_service():
    service = Mock()
    set_auth_service(service)

    assert get_auth_service() is service


# ------------------------------------------------------------------
# get_current_user
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_current_user_open_mode_no_token():
    service = Mock()
    service.is_open_mode = True
    service.get_open_mode_context.return_value = UserContext(
        user_id="anon",
        is_authenticated=False,
    )

    result = await get_current_user(
        credentials=None,
        auth_service=service,
    )

    assert result.user_id == "anon"
    assert result.is_authenticated is False


@pytest.mark.asyncio
async def test_get_current_user_secure_mode_no_token():
    service = Mock()
    service.is_open_mode = False

    with pytest.raises(HTTPException) as exc:
        await get_current_user(
            credentials=None,
            auth_service=service,
        )

    assert exc.value.status_code == 401
    assert exc.value.detail == "Authentication required"


@pytest.mark.asyncio
async def test_get_current_user_valid_token():
    service = Mock()
    service.validate_token.return_value = UserContext(
        user_id="user-1",
        is_authenticated=True,
    )

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="token",
    )

    result = await get_current_user(
        credentials=credentials,
        auth_service=service,
    )

    assert result.user_id == "user-1"


@pytest.mark.asyncio
async def test_get_current_user_invalid_token():
    service = Mock()
    service.validate_token.side_effect = AuthenticationError(
        error="invalid_token",
        description="bad token",
    )

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="bad",
    )

    with pytest.raises(HTTPException) as exc:
        await get_current_user(
            credentials=credentials,
            auth_service=service,
        )

    assert exc.value.status_code == 401
    assert exc.value.detail == "bad token"


# ------------------------------------------------------------------
# require_authenticated_user
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_require_authenticated_user_success():
    service = Mock()
    service.validate_token.return_value = UserContext(
        user_id="user-2",
        is_authenticated=True,
    )

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="token",
    )

    result = await require_authenticated_user(
        credentials=credentials,
        auth_service=service,
    )

    assert result.user_id == "user-2"


@pytest.mark.asyncio
async def test_require_authenticated_user_invalid():
    service = Mock()
    service.validate_token.side_effect = AuthenticationError(
        error="expired",
        description="expired",
    )

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="expired",
    )

    with pytest.raises(HTTPException) as exc:
        await require_authenticated_user(
            credentials=credentials,
            auth_service=service,
        )

    assert exc.value.status_code == 401
    assert exc.value.detail == "expired"


# ------------------------------------------------------------------
# get_optional_user
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_optional_user_no_token():
    service = Mock()

    result = await get_optional_user(
        credentials=None,
        auth_service=service,
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_optional_user_valid_token():
    service = Mock()
    service.validate_token.return_value = UserContext(
        user_id="user-3",
        is_authenticated=True,
    )

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="token",
    )

    result = await get_optional_user(
        credentials=credentials,
        auth_service=service,
    )

    assert result.user_id == "user-3"


@pytest.mark.asyncio
async def test_get_optional_user_invalid_token():
    service = Mock()
    service.validate_token.side_effect = AuthenticationError(
        error="invalid",
        description="invalid",
    )

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="bad",
    )

    result = await get_optional_user(
        credentials=credentials,
        auth_service=service,
    )

    assert result is None
