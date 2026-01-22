import pytest
from unittest.mock import patch

from api.auth.providers.local import LocalAuthProvider
from api.auth.providers.base import AuthResult


# ------------------------------------------------------------------
# Constructor + properties
# ------------------------------------------------------------------

def test_provider_properties_secure_mode():
    provider = LocalAuthProvider(app_secret="secret")

    assert provider.provider_name == "local"
    assert provider.requires_credentials is True
    assert provider.is_open_mode is False


def test_provider_properties_open_mode():
    provider = LocalAuthProvider(app_secret="")

    assert provider.provider_name == "local"
    assert provider.requires_credentials is False
    assert provider.is_open_mode is True


# ------------------------------------------------------------------
# validate() – open mode
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_open_mode_with_username():
    provider = LocalAuthProvider(app_secret="")

    result = await provider.validate(
        {"username": "user-1"}
    )

    assert isinstance(result, AuthResult)
    assert result.success is True
    assert result.user_id == "user-1"


@pytest.mark.asyncio
async def test_validate_open_mode_without_username():
    provider = LocalAuthProvider(app_secret="")

    with patch.object(provider, "_generate_anonymous_id", return_value="anon-id"):
        result = await provider.validate({})

    assert result.success is True
    assert result.user_id == "anon-id"


# ------------------------------------------------------------------
# validate() – secure mode
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_secure_mode_missing_secret():
    provider = LocalAuthProvider(app_secret="secret")

    result = await provider.validate({})

    assert result.success is False
    assert result.error == "missing_credentials"
    assert "required" in result.error_description


@pytest.mark.asyncio
async def test_validate_secure_mode_invalid_secret():
    provider = LocalAuthProvider(app_secret="secret")

    with patch("api.auth.providers.local.secrets.compare_digest", return_value=False):
        result = await provider.validate(
            {"app_secret": "wrong"}
        )

    assert result.success is False
    assert result.error == "invalid_credentials"
    assert "Invalid" in result.error_description


@pytest.mark.asyncio
async def test_validate_secure_mode_valid_secret_with_username():
    provider = LocalAuthProvider(app_secret="secret")

    with patch("api.auth.providers.local.secrets.compare_digest", return_value=True):
        result = await provider.validate(
            {"app_secret": "secret", "username": "user-x"}
        )

    assert result.success is True
    assert result.user_id == "user-x"


@pytest.mark.asyncio
async def test_validate_secure_mode_valid_secret_without_username():
    provider = LocalAuthProvider(app_secret="secret")

    with patch("api.auth.providers.local.secrets.compare_digest", return_value=True):
        result = await provider.validate(
            {"app_secret": "secret"}
        )

    assert result.success is True
    assert result.user_id == "local_user"


# ------------------------------------------------------------------
# _generate_anonymous_id()
# ------------------------------------------------------------------

def test_generate_anonymous_id():
    provider = LocalAuthProvider(app_secret="")

    anon_id = provider._generate_anonymous_id()

    assert anon_id.startswith("anonymous_")
    assert len(anon_id) > len("anonymous_")
