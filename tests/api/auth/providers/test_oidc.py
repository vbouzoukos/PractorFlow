import pytest
from unittest.mock import Mock, AsyncMock, patch

import httpx
from jwt.exceptions import InvalidTokenError

from api.auth.providers.oidc import OIDCAuthProvider
from api.auth.providers.base import AuthResult


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def oidc_config():
    cfg = Mock()
    cfg.issuer_url = "https://issuer.example.com/"
    cfg.audience = "aud"
    cfg.client_id = "client-id"
    return cfg


@pytest.fixture()
def provider(oidc_config):
    return OIDCAuthProvider(oidc_config)


# ------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------

def test_properties(provider):
    assert provider.provider_name == "oidc"
    assert provider.requires_credentials is True


# ------------------------------------------------------------------
# validate() – missing token
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_missing_identity_token(provider):
    result = await provider.validate({})

    assert result.success is False
    assert result.error == "missing_credentials"


# ------------------------------------------------------------------
# validate() – missing jwks_uri
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_missing_jwks_uri(provider):
    provider._oidc_config = {}  # cached but invalid

    result = await provider.validate({"identity_token": "token"})

    assert result.success is False
    assert result.error == "provider_error"
    assert "JWKS URI" in result.error_description


# ------------------------------------------------------------------
# validate() – success path
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_success(provider):
    provider._oidc_config = {"jwks_uri": "https://jwks"}

    signing_key = Mock()
    signing_key.key = "key"

    jwks_client = Mock()
    jwks_client.get_signing_key_from_jwt.return_value = signing_key

    payload = {
        "sub": "user-1",
        "preferred_username": "user-name",
        "iss": "https://issuer.example.com",
        "iat": 1,
        "exp": 2,
    }

    with patch("api.auth.providers.oidc.PyJWKClient", return_value=jwks_client):
        with patch("api.auth.providers.oidc.jwt.decode", return_value=payload):
            result = await provider.validate({"identity_token": "token"})

    assert result.success is True
    assert result.user_id == "user-name"


# ------------------------------------------------------------------
# validate() – InvalidTokenError
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_invalid_token(provider):
    provider._oidc_config = {"jwks_uri": "https://jwks"}

    jwks_client = Mock()
    jwks_client.get_signing_key_from_jwt.side_effect = InvalidTokenError("bad")

    with patch("api.auth.providers.oidc.PyJWKClient", return_value=jwks_client):
        result = await provider.validate({"identity_token": "token"})

    assert result.success is False
    assert result.error == "invalid_token"
    assert "bad" in result.error_description


# ------------------------------------------------------------------
# validate() – HTTP error
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_http_error(provider):
    provider._oidc_config = None

    async def raise_http(*args, **kwargs):
        raise httpx.HTTPError("down")

    with patch.object(provider, "_fetch_oidc_config", side_effect=raise_http):
        result = await provider.validate({"identity_token": "token"})

    assert result.success is False
    assert result.error == "provider_error"


# ------------------------------------------------------------------
# validate() – generic exception
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_generic_exception(provider):
    provider._oidc_config = {"jwks_uri": "https://jwks"}

    with patch("api.auth.providers.oidc.PyJWKClient", side_effect=Exception("boom")):
        result = await provider.validate({"identity_token": "token"})

    assert result.success is False
    assert result.error == "validation_error"
    assert "boom" in result.error_description


# ------------------------------------------------------------------
# _fetch_oidc_config
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_oidc_config(provider):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"jwks_uri": "https://jwks"}

    client = AsyncMock()
    client.get.return_value = response

    with patch("api.auth.providers.oidc.httpx.AsyncClient") as client_cls:
        client_cls.return_value.__aenter__.return_value = client
        await provider._fetch_oidc_config()

    assert provider._oidc_config["jwks_uri"] == "https://jwks"


# ------------------------------------------------------------------
# _extract_user_id
# ------------------------------------------------------------------

def test_extract_user_id_preferred_username(provider):
    payload = {
        "preferred_username": "user",
        "sub": "fallback",
    }

    assert provider._extract_user_id(payload) == "user"


def test_extract_user_id_email(provider):
    payload = {
        "email": "user@example.com",
        "sub": "fallback",
    }

    assert provider._extract_user_id(payload) == "user@example.com"


def test_extract_user_id_fallback_branch(provider):
    # Ensure the loop doesn't return (all claims present but falsy),
    # so we hit: return str(payload["sub"])
    payload = {
        "preferred_username": None,
        "email": "",
        "sub": "",
    }

    assert provider._extract_user_id(payload) == ""


# ------------------------------------------------------------------
# clear_cache
# ------------------------------------------------------------------

def test_clear_cache(provider):
    provider._oidc_config = {"x": 1}
    provider._jwks_client = Mock()

    provider.clear_cache()

    assert provider._oidc_config is None
    assert provider._jwks_client is None
