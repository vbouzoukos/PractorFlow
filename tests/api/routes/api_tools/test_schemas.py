"""Tests for api_tools schemas."""

import pytest
from datetime import datetime, timezone

from api.routes.api_tools.schemas import (
    ApiToolCreateRequest,
    ApiToolResponse,
    ApiToolSecretsResponse,
    ApiToolListResponse,
    ToolParameterSchema,
)
from practorflow.llm.tools.api.models.models import ApiToolConfig
from practorflow.llm.tools.api.models.enums import AuthType, AuthKeyLocation, ParamLocation


def _make_config(**kwargs):
    defaults = {
        "user_id": "user1",
        "name": "my_tool",
        "base_url": "https://example.com",
        "path": "/api/data",
        "description": "A test tool",
        "keywords": ["test"],
    }
    defaults.update(kwargs)
    return ApiToolConfig(**defaults)


# ---------------------------------------------------------------------------
# ApiToolCreateRequest validation
# ---------------------------------------------------------------------------


def test_create_request_valid_no_auth():
    req = ApiToolCreateRequest(
        name="my_tool",
        base_url="https://example.com",
        path="/data",
        description="A tool",
        keywords=["kw"],
    )
    assert req.name == "my_tool"


def test_create_request_api_key_valid():
    req = ApiToolCreateRequest(
        name="api_key_tool",
        base_url="https://example.com",
        path="/data",
        description="A tool",
        keywords=["kw"],
        auth_type=AuthType.API_KEY,
        auth_secret="secret",
        auth_key_name="X-API-Key",
        auth_key_location=AuthKeyLocation.HEADER,
    )
    assert req.auth_type == AuthType.API_KEY


def test_create_request_api_key_missing_secret_raises():
    with pytest.raises(Exception, match="auth_secret is required"):
        ApiToolCreateRequest(
            name="t",
            base_url="https://example.com",
            path="/data",
            description="d",
            keywords=["k"],
            auth_type=AuthType.API_KEY,
            auth_key_name="X-API-Key",
            auth_key_location=AuthKeyLocation.HEADER,
        )


def test_create_request_api_key_missing_key_name_raises():
    with pytest.raises(Exception, match="auth_key_name is required"):
        ApiToolCreateRequest(
            name="t",
            base_url="https://example.com",
            path="/data",
            description="d",
            keywords=["k"],
            auth_type=AuthType.API_KEY,
            auth_secret="secret",
            auth_key_location=AuthKeyLocation.HEADER,
        )


def test_create_request_api_key_missing_key_location_raises():
    with pytest.raises(Exception, match="auth_key_location is required"):
        ApiToolCreateRequest(
            name="t",
            base_url="https://example.com",
            path="/data",
            description="d",
            keywords=["k"],
            auth_type=AuthType.API_KEY,
            auth_secret="secret",
            auth_key_name="X-API-Key",
        )


def test_create_request_bearer_missing_secret_raises():
    with pytest.raises(Exception, match="auth_secret is required"):
        ApiToolCreateRequest(
            name="t",
            base_url="https://example.com",
            path="/data",
            description="d",
            keywords=["k"],
            auth_type=AuthType.BEARER,
        )


def test_create_request_basic_missing_username_raises():
    with pytest.raises(Exception, match="auth_username is required"):
        ApiToolCreateRequest(
            name="t",
            base_url="https://example.com",
            path="/data",
            description="d",
            keywords=["k"],
            auth_type=AuthType.BASIC,
            auth_password="pass",
        )


def test_create_request_basic_missing_password_raises():
    with pytest.raises(Exception, match="auth_password is required"):
        ApiToolCreateRequest(
            name="t",
            base_url="https://example.com",
            path="/data",
            description="d",
            keywords=["k"],
            auth_type=AuthType.BASIC,
            auth_username="user",
        )


# ---------------------------------------------------------------------------
# ApiToolResponse.mask_secret
# ---------------------------------------------------------------------------


def test_mask_secret_returns_stars_when_present():
    assert ApiToolResponse.mask_secret("any-value") == "***"


def test_mask_secret_returns_none_when_absent():
    assert ApiToolResponse.mask_secret(None) is None


def test_mask_secret_returns_none_for_empty_string():
    assert ApiToolResponse.mask_secret("") is None


# ---------------------------------------------------------------------------
# ApiToolResponse.from_config
# ---------------------------------------------------------------------------


def test_from_config_masks_secret():
    config = _make_config(auth_secret="my-encrypted-secret")
    resp = ApiToolResponse.from_config(config)
    assert resp.auth_secret == "***"


def test_from_config_secret_none_when_not_set():
    config = _make_config()
    resp = ApiToolResponse.from_config(config)
    assert resp.auth_secret is None
    assert resp.auth_username is None
    assert resp.auth_password is None


def test_from_config_preserves_metadata():
    config = _make_config(description="desc", keywords=["kw"])
    resp = ApiToolResponse.from_config(config)
    assert resp.description == "desc"
    assert resp.keywords == ["kw"]
    assert resp.name == "my_tool"


# ---------------------------------------------------------------------------
# ApiToolSecretsResponse
# ---------------------------------------------------------------------------


def test_secrets_response_fields():
    resp = ApiToolSecretsResponse(
        tool_id="t1",
        auth_secret="plain",
        auth_username=None,
        auth_password=None,
    )
    assert resp.tool_id == "t1"
    assert resp.auth_secret == "plain"


# ---------------------------------------------------------------------------
# ApiToolListResponse
# ---------------------------------------------------------------------------


def test_list_response_count():
    config = _make_config()
    tool_resp = ApiToolResponse.from_config(config)
    resp = ApiToolListResponse(tools=[tool_resp], count=1)
    assert resp.count == 1
    assert len(resp.tools) == 1
