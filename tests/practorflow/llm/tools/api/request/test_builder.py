"""Tests for RequestBuilder."""

import base64

import pytest

from practorflow.llm.tools.api.models.enums import (
    AuthKeyLocation,
    AuthType,
    BodyContentType,
    ParamLocation,
    ParamType,
)
from practorflow.llm.tools.api.models.models import ApiToolConfig, ToolParameter
from practorflow.llm.tools.api.request.builder import RequestBuilder


def _make_config(**kwargs):
    defaults = {
        "user_id": "u1",
        "name": "my_tool",
        "base_url": "https://api.example.com",
        "path": "/data",
        "description": "A tool",
        "keywords": ["test"],
    }
    defaults.update(kwargs)
    return ApiToolConfig(**defaults)


def _make_param(name, location, default_value=None):
    return ToolParameter(
        name=name,
        location=location,
        type=ParamType.STRING,
        description="",
        default_value=default_value,
    )


# build() integration


def test_build_basic_get_request():
    builder = RequestBuilder()
    config = _make_config()

    result = builder.build(config)

    assert result.url == "https://api.example.com/data"
    assert result.method == "GET"
    assert result.body is None
    assert result.content_type is None


# _build_url


def test_build_url_substitutes_path_params():
    builder = RequestBuilder()
    config = _make_config(
        path="/users/{user_id}/posts",
        parameters=[_make_param("user_id", ParamLocation.PATH)],
    )

    result = builder.build(config, user_id="42")

    assert result.url == "https://api.example.com/users/42/posts"


def test_build_url_trims_trailing_slash_from_base():
    builder = RequestBuilder()
    config = _make_config(base_url="https://api.example.com/", path="/data")

    result = builder.build(config)

    assert result.url == "https://api.example.com/data"


def test_build_url_path_without_leading_slash():
    builder = RequestBuilder()
    config = _make_config(path="/items")

    result = builder.build(config)

    assert result.url.startswith("https://api.example.com/")


# _build_headers


def test_build_headers_from_header_params():
    builder = RequestBuilder()
    config = _make_config(parameters=[_make_param("X-Token", ParamLocation.HEADER)])

    result = builder.build(config, **{"X-Token": "abc123"})

    assert result.headers.get("X-Token") == "abc123"


# _build_body


def test_build_body_none_when_content_type_none():
    builder = RequestBuilder()
    config = _make_config(body_content_type=BodyContentType.NONE)

    result = builder.build(config)

    assert result.body is None
    assert result.content_type is None


def test_build_body_none_when_no_body_params():
    builder = RequestBuilder()
    config = _make_config(body_content_type=BodyContentType.JSON)

    result = builder.build(config)  # No body params provided

    assert result.body is None


def test_build_body_json():
    builder = RequestBuilder()
    config = _make_config(
        body_content_type=BodyContentType.JSON,
        parameters=[_make_param("data", ParamLocation.BODY)],
    )

    result = builder.build(config, data="hello")

    assert result.body == {"data": "hello"}
    assert result.content_type == "application/json"
    assert result.headers.get("Content-Type") == "application/json"


def test_build_body_form():
    builder = RequestBuilder()
    config = _make_config(
        body_content_type=BodyContentType.FORM,
        parameters=[_make_param("field", ParamLocation.BODY)],
    )

    result = builder.build(config, field="val")

    assert result.content_type == "application/x-www-form-urlencoded"
    assert result.body == {"field": "val"}


def test_build_body_multipart():
    builder = RequestBuilder()
    config = _make_config(
        body_content_type=BodyContentType.MULTIPART,
        parameters=[_make_param("file", ParamLocation.BODY)],
    )

    result = builder.build(config, file="data")

    assert result.content_type == "multipart/form-data"


# _inject_auth - NONE


def test_inject_auth_none_no_authorization_header():
    builder = RequestBuilder()
    config = _make_config(auth_type=AuthType.NONE)

    result = builder.build(config)

    assert "Authorization" not in result.headers


# _inject_auth - BEARER


def test_inject_auth_bearer_with_secret():
    builder = RequestBuilder()
    config = _make_config(auth_type=AuthType.BEARER)

    result = builder.build(config, decrypted_secret="mytoken")

    assert result.headers.get("Authorization") == "Bearer mytoken"


def test_inject_auth_bearer_no_secret_skips_header():
    builder = RequestBuilder()
    config = _make_config(auth_type=AuthType.BEARER)

    result = builder.build(config)

    assert "Authorization" not in result.headers


# _inject_auth - BASIC


def test_inject_auth_basic_with_credentials():
    builder = RequestBuilder()
    config = _make_config(auth_type=AuthType.BASIC)

    result = builder.build(config, decrypted_username="user", decrypted_password="pass")

    expected = base64.b64encode(b"user:pass").decode("utf-8")
    assert result.headers.get("Authorization") == f"Basic {expected}"


def test_inject_auth_basic_missing_credentials_skips_header():
    builder = RequestBuilder()
    config = _make_config(auth_type=AuthType.BASIC)

    result = builder.build(config)

    assert "Authorization" not in result.headers


# _inject_auth - API_KEY


def test_inject_auth_api_key_in_header():
    builder = RequestBuilder()
    config = _make_config(
        auth_type=AuthType.API_KEY,
        auth_key_location=AuthKeyLocation.HEADER,
        auth_key_name="X-API-Key",
    )

    result = builder.build(config, decrypted_secret="apikey123")

    assert result.headers.get("X-API-Key") == "apikey123"


def test_inject_auth_api_key_in_query():
    builder = RequestBuilder()
    config = _make_config(
        auth_type=AuthType.API_KEY,
        auth_key_location=AuthKeyLocation.QUERY,
        auth_key_name="apikey",
    )

    result = builder.build(config, decrypted_secret="qparam_key")

    assert result.query_params.get("apikey") == "qparam_key"


def test_inject_auth_api_key_no_secret_skips():
    builder = RequestBuilder()
    config = _make_config(
        auth_type=AuthType.API_KEY,
        auth_key_location=AuthKeyLocation.HEADER,
        auth_key_name="X-API-Key",
    )

    result = builder.build(config)  # No decrypted_secret

    assert "X-API-Key" not in result.headers


def test_inject_auth_api_key_no_key_name_skips():
    builder = RequestBuilder()
    config = _make_config(
        auth_type=AuthType.API_KEY,
        auth_key_location=AuthKeyLocation.HEADER,
        auth_key_name=None,
    )

    result = builder.build(config, decrypted_secret="key")

    assert "Authorization" not in result.headers


# _separate_params_by_location


def test_separate_params_uses_default_value():
    builder = RequestBuilder()
    config = _make_config(
        parameters=[_make_param("format", ParamLocation.QUERY, default_value="json")],
    )

    result = builder.build(config)

    assert result.query_params.get("format") == "json"


def test_separate_params_explicit_value_overrides_default():
    builder = RequestBuilder()
    config = _make_config(
        parameters=[_make_param("q", ParamLocation.QUERY, default_value="default")],
    )

    result = builder.build(config, q="custom")

    assert result.query_params.get("q") == "custom"


def test_separate_params_missing_param_without_default_skipped():
    builder = RequestBuilder()
    config = _make_config(
        parameters=[_make_param("optional_param", ParamLocation.QUERY)],
    )

    result = builder.build(config)  # param not provided and no default

    assert "optional_param" not in result.query_params
