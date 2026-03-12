"""Tests for ApiToolConfig model validators."""

import pytest

from practorflow.llm.tools.api.models.models import ApiToolConfig


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


# validate_name


def test_validate_name_valid():
    config = _make_config(name="valid_tool_name")
    assert config.name == "valid_tool_name"


def test_validate_name_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        _make_config(name="")


def test_validate_name_whitespace_only_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        _make_config(name="   ")


def test_validate_name_special_chars_raises():
    with pytest.raises(ValueError, match="alphanumeric with underscores"):
        _make_config(name="bad-name!")


def test_validate_name_hyphen_raises():
    with pytest.raises(ValueError, match="alphanumeric with underscores"):
        _make_config(name="bad-name")


# validate_base_url


def test_validate_base_url_valid_https():
    config = _make_config(base_url="https://api.example.com")
    assert config.base_url == "https://api.example.com"


def test_validate_base_url_valid_http():
    config = _make_config(base_url="http://api.example.com")
    assert config.base_url == "http://api.example.com"


def test_validate_base_url_strips_trailing_slash():
    config = _make_config(base_url="https://api.example.com/")
    assert config.base_url == "https://api.example.com"


def test_validate_base_url_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        _make_config(base_url="")


def test_validate_base_url_no_protocol_raises():
    with pytest.raises(ValueError, match="must start with http"):
        _make_config(base_url="api.example.com")


def test_validate_base_url_ftp_raises():
    with pytest.raises(ValueError, match="must start with http"):
        _make_config(base_url="ftp://api.example.com")


# validate_path


def test_validate_path_valid():
    config = _make_config(path="/api/v1/data")
    assert config.path == "/api/v1/data"


def test_validate_path_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        _make_config(path="")


def test_validate_path_no_leading_slash_raises():
    with pytest.raises(ValueError, match="must start with /"):
        _make_config(path="api/data")


def test_validate_path_whitespace_only_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        _make_config(path="   ")
