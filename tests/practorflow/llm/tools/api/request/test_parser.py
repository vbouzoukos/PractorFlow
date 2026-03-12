"""Tests for ResponseParser."""

import pytest
from unittest.mock import MagicMock

from practorflow.llm.tools.api.models.enums import ParseType
from practorflow.llm.tools.api.models.models import ApiToolConfig, OutputMapping, ErrorMapping
from practorflow.llm.tools.api.request.parser import ParsedResponse, ResponseParser


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


def _make_response(status_code=200, json_data=None, text="response", content=b"bytes"):
    response = MagicMock()
    response.status_code = status_code
    if json_data is not None:
        response.json.return_value = json_data
    else:
        response.json.side_effect = ValueError("not json")
    response.text = text
    response.content = content
    return response


# parse() - success path


def test_parse_success_json():
    parser = ResponseParser()
    config = _make_config(parse_as=ParseType.JSON, success_codes=[200])
    response = _make_response(status_code=200, json_data={"key": "val"})

    result = parser.parse(response, config)

    assert result.success is True
    assert result.data == {"key": "val"}
    assert result.error is None
    assert result.status_code == 200


def test_parse_success_with_output_mapping():
    parser = ResponseParser()
    config = _make_config(
        parse_as=ParseType.JSON,
        success_codes=[200],
        output_mapping=[OutputMapping(field_name="value", json_path="$.key")],
    )
    response = _make_response(status_code=200, json_data={"key": "extracted"})

    result = parser.parse(response, config)

    assert result.success is True
    assert result.data == {"value": "extracted"}


def test_parse_failure_returns_http_status_error():
    parser = ResponseParser()
    config = _make_config(success_codes=[200])
    response = _make_response(status_code=404, text="Not found")

    result = parser.parse(response, config)

    assert result.success is False
    assert "404" in result.error
    assert result.data is None


def test_parse_failure_with_error_mapping():
    parser = ResponseParser()
    config = _make_config(
        parse_as=ParseType.JSON,
        success_codes=[200],
        error_mapping=[ErrorMapping(field_name="msg", json_path="$.error")],
    )
    response = _make_response(status_code=500, json_data={"error": "server failed"})

    result = parser.parse(response, config)

    assert result.success is False
    assert "server failed" in result.error


# _parse_content


def test_parse_content_json():
    parser = ResponseParser()
    response = MagicMock()
    response.json.return_value = {"x": 1}

    result = parser._parse_content(response, ParseType.JSON)

    assert result == {"x": 1}


def test_parse_content_json_fallback_to_text():
    parser = ResponseParser()
    response = MagicMock()
    response.json.side_effect = ValueError("not json")
    response.text = "plain text"

    result = parser._parse_content(response, ParseType.JSON)

    assert result == "plain text"


def test_parse_content_text():
    parser = ResponseParser()
    response = MagicMock()
    response.text = "hello"

    result = parser._parse_content(response, ParseType.TEXT)

    assert result == "hello"


def test_parse_content_binary():
    parser = ResponseParser()
    response = MagicMock()
    response.content = b"\x00\x01"

    result = parser._parse_content(response, ParseType.BINARY)

    assert result == b"\x00\x01"


def test_parse_content_unknown_falls_back_to_text():
    parser = ResponseParser()
    response = MagicMock()
    response.text = "fallback"

    # Use a mock parse type that is not handled
    result = parser._parse_content(response, "unknown_type")

    assert result == "fallback"


# _query_jsonpath


def test_query_jsonpath_root():
    parser = ResponseParser()
    assert parser._query_jsonpath({"a": 1}, "$") == {"a": 1}


def test_query_jsonpath_empty_path():
    parser = ResponseParser()
    assert parser._query_jsonpath({"a": 1}, "") == {"a": 1}


def test_query_jsonpath_simple_field():
    parser = ResponseParser()
    assert parser._query_jsonpath({"key": "value"}, "$.key") == "value"


def test_query_jsonpath_nested_field():
    parser = ResponseParser()
    data = {"a": {"b": "deep"}}
    assert parser._query_jsonpath(data, "$.a.b") == "deep"


def test_query_jsonpath_array_index():
    parser = ResponseParser()
    data = {"items": ["a", "b", "c"]}
    assert parser._query_jsonpath(data, "$.items[1]") == "b"


def test_query_jsonpath_array_wildcard():
    parser = ResponseParser()
    data = {"items": [{"name": "x"}, {"name": "y"}]}
    result = parser._query_jsonpath(data, "$.items[*].name")
    assert result == ["x", "y"]


def test_query_jsonpath_missing_field_returns_none():
    parser = ResponseParser()
    assert parser._query_jsonpath({"a": 1}, "$.missing") is None


def test_query_jsonpath_array_index_out_of_range():
    parser = ResponseParser()
    data = {"items": ["only_one"]}
    assert parser._query_jsonpath(data, "$.items[5]") is None


def test_query_jsonpath_on_non_dict_returns_none():
    parser = ResponseParser()
    assert parser._query_jsonpath("a string", "$.field") is None


def test_query_jsonpath_array_field_on_none():
    parser = ResponseParser()
    data = {"items": None}
    assert parser._query_jsonpath(data, "$.items[0]") is None


def test_query_jsonpath_array_field_not_a_list():
    parser = ResponseParser()
    data = {"items": "not a list"}
    assert parser._query_jsonpath(data, "$.items[0]") is None


def test_query_jsonpath_list_field_traversal():
    parser = ResponseParser()
    data = [{"name": "a"}, {"name": "b"}]
    result = parser._query_jsonpath(data, "name")
    assert result == ["a", "b"]


def test_query_jsonpath_nested_none_returns_none():
    parser = ResponseParser()
    data = {"a": None}
    assert parser._query_jsonpath(data, "$.a.b") is None


# _apply_error_mapping


def test_apply_error_mapping_returns_none_when_all_missing():
    parser = ResponseParser()
    mappings = [ErrorMapping(field_name="msg", json_path="$.missing")]
    result = parser._apply_error_mapping({}, mappings)
    assert result is None


def test_apply_error_mapping_joins_multiple_errors():
    parser = ResponseParser()
    data = {"code": "E001", "msg": "bad request"}
    mappings = [
        ErrorMapping(field_name="code", json_path="$.code"),
        ErrorMapping(field_name="msg", json_path="$.msg"),
    ]
    result = parser._apply_error_mapping(data, mappings)
    assert "E001" in result
    assert "bad request" in result
