import json
import pytest

from practorflow.llm.tools.json_transform import JsonTransformTool


class TestJsonTransformTool:
    def test_init_name_description_and_parameters(self):
        tool = JsonTransformTool()

        assert tool._max_input_length == 100000
        assert tool.name == "json_transform"
        assert isinstance(tool.description, str)

        params = tool.parameters
        assert len(params) == 4

        assert params[0].name == "json_data"
        assert params[1].name == "operation"
        assert params[1].enum == ["parse", "query", "extract", "flatten", "keys", "values"]

    def test_parse_json_with_dict_and_list(self):
        tool = JsonTransformTool()

        assert tool._parse_json({"a": 1}) == {"a": 1}
        assert tool._parse_json([1, 2]) == [1, 2]

    def test_parse_json_string_and_length_error(self):
        tool = JsonTransformTool(max_input_length=8)

        with pytest.raises(ValueError):
            tool._parse_json("123456789")

        parsed = tool._parse_json('{"a": 1}')
        assert parsed["a"] == 1

    def test_parse_json_invalid_type(self):
        tool = JsonTransformTool()

        with pytest.raises(ValueError):
            tool._parse_json(123)

    def test_query_jsonpath_basic_and_root(self):
        tool = JsonTransformTool()
        data = {"a": {"b": 2}}

        assert tool._query_jsonpath(data, "$") == data
        assert tool._query_jsonpath(data, "a.b") == 2
        assert tool._query_jsonpath(data, "a.c") is None

    def test_query_jsonpath_array_index_and_out_of_bounds(self):
        tool = JsonTransformTool()
        data = {"items": [{"x": 1}, {"x": 2}]}

        assert tool._query_jsonpath(data, "items[0].x") == 1
        assert tool._query_jsonpath(data, "items[5]") is None

    def test_query_jsonpath_wildcard_and_remaining_path(self):
        tool = JsonTransformTool()
        data = {"items": [{"x": 1}, {"x": 2}]}

        result = tool._query_jsonpath(data, "items[*].x")
        assert result == [1, 2]

        result_no_remaining = tool._query_jsonpath(data, "items[*]")
        assert result_no_remaining == data["items"]

    def test_query_jsonpath_list_fallback_and_invalid_current(self):
        tool = JsonTransformTool()
        data = [{"a": 1}, {"a": 2}]

        assert tool._query_jsonpath(data, "a") == [1, 2]
        assert tool._query_jsonpath(123, "a") is None

    def test_extract_path_and_fields(self):
        tool = JsonTransformTool()
        data = {"user": {"name": "Alice", "age": 30}}

        assert tool._extract_path(data, "user.name") == "Alice"

        fields = tool._extract_fields(data, ["user.name", "user.age"])
        assert fields == {"name": "Alice", "age": 30}

    def test_flatten_dict_and_list_and_scalar(self):
        tool = JsonTransformTool()

        data = {"a": {"b": 1}, "c": [2, 3]}
        flat = tool._flatten(data)

        assert flat["a.b"] == 1
        assert flat["c[0]"] == 2
        assert flat["c[1]"] == 3

        scalar_flat = tool._flatten(5, prefix="x")
        assert scalar_flat == {"x": 5}

    def test_execute_parse(self):
        tool = JsonTransformTool()
        result = tool.execute(json_data='{"a": 1}', operation="parse")

        assert result.success is True
        assert result.data["a"] == 1
        assert result.metadata["operation"] == "parse"

    def test_execute_query_missing_path(self):
        tool = JsonTransformTool()
        result = tool.execute(json_data={"a": 1}, operation="query")

        assert result.success is False
        assert "'path' is required" in result.error

    def test_execute_query_success(self):
        tool = JsonTransformTool()
        result = tool.execute(json_data={"a": {"b": 2}}, operation="query", path="a.b")

        assert result.success is True
        assert result.data == 2

    def test_execute_extract_path_and_fields_and_error(self):
        tool = JsonTransformTool()
        data = {"a": {"b": 2, "c": 3}}

        result_path = tool.execute(json_data=data, operation="extract", path="a.b")
        assert result_path.success is True
        assert result_path.data == 2

        result_fields = tool.execute(
            json_data=data,
            operation="extract",
            fields=["a.b", "a.c"],
        )
        assert result_fields.data == {"b": 2, "c": 3}

        result_error = tool.execute(json_data=data, operation="extract")
        assert result_error.success is False

    def test_execute_flatten(self):
        tool = JsonTransformTool()
        result = tool.execute(json_data={"a": {"b": 1}}, operation="flatten")

        assert result.success is True
        assert result.data == {"a.b": 1}
        assert result.metadata["keys_count"] == 1

    def test_execute_keys_and_values_success_and_error(self):
        tool = JsonTransformTool()

        result_keys = tool.execute(json_data={"a": 1}, operation="keys")
        assert result_keys.success is True
        assert result_keys.data == ["a"]

        result_values = tool.execute(json_data={"a": 1}, operation="values")
        assert result_values.success is True
        assert result_values.data == [1]

        result_keys_error = tool.execute(json_data=[1, 2], operation="keys")
        assert result_keys_error.success is False

        result_values_error = tool.execute(json_data=[1, 2], operation="values")
        assert result_values_error.success is False

    def test_execute_unknown_operation(self):
        tool = JsonTransformTool()
        result = tool.execute(json_data={}, operation="unknown")

        assert result.success is False
        assert "Unknown operation" in result.error

    def test_execute_invalid_json_and_generic_exception(self, monkeypatch):
        tool = JsonTransformTool()

        result_invalid = tool.execute(json_data="{bad json}", operation="parse")
        assert result_invalid.success is False
        assert result_invalid.error.startswith("Invalid JSON:")

        def boom(_):
            raise RuntimeError("boom")

        monkeypatch.setattr(tool, "_parse_json", boom)

        result_exception = tool.execute(json_data={}, operation="parse")
        assert result_exception.success is False
        assert result_exception.error.startswith("JSON transformation failed:")
        assert "boom" in result_exception.error

    def test_query_jsonpath_skips_empty_parts(self):
        tool = JsonTransformTool()

        data = {
            "a": {
                "b": {
                    "c": 42
                }
            }
        }

        # Path with consecutive dots creates empty parts
        result = tool._query_jsonpath(data, "a..b...c")

        assert result == 42

    def test_query_jsonpath_array_match_returns_none_when_current_not_dict(self):
        tool = JsonTransformTool()

        # current is an int, so field lookup can't happen -> hits:
        # if field_name:
        #   if isinstance(current, dict): ...
        #   else: return None
        data = 123
        assert tool._query_jsonpath(data, "items[0]") is None


    def test_query_jsonpath_array_match_returns_none_when_field_missing(self):
        tool = JsonTransformTool()

        # current is dict but missing "items" -> current becomes None -> hits:
        # if current is None: return None
        data = {"other": []}
        assert tool._query_jsonpath(data, "items[0]") is None

    def test_query_jsonpath_array_match_current_not_list_returns_none(self):
        tool = JsonTransformTool()

        # field exists but is NOT a list
        data = {
            "items": {"a": 1}
        }

        result = tool._query_jsonpath(data, "items[0]")

        assert result is None

    def test_flatten_list_with_nested_dict_and_list(self):
        tool = JsonTransformTool()

        data = {
            "items": [
                {"a": 1},
                [2, 3]
            ]
        }

        result = tool._flatten(data)

        assert result["items[0].a"] == 1
        assert result["items[1][0]"] == 2
        assert result["items[1][1]"] == 3
