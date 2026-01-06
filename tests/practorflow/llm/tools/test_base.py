"""
Unit tests for base tool classes.

Tests cover ToolParameter, ToolResult, and BaseTool for 100% code coverage.
"""

import pytest
from typing import List

from practorflow.llm.tools.base import BaseTool, ToolParameter, ToolResult


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_init_required_fields(self):
        """ToolParameter initializes with required fields."""
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
        )

        assert param.name == "query"
        assert param.type == "string"
        assert param.description == "Search query"

    def test_init_default_values(self):
        """ToolParameter uses default values for optional fields."""
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
        )

        assert param.required is True
        assert param.default is None
        assert param.enum is None

    def test_init_custom_required(self):
        """ToolParameter accepts custom required value."""
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Max results",
            required=False,
        )

        assert param.required is False

    def test_init_custom_default(self):
        """ToolParameter accepts custom default value."""
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Max results",
            default=10,
        )

        assert param.default == 10

    def test_init_with_enum(self):
        """ToolParameter accepts enum values."""
        param = ToolParameter(
            name="format",
            type="string",
            description="Output format",
            enum=["json", "xml", "csv"],
        )

        assert param.enum == ["json", "xml", "csv"]


class TestToolResultToContextString:
    """Tests for ToolResult.to_context_string()"""

    def test_to_context_string_failed_result(self):
        """to_context_string returns error message for failed result."""
        result = ToolResult(success=False, error="Connection timeout")

        context = result.to_context_string()

        assert "Tool execution failed" in context
        assert "Connection timeout" in context

    def test_to_context_string_string_data(self):
        """to_context_string returns string data directly."""
        result = ToolResult(success=True, data="Simple string result")

        context = result.to_context_string()

        assert context == "Simple string result"

    def test_to_context_string_dict_data(self):
        """to_context_string formats dict data."""
        result = ToolResult(success=True, data={"key": "value", "count": 42})

        context = result.to_context_string()

        assert "key: value" in context
        assert "count: 42" in context

    def test_to_context_string_list_data(self):
        """to_context_string formats list data."""
        result = ToolResult(success=True, data=["item1", "item2", "item3"])

        context = result.to_context_string()

        assert "1. item1" in context
        assert "2. item2" in context
        assert "3. item3" in context

    def test_to_context_string_other_data(self):
        """to_context_string converts other types to string."""
        result = ToolResult(success=True, data=12345)

        context = result.to_context_string()

        assert context == "12345"

    def test_to_context_string_none_data_success(self):
        """to_context_string handles None data with success."""
        result = ToolResult(success=True, data=None)

        context = result.to_context_string()

        assert context == "None"


class TestToolResultFormatDict:
    """Tests for ToolResult._format_dict()"""

    def test_format_dict_simple(self):
        """_format_dict formats simple key-value pairs."""
        result = ToolResult(success=True)

        formatted = result._format_dict({"name": "test", "value": 123})

        assert "name: test" in formatted
        assert "value: 123" in formatted

    def test_format_dict_nested_dict(self):
        """_format_dict handles nested dictionaries."""
        result = ToolResult(success=True)
        data = {
            "outer": {
                "inner": "nested value"
            }
        }

        formatted = result._format_dict(data)

        assert "outer:" in formatted
        assert "inner: nested value" in formatted

    def test_format_dict_nested_list(self):
        """_format_dict handles nested lists."""
        result = ToolResult(success=True)
        data = {
            "items": ["a", "b", "c"]
        }

        formatted = result._format_dict(data)

        assert "items:" in formatted
        assert "1. a" in formatted
        assert "2. b" in formatted

    def test_format_dict_with_indent(self):
        """_format_dict applies correct indentation."""
        result = ToolResult(success=True)
        data = {"key": "value"}

        formatted = result._format_dict(data, indent=2)

        assert formatted.startswith("    ")  # 2 * 2 spaces


class TestToolResultFormatList:
    """Tests for ToolResult._format_list()"""

    def test_format_list_simple(self):
        """_format_list formats simple items."""
        result = ToolResult(success=True)

        formatted = result._format_list(["first", "second"])

        assert "1. first" in formatted
        assert "2. second" in formatted

    def test_format_list_with_dict_items(self):
        """_format_list handles dict items in list."""
        result = ToolResult(success=True)
        data = [
            {"name": "item1", "value": 100},
            {"name": "item2", "value": 200},
        ]

        formatted = result._format_list(data)

        assert "1." in formatted
        assert "name: item1" in formatted
        assert "2." in formatted
        assert "name: item2" in formatted

    def test_format_list_with_indent(self):
        """_format_list applies correct indentation."""
        result = ToolResult(success=True)

        formatted = result._format_list(["item"], indent=1)

        assert formatted.startswith("  ")  # 1 * 2 spaces


class TestToolResultMetadata:
    """Tests for ToolResult metadata field."""

    def test_metadata_default_empty_dict(self):
        """metadata defaults to empty dict."""
        result = ToolResult(success=True)

        assert result.metadata == {}

    def test_metadata_custom_value(self):
        """metadata accepts custom dict."""
        result = ToolResult(
            success=True,
            metadata={"query": "test", "count": 5},
        )

        assert result.metadata["query"] == "test"
        assert result.metadata["count"] == 5


# Concrete implementation for testing BaseTool
class ConcreteTool(BaseTool):
    """Concrete tool implementation for testing BaseTool methods."""

    def __init__(self, params: List[ToolParameter] = None):
        self._params = params or [
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Max results",
                required=False,
                default=10,
            ),
        ]
        self._execute_result = ToolResult(success=True, data="executed")

    @property
    def name(self) -> str:
        return "concrete_tool"

    @property
    def description(self) -> str:
        return "A concrete tool for testing"

    @property
    def parameters(self) -> List[ToolParameter]:
        return self._params

    def execute(self, **kwargs) -> ToolResult:
        return self._execute_result

    def set_execute_result(self, result: ToolResult):
        self._execute_result = result


class TestBaseToolValidateParameters:
    """Tests for BaseTool.validate_parameters()"""

    def test_validate_parameters_valid(self):
        """validate_parameters returns None for valid parameters."""
        tool = ConcreteTool()

        error = tool.validate_parameters(query="test")

        assert error is None

    def test_validate_parameters_missing_required(self):
        """validate_parameters returns error for missing required parameter."""
        tool = ConcreteTool()

        error = tool.validate_parameters()

        assert error is not None
        assert "Missing required parameter" in error
        assert "query" in error

    def test_validate_parameters_required_with_default(self):
        """validate_parameters allows missing required param with default."""
        tool = ConcreteTool(params=[
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True,
                default="default_value",
            ),
        ])

        error = tool.validate_parameters()

        assert error is None

    def test_validate_parameters_unknown_parameter(self):
        """validate_parameters returns error for unknown parameters."""
        tool = ConcreteTool()

        error = tool.validate_parameters(query="test", unknown_param="value")

        assert error is not None
        assert "Unknown parameters" in error
        assert "unknown_param" in error

    def test_validate_parameters_multiple_unknown(self):
        """validate_parameters lists all unknown parameters."""
        tool = ConcreteTool()

        error = tool.validate_parameters(query="test", foo="bar", baz="qux")

        assert error is not None
        assert "foo" in error
        assert "baz" in error


class TestBaseToolGetSchema:
    """Tests for BaseTool.get_schema()"""

    def test_get_schema_structure(self):
        """get_schema returns correct structure."""
        tool = ConcreteTool()

        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "concrete_tool"
        assert schema["function"]["description"] == "A concrete tool for testing"
        assert "parameters" in schema["function"]

    def test_get_schema_properties(self):
        """get_schema includes parameter properties."""
        tool = ConcreteTool()

        schema = tool.get_schema()
        properties = schema["function"]["parameters"]["properties"]

        assert "query" in properties
        assert properties["query"]["type"] == "string"
        assert "limit" in properties
        assert properties["limit"]["type"] == "integer"

    def test_get_schema_required(self):
        """get_schema lists required parameters."""
        tool = ConcreteTool()

        schema = tool.get_schema()
        required = schema["function"]["parameters"]["required"]

        assert "query" in required
        assert "limit" not in required

    def test_get_schema_with_enum(self):
        """get_schema includes enum values when present."""
        tool = ConcreteTool(params=[
            ToolParameter(
                name="format",
                type="string",
                description="Output format",
                required=True,
                enum=["json", "xml", "csv"],
            ),
        ])

        schema = tool.get_schema()
        properties = schema["function"]["parameters"]["properties"]

        assert properties["format"]["enum"] == ["json", "xml", "csv"]

    def test_get_schema_without_enum(self):
        """get_schema omits enum when not present."""
        tool = ConcreteTool()

        schema = tool.get_schema()
        properties = schema["function"]["parameters"]["properties"]

        assert "enum" not in properties["query"]


class TestBaseToolCall:
    """Tests for BaseTool.__call__()"""

    def test_call_executes_with_valid_params(self):
        """__call__ executes tool with valid parameters."""
        tool = ConcreteTool()

        result = tool(query="test")

        assert result.success is True
        assert result.data == "executed"

    def test_call_returns_validation_error(self):
        """__call__ returns error for invalid parameters."""
        tool = ConcreteTool()

        result = tool()  # Missing required 'query'

        assert result.success is False
        assert "Missing required parameter" in result.error

    def test_call_returns_validation_error_unknown_params(self):
        """__call__ returns error for unknown parameters."""
        tool = ConcreteTool()

        result = tool(query="test", invalid_param="value")

        assert result.success is False
        assert "Unknown parameters" in result.error


class TestBaseToolRepr:
    """Tests for BaseTool.__repr__()"""

    def test_repr_format(self):
        """__repr__ returns expected format."""
        tool = ConcreteTool()

        repr_str = repr(tool)

        assert repr_str == "ConcreteTool(name='concrete_tool')"