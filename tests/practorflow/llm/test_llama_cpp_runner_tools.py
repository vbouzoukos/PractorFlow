"""
Tests for LlamaCppRunner tool/function calling support.
"""

import json

import pytest
from unittest.mock import MagicMock

from practorflow.llm.llama_cpp_runner import LlamaCppRunner
from tests.practorflow.common.fixtures import sample_tool_definitions
from tests.practorflow.llm.common_runner import (
    create_mock_model_handle,
    create_llama_completion_response,
    create_tool_call,
)


class TestLlamaCppRunnerSupportsFunctionCalling:
    """Tests for LlamaCppRunner.supports_function_calling()"""

    def test_supports_function_calling_true_with_tool_in_template(self):
        """supports_function_calling() returns True when chat template contains 'tool'."""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template="{% if tools %}Use tools: {{ tools }}{% endif %}",
        )
        runner = LlamaCppRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_true_with_function_in_template(self):
        """supports_function_calling() returns True when chat template contains 'function'."""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template="{% if functions %}Call function: {{ functions }}{% endif %}",
        )
        runner = LlamaCppRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_true_with_tool_call_tag(self):
        """supports_function_calling() returns True when template has <tool_call> tag."""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template="<tool_call>{{ tool }}</tool_call>",
        )
        runner = LlamaCppRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_false_without_template(self):
        """supports_function_calling() returns False when no tool keywords in template."""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template="Standard chat template without special keywords",
        )
        runner = LlamaCppRunner(handle)

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_false_with_no_metadata(self):
        """supports_function_calling() returns False when model has no metadata."""
        handle = create_mock_model_handle(backend="llama_cpp")
        handle.model.metadata = {}
        runner = LlamaCppRunner(handle)

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_handles_missing_metadata_attribute(self):
        """supports_function_calling() returns False when metadata attribute missing."""
        handle = create_mock_model_handle(backend="llama_cpp")
        del handle.model.metadata
        runner = LlamaCppRunner(handle)

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_handles_exception(self):
        """supports_function_calling() returns False on exception."""
        handle = create_mock_model_handle(backend="llama_cpp")
        handle.model.metadata = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
        runner = LlamaCppRunner(handle)

        assert runner.supports_function_calling() is False


class TestLlamaCppRunnerConvertToolsToLlamaFormat:
    """Tests for LlamaCppRunner._convert_tools_to_llama_format()"""

    @pytest.fixture
    def runner(self):
        """Create a LlamaCppRunner instance for testing."""
        handle = create_mock_model_handle(backend="llama_cpp")
        return LlamaCppRunner(handle)

    def test_convert_tools_openai_format_passthrough(self, runner):
        """Tools already in OpenAI format are passed through unchanged."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = runner._convert_tools_to_llama_format(tools)

        assert result == tools

    def test_convert_tools_pydantic_format(self, runner):
        """Pydantic AI format tools are converted to OpenAI format."""
        tools = [
            {
                "name": "search",
                "description": "Search documents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            }
        ]

        result = runner._convert_tools_to_llama_format(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["description"] == "Search documents"
        assert result[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"

    def test_convert_tools_pydantic_format_with_parameters_json_schema(self, runner):
        """Pydantic AI format with parameters_json_schema is converted."""
        tools = [
            {
                "name": "calculate",
                "description": "Calculate expression",
                "parameters_json_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                },
            }
        ]

        result = runner._convert_tools_to_llama_format(tools)

        assert result[0]["function"]["parameters"]["properties"]["expression"]["type"] == "string"

    def test_convert_tools_unknown_format_skipped(self, runner):
        """Unknown tool formats are skipped with warning."""
        tools = [
            {"unknown_field": "value"},
        ]

        result = runner._convert_tools_to_llama_format(tools)

        assert len(result) == 0

    def test_convert_tools_mixed_formats(self, runner):
        """Mixed format tools are all converted appropriately."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "description": "Tool A",
                    "parameters": {},
                },
            },
            {
                "name": "tool_b",
                "description": "Tool B",
                "parameters": {},
            },
        ]

        result = runner._convert_tools_to_llama_format(tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"


class TestLlamaCppRunnerExtractToolCallsFromResponse:
    """Tests for LlamaCppRunner._extract_tool_calls_from_response()"""

    @pytest.fixture
    def runner(self):
        """Create a LlamaCppRunner instance for testing."""
        handle = create_mock_model_handle(backend="llama_cpp")
        return LlamaCppRunner(handle)

    def test_extract_tool_calls_single_call(self, runner):
        """Extracts single tool call from response."""
        tool_calls = [
            create_tool_call("get_weather", {"location": "Paris"}),
        ]
        response = create_llama_completion_response(content="", tool_calls=tool_calls)

        result = runner._extract_tool_calls_from_response(response)

        assert len(result) == 1
        assert result[0]["tool_name"] == "get_weather"
        assert result[0]["args"]["location"] == "Paris"
        assert "tool_call_id" in result[0]

    def test_extract_tool_calls_multiple_calls(self, runner):
        """Extracts multiple tool calls from response."""
        tool_calls = [
            create_tool_call("get_weather", {"location": "Paris"}, call_id="call_1"),
            create_tool_call("search", {"query": "restaurants"}, call_id="call_2"),
        ]
        response = create_llama_completion_response(content="", tool_calls=tool_calls)

        result = runner._extract_tool_calls_from_response(response)

        assert len(result) == 2
        assert result[0]["tool_name"] == "get_weather"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[1]["tool_name"] == "search"
        assert result[1]["tool_call_id"] == "call_2"

    def test_extract_tool_calls_empty_choices(self, runner):
        """Returns empty list when response has no choices."""
        response = {"choices": []}

        result = runner._extract_tool_calls_from_response(response)

        assert result == []

    def test_extract_tool_calls_no_tool_calls_in_message(self, runner):
        """Returns empty list when message has no tool_calls."""
        response = create_llama_completion_response(content="Regular response")

        result = runner._extract_tool_calls_from_response(response)

        assert result == []

    def test_extract_tool_calls_invalid_json_arguments(self, runner):
        """Handles invalid JSON in arguments gracefully."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "test_tool",
                                    "arguments": "not valid json",
                                },
                            }
                        ]
                    }
                }
            ]
        }

        result = runner._extract_tool_calls_from_response(response)

        assert len(result) == 1
        assert result[0]["tool_name"] == "test_tool"
        assert result[0]["args"] == {}

    def test_extract_tool_calls_preserves_call_id(self, runner):
        """Preserves tool call ID from response."""
        tool_calls = [
            create_tool_call("my_tool", {}, call_id="unique_id_123"),
        ]
        response = create_llama_completion_response(content="", tool_calls=tool_calls)

        result = runner._extract_tool_calls_from_response(response)

        assert result[0]["tool_call_id"] == "unique_id_123"

    def test_extract_tool_calls_generates_id_when_missing(self, runner):
        """Generates tool call ID when not present in response."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "test_tool",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                }
            ]
        }

        result = runner._extract_tool_calls_from_response(response)

        assert "tool_call_id" in result[0]
        assert result[0]["tool_call_id"].startswith("call_")