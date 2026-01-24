import pytest
from unittest.mock import MagicMock, AsyncMock

from pydantic_ai.messages import (
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ModelRequest,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage
from pydantic_ai.tools import ToolDefinition

from practorflow.llm.pyai.model import LocalLLMModel
from practorflow.llm.base.llm_runner import StreamChunk


def make_tool(name: str) -> ToolDefinition:
    tool = MagicMock(spec=ToolDefinition)
    tool.name = name
    tool.description = f"{name} desc"
    tool.parameters_json_schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": [],
    }
    return tool


@pytest.fixture
def runner():
    runner = MagicMock()
    runner.model_name = "test-model"
    return runner


def test_model_properties(runner):
    model = LocalLLMModel(runner)
    assert model.model_name == "local:test-model"
    assert model.system == "local"


def test_build_tool_prompt_with_and_without_tools(runner):
    model = LocalLLMModel(runner)

    assert model._build_tool_prompt([], []) == ""

    tool = make_tool("calc")
    prompt = model._build_tool_prompt([tool], [])

    assert "AVAILABLE TOOLS" in prompt
    assert "calc" in prompt
    assert "IMPORTANT:" in prompt


def test_extract_tool_calls_invalid_json_is_ignored(runner):
    model = LocalLLMModel(runner)
    tool = make_tool("search")

    text = "```json {not valid} ```"

    calls = model._extract_tool_calls(text, [tool])

    assert calls == []


def test_extract_tool_calls_non_dict_args(runner):
    model = LocalLLMModel(runner)
    tool = make_tool("search")

    # Pattern 1 (code block): has "tool" key -> matches.append(obj) hit, args coerced
    text_tool_codeblock = """```json
{"tool": "search", "args": "bad"}
```"""

    calls = model._extract_tool_calls(text_tool_codeblock, [tool])

    assert len(calls) == 1
    assert isinstance(calls[0], ToolCallPart)
    assert calls[0].tool_name == "search"
    assert calls[0].args == {}

    # Pattern 1 (code block): has "name" key -> matches.append(obj) hit
    text_name_codeblock = """```json
{"name": "search", "arguments": {"x": 1}}
```"""

    calls = model._extract_tool_calls(text_name_codeblock, [tool])

    assert len(calls) == 1
    assert calls[0].tool_name == "search"
    assert calls[0].args == {"x": 1}

    # Pattern 1 (code block): has "tool" key but tool not in tool_names -> skipped
    text_unknown_codeblock = """```json
{"tool": "unknown", "args": {"x": 1}}
```"""

    calls = model._extract_tool_calls(text_unknown_codeblock, [tool])

    assert calls == []


def test_clean_text_from_tool_calls(runner):
    model = LocalLLMModel(runner)

    text = """
    before
    ```json
    {"tool": "search", "args": {"q": "x"}}
    ```
    after
    """

    cleaned = model._clean_text_from_tool_calls(text)

    assert "search" not in cleaned
    assert "before" in cleaned
    assert "after" in cleaned


def test_has_tool_results_true_and_false(runner):
    model = LocalLLMModel(runner)

    msg_with_tool = ModelRequest(
        parts=[ToolReturnPart(tool_name="x", content="y")],
        instructions=None,
    )
    msg_without_tool = ModelRequest(parts=[TextPart(content="hi")], instructions=None)

    assert model._has_tool_results([msg_with_tool]) is True
    assert model._has_tool_results([msg_without_tool]) is False


@pytest.mark.asyncio
async def test_request_without_tools(runner, monkeypatch):
    model = LocalLLMModel(runner)

    conversion = MagicMock()
    conversion.system_prompt = "SYS"
    conversion.messages = ["msg"]

    monkeypatch.setattr(
        model._converter,
        "convert_messages",
        MagicMock(return_value=conversion),
    )

    params = ModelRequestParameters(function_tools=None, output_tools=None)

    # ---- Call 1: string reply, non-empty ----
    runner.generate = AsyncMock(
        return_value={
            "reply": "hello",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    )

    resp = await model.request(
        messages=[],
        model_settings=None,
        model_request_parameters=params,
    )

    assert isinstance(resp.parts[0], TextPart)
    assert resp.parts[0].content == "hello"
    assert resp.usage.input_tokens == 1
    assert resp.usage.output_tokens == 2

    # ---- Call 2: reply empty → fallback ----
    runner.generate = AsyncMock(
        return_value={
            "reply": {
                "choices": [
                    {"message": {"content": ""}}
                ]
            },
            "usage": {},
        }
    )

    resp = await model.request(
        messages=[],
        model_settings={"temperature": 0.5, "top_p": 0.8},
        model_request_parameters=params,
    )

    assert len(resp.parts) == 1
    assert isinstance(resp.parts[0], TextPart)
    assert resp.parts[0].content == ""


@pytest.mark.asyncio
async def test_request_with_tool_call(runner):
    runner.generate = AsyncMock(
        return_value={
            "reply": '{"tool": "calc", "args": {"x": 1}}',
            "usage": {},
        }
    )

    model = LocalLLMModel(runner)
    tool = make_tool("calc")

    params = ModelRequestParameters(function_tools=[tool], output_tools=None)

    resp = await model.request(
        messages=[],
        model_settings=None,
        model_request_parameters=params,
    )

    assert isinstance(resp.parts[0], ToolCallPart)
    assert resp.parts[0].tool_name == "calc"


@pytest.mark.asyncio
async def test_request_stream_yields_streamed_response(runner, monkeypatch):
    model = LocalLLMModel(runner)

    # Force conversion.system_prompt → covers system_parts.append(conversion.system_prompt)
    conversion = MagicMock()
    conversion.system_prompt = "SYS"
    conversion.messages = ["msg"]

    monkeypatch.setattr(
        model._converter,
        "convert_messages",
        MagicMock(return_value=conversion),
    )

    # Stub _build_tool_prompt to ensure it is called and appended
    monkeypatch.setattr(
        model,
        "_build_tool_prompt",
        MagicMock(return_value="TOOL PROMPT"),
    )

    # generate_stream is never iterated here, but must exist
    runner.generate_stream = MagicMock()

    # Tool definitions to force has_tools = True
    tool = make_tool("calc")

    params = ModelRequestParameters(
        function_tools=[tool],
        output_tools=[tool],
    )

    # Provide model_settings to cover temperature/top_p branches
    model_settings = {"temperature": 0.3, "top_p": 0.9}

    async with model.request_stream(
        messages=[],
        model_settings=model_settings,
        model_request_parameters=params,
        run_context={"ctx": "x"},
    ) as stream:
        # Validate LocalStreamedResponse construction
        assert stream._runner is runner
        assert stream._gen_kwargs["temperature"] == 0.3
        assert stream._gen_kwargs["top_p"] == 0.9
        assert "SYS" in stream._gen_kwargs["instructions"]
        assert "TOOL PROMPT" in stream._gen_kwargs["instructions"]
        assert len(stream._available_tools) == 2
