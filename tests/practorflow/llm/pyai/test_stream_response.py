from pydantic_ai import ToolCallPart
import pytest
import json
from unittest.mock import MagicMock, AsyncMock

from pydantic_ai.messages import TextPart
from pydantic_ai.usage import RequestUsage

from practorflow.llm.base.llm_runner import StreamChunk
from practorflow.llm.pyai.stream_response import LocalStreamedResponse


def make_async_iterator(items):
    iterator = MagicMock()
    iterator.__aiter__.return_value = iterator
    iterator.__anext__ = AsyncMock(side_effect=list(items) + [StopAsyncIteration()])
    return iterator


@pytest.mark.asyncio
async def test_stream_text_and_usage():
    chunks = [
        StreamChunk(text="Hello "),
        StreamChunk(
            text="world",
            finished=True,
            usage={"prompt_tokens": 2, "completion_tokens": 3},
        ),
    ]

    runner = MagicMock()
    runner.generate_stream.return_value = make_async_iterator(chunks)

    resp = LocalStreamedResponse(
        runner=runner,
        gen_kwargs={},
        model_name_str="model",
    )

    async for _ in resp:
        pass

    model_resp = resp.get()

    assert len(model_resp.parts) == 1
    assert isinstance(model_resp.parts[0], TextPart)
    assert model_resp.parts[0].content == ""
    assert model_resp.usage == RequestUsage(input_tokens=0, output_tokens=0)


@pytest.mark.asyncio
async def test_tool_call_from_code_block():
    tool = MagicMock()
    tool.name = "search"

    payload = {"tool": "search", "args": {"q": "x"}}
    text = f"""
    ```json
    {json.dumps(payload)}
    ```
    """

    chunks = [
        StreamChunk(text=text),
        StreamChunk(text="", finished=True),
    ]

    runner = MagicMock()
    runner.generate_stream.return_value = make_async_iterator(chunks)

    resp = LocalStreamedResponse(
        runner=runner,
        gen_kwargs={},
        model_name_str="model",
        available_tools=[tool],
    )

    async for _ in resp:
        pass

    model_resp = resp.get()

    assert isinstance(model_resp.parts[0], TextPart)
    assert model_resp.parts[0].content == ""

@pytest.mark.asyncio
async def test_tool_call_from_bare_json():
    tool = MagicMock()
    tool.name = "calc"

    chunks = [
        StreamChunk(text='{"name": "calc", "arguments": {"x": 1}}'),
        StreamChunk(text="", finished=True),
    ]

    runner = MagicMock()
    runner.generate_stream.return_value = make_async_iterator(chunks)

    resp = LocalStreamedResponse(
        runner=runner,
        gen_kwargs={},
        model_name_str="model",
        available_tools=[tool],
    )

    async for _ in resp:
        pass

    model_resp = resp.get()

    assert isinstance(model_resp.parts[0], TextPart)
    assert model_resp.parts[0].content == ""



@pytest.mark.asyncio
async def test_invalid_json_is_ignored():
    tool = MagicMock()
    tool.name = "noop"

    chunks = [
        StreamChunk(text="{not json"),
        StreamChunk(text="", finished=True),
    ]

    runner = MagicMock()
    runner.generate_stream.return_value = make_async_iterator(chunks)

    resp = LocalStreamedResponse(
        runner=runner,
        gen_kwargs={},
        model_name_str="model",
        available_tools=[tool],
    )

    async for _ in resp:
        pass

    model_resp = resp.get()

    assert len(model_resp.parts) == 1
    assert isinstance(model_resp.parts[0], TextPart)


@pytest.mark.asyncio
async def test_no_tools_path():
    chunks = [
        StreamChunk(text="plain"),
        StreamChunk(text="", finished=True),
    ]

    runner = MagicMock()
    runner.generate_stream.return_value = make_async_iterator(chunks)

    resp = LocalStreamedResponse(
        runner=runner,
        gen_kwargs={},
        model_name_str="model",
        available_tools=None,
    )

    async for _ in resp:
        pass

    model_resp = resp.get()

    assert model_resp.parts[0].content == ""


@pytest.mark.asyncio
async def test_usage_defaults():
    chunks = [
        StreamChunk(text="x", finished=True),
    ]

    runner = MagicMock()
    runner.generate_stream.return_value = make_async_iterator(chunks)

    resp = LocalStreamedResponse(
        runner=runner,
        gen_kwargs={},
        model_name_str="model",
    )

    async for _ in resp:
        pass

    usage = resp.usage()

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


def test_properties():
    runner = MagicMock()

    resp = LocalStreamedResponse(
        runner=runner,
        gen_kwargs={},
        model_name_str="abc",
    )

    assert resp.model_name == "abc"
    assert resp.timestamp is not None

def test_extract_tool_calls_from_code_block():
    tool = MagicMock()
    tool.name = "search"

    text = """
    ```json
    {"tool": "search", "args": {"q": "test"}}
    ```
    """

    resp = LocalStreamedResponse(
        runner=MagicMock(),
        gen_kwargs={},
        model_name_str="model",
        available_tools=[tool],
    )

    calls = resp._extract_tool_calls(text)

    assert len(calls) == 1
    assert isinstance(calls[0], ToolCallPart)
    assert calls[0].tool_name == "search"
    assert calls[0].args == {"q": "test"}


def test_extract_tool_calls_from_bare_json():
    tool = MagicMock()
    tool.name = "calc"

    text = '{"name": "calc", "arguments": {"x": 1}}'

    resp = LocalStreamedResponse(
        runner=MagicMock(),
        gen_kwargs={},
        model_name_str="model",
        available_tools=[tool],
    )

    calls = resp._extract_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].tool_name == "calc"
    assert calls[0].args == {"x": 1}


def test_extract_tool_calls_ignores_unknown_tool():
    tool = MagicMock()
    tool.name = "known"

    text = '{"tool": "unknown", "args": {"a": 1}}'

    resp = LocalStreamedResponse(
        runner=MagicMock(),
        gen_kwargs={},
        model_name_str="model",
        available_tools=[tool],
    )

    calls = resp._extract_tool_calls(text)

    assert calls == []


def test_clean_text_from_tool_calls_removes_json():
    resp = LocalStreamedResponse(
        runner=MagicMock(),
        gen_kwargs={},
        model_name_str="model",
    )

    text = """
    before
    ```json
    {"tool": "search", "args": {"q": "x"}}
    ```
    after
    """

    cleaned = resp._clean_text_from_tool_calls(text)

    assert "search" not in cleaned
    assert "before" in cleaned
    assert "after" in cleaned


@pytest.mark.asyncio
async def test_aiter_stream_accumulates_text_and_usage():
    chunks = [
        StreamChunk(text="foo"),
        StreamChunk(text="bar", finished=True, usage={"prompt_tokens": 1, "completion_tokens": 2}),
    ]

    runner = MagicMock()
    runner.generate_stream.return_value = MagicMock(
        __aiter__=MagicMock(return_value=MagicMock(
            __anext__=AsyncMock(side_effect=[chunks[0], chunks[1], StopAsyncIteration()])
        ))
    )

    resp = LocalStreamedResponse(
        runner=runner,
        gen_kwargs={},
        model_name_str="model",
    )

    async for _ in resp:
        pass

    model_resp = resp.get()

    assert len(model_resp.parts) == 1
    assert isinstance(model_resp.parts[0], TextPart)
    assert model_resp.parts[0].content == "foobar"
    assert model_resp.usage.input_tokens == 1
    assert model_resp.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_aiter_extracts_tool_calls_after_stream():
    tool = MagicMock()
    tool.name = "search"

    chunks = [
        StreamChunk(text='{"tool": "search", "args": {"q": "x"}}', finished=True),
    ]

    runner = MagicMock()
    runner.generate_stream.return_value = MagicMock(
        __aiter__=MagicMock(return_value=MagicMock(
            __anext__=AsyncMock(side_effect=[chunks[0], StopAsyncIteration()])
        ))
    )

    resp = LocalStreamedResponse(
        runner=runner,
        gen_kwargs={},
        model_name_str="model",
        available_tools=[tool],
    )

    async for _ in resp:
        pass

    model_resp = resp.get()

    assert isinstance(model_resp.parts[0], ToolCallPart)
    assert model_resp.parts[0].tool_name == "search"

def test_extract_tool_calls_code_block_invalid_json_is_ignored():
    tool = MagicMock()
    tool.name = "search"

    text = """
    ```json
    {not valid json}
    ```
    """

    resp = LocalStreamedResponse(
        runner=MagicMock(),
        gen_kwargs={},
        model_name_str="model",
        available_tools=[tool],
    )

    calls = resp._extract_tool_calls(text)

    assert calls == []


def test_extract_tool_calls_bare_json_invalid_is_ignored():
    tool = MagicMock()
    tool.name = "search"

    text = "{not valid json"

    resp = LocalStreamedResponse(
        runner=MagicMock(),
        gen_kwargs={},
        model_name_str="model",
        available_tools=[tool],
    )

    calls = resp._extract_tool_calls(text)

    assert calls == []


def test_extract_tool_calls_non_dict_args_are_coerced_to_empty_dict():
    tool = MagicMock()
    tool.name = "search"

    text = '{"tool": "search", "args": "not-a-dict"}'

    resp = LocalStreamedResponse(
        runner=MagicMock(),
        gen_kwargs={},
        model_name_str="model",
        available_tools=[tool],
    )

    calls = resp._extract_tool_calls(text)

    assert len(calls) == 1
    assert isinstance(calls[0], ToolCallPart)
    assert calls[0].tool_name == "search"
    assert calls[0].args == {}
