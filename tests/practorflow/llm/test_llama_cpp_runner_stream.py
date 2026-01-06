"""
Tests for LlamaCppRunner.generate_stream() method.
"""

import pytest
from unittest.mock import MagicMock, patch

from practorflow.llm.llama_cpp_runner import LlamaCppRunner
from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.llm.common_runner import (
    create_mock_model_handle,
    create_llama_stream_chunks,
    create_tool_call,
    create_incremental_tool_call_stream_chunks,
)


class TestLlamaCppRunnerGenerateStream:
    """Tests for LlamaCppRunner.generate_stream()"""

    @pytest.fixture
    def runner(self):
        """Create a LlamaCppRunner instance for testing."""
        handle = create_mock_model_handle(backend="llama_cpp")
        return LlamaCppRunner(handle)

    @pytest.mark.asyncio
    async def test_generate_stream_yields_chunks(self, runner):
        """generate_stream() yields text chunks from streaming response."""
        stream_chunks = create_llama_stream_chunks(["Hello", " ", "world", "!"])
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Say hello"):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if c.text]
        # 4 content chunks + 1 part
        assert len(text_chunks) == 4
        assert text_chunks[0].text == "Hello"
        assert text_chunks[1].text == " "
        assert text_chunks[2].text == "world"
        assert text_chunks[3].text == "!"

    @pytest.mark.asyncio
    async def test_generate_stream_final_chunk_has_metadata(self, runner):
        """generate_stream() final chunk has finished=True and metadata."""
        stream_chunks = create_llama_stream_chunks(["Response"], finish_reason="stop")
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Hello"):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.finished is True
        assert final_chunk.finish_reason == "stop"
        assert final_chunk.latency_seconds is not None
        assert final_chunk.latency_seconds >= 0

    @pytest.mark.asyncio
    async def test_generate_stream_with_context(self, runner):
        """generate_stream() includes context in final chunk when pending."""
        runner._pending_context = "Document context for the query."
        stream_chunks = create_llama_stream_chunks(["Answer"])
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="What does the doc say?"):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.context_used == "Document context for the query."
        assert runner._pending_context is None

    @pytest.mark.asyncio
    async def test_generate_stream_with_tool_calls(self, runner):
        """generate_stream() accumulates and includes tool calls in final chunk."""
        tool_calls = [
            create_tool_call("get_weather", {"location": "NYC"}),
        ]
        stream_chunks = create_llama_stream_chunks(
            [""],
            finish_reason="tool_calls",
            tool_calls=tool_calls,
        )
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="What is the weather?"):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.finished is True
        assert final_chunk.tool_calls is not None
        assert len(final_chunk.tool_calls) == 1
        assert final_chunk.tool_calls[0]["tool_name"] == "get_weather"
        assert final_chunk.tool_calls[0]["args"]["location"] == "NYC"

    @pytest.mark.asyncio
    async def test_generate_stream_handles_error(self, runner):
        """generate_stream() yields error chunk when exception occurs."""
        def raise_error(*args, **kwargs):
            raise RuntimeError("Model inference failed")

        runner.model.create_chat_completion.side_effect = raise_error

        chunks = []
        async for chunk in runner.generate_stream(prompt="Hello"):
            chunks.append(chunk)

        assert len(chunks) == 1
        final_chunk = chunks[0]
        assert final_chunk.finished is True
        assert "error" in final_chunk.finish_reason
        assert "Model inference failed" in final_chunk.finish_reason

    @pytest.mark.asyncio
    async def test_generate_stream_with_prompt(self, runner):
        """generate_stream() works with prompt parameter."""
        stream_chunks = create_llama_stream_chunks(["OK"])
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Test prompt"):
            chunks.append(chunk)

        runner.model.create_chat_completion.assert_called_once()
        call_kwargs = runner.model.create_chat_completion.call_args[1]
        assert call_kwargs["stream"] is True
        messages = call_kwargs["messages"]
        assert any("Test prompt" in msg["content"] for msg in messages)

    @pytest.mark.asyncio
    async def test_generate_stream_with_messages(self, runner):
        """generate_stream() works with messages parameter."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]
        stream_chunks = create_llama_stream_chunks(["I'm good"])
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(messages=messages):
            chunks.append(chunk)

        call_kwargs = runner.model.create_chat_completion.call_args[1]
        assert len(call_kwargs["messages"]) == 3

    @pytest.mark.asyncio
    async def test_generate_stream_with_instructions(self, runner):
        """generate_stream() includes instructions in system message."""
        stream_chunks = create_llama_stream_chunks(["Response"])
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(
            prompt="Hello",
            instructions="Be concise and helpful.",
        ):
            chunks.append(chunk)

        call_kwargs = runner.model.create_chat_completion.call_args[1]
        messages = call_kwargs["messages"]
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) > 0
        assert "concise and helpful" in system_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_generate_stream_uses_temperature_and_top_p(self, runner):
        """generate_stream() passes temperature and top_p to model."""
        stream_chunks = create_llama_stream_chunks(["OK"])
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(
            prompt="Hello",
            temperature=0.5,
            top_p=0.8,
        ):
            chunks.append(chunk)

        call_kwargs = runner.model.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.8

    @pytest.mark.asyncio
    async def test_generate_stream_non_finished_chunks_have_finished_false(self, runner):
        """generate_stream() intermediate chunks have finished=False."""
        stream_chunks = create_llama_stream_chunks(["One", "Two", "Three"])
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Count"):
            chunks.append(chunk)

        non_final_chunks = chunks[:-1]
        for chunk in non_final_chunks:
            assert chunk.finished is False

    @pytest.mark.asyncio
    async def test_generate_stream_with_context_metadata(self, runner):
        """generate_stream() includes search_metadata in final chunk when available."""
        from practorflow.llm.tools.base import ToolResult

        runner._pending_context = "Document content."
        mock_result = ToolResult(
            success=True,
            data="search results",
            metadata={"source": "doc1.txt", "score": 0.95},
        )
        runner.tool_registry._last_result = mock_result

        stream_chunks = create_llama_stream_chunks(["Answer"])
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Question"):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.context_used == "Document content."
        assert final_chunk.search_metadata is not None
        assert final_chunk.search_metadata["source"] == "doc1.txt"
        assert final_chunk.search_metadata["score"] == 0.95

    @pytest.mark.asyncio
    async def test_generate_stream_with_stop_tokens(self, runner):
        """generate_stream() passes stop_tokens to model when configured."""
        runner.config.stop_tokens = ["</s>", "<|end|>"]
        stream_chunks = create_llama_stream_chunks(["Response"])
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Hello"):
            chunks.append(chunk)

        call_kwargs = runner.model.create_chat_completion.call_args[1]
        assert call_kwargs["stop"] == ["</s>", "<|end|>"]

    @pytest.mark.asyncio
    async def test_generate_stream_tool_calls_incremental_accumulation(self, runner):
        """generate_stream() accumulates tool calls from incremental chunks."""
        chunks_data = create_incremental_tool_call_stream_chunks(
            tool_name="get_weather",
            arguments={"location": "NYC"},
            call_id="call_123",
            include_id_in_first_chunk=True,
        )
        runner.model.create_chat_completion.return_value = iter(chunks_data)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Weather?"):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.finished is True
        assert final_chunk.tool_calls is not None
        assert len(final_chunk.tool_calls) == 1
        assert final_chunk.tool_calls[0]["tool_name"] == "get_weather"
        assert final_chunk.tool_calls[0]["args"]["location"] == "NYC"
        assert final_chunk.tool_calls[0]["tool_call_id"] == "call_123"

    @pytest.mark.asyncio
    async def test_generate_stream_tool_calls_without_id_in_chunk(self, runner):
        """generate_stream() handles tool call chunks without id field."""
        chunks_data = create_incremental_tool_call_stream_chunks(
            tool_name="test_func",
            arguments={"param": "value"},
            call_id="ignored",
            include_id_in_first_chunk=False,
        )
        runner.model.create_chat_completion.return_value = iter(chunks_data)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Test"):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.tool_calls is not None
        assert len(final_chunk.tool_calls) == 1
        assert final_chunk.tool_calls[0]["tool_name"] == "test_func"
        assert "tool_call_id" in final_chunk.tool_calls[0]

    @pytest.mark.asyncio
    async def test_generate_stream_tool_calls_with_invalid_json_args(self, runner):
        """generate_stream() handles invalid JSON in tool call arguments."""
        incremental_chunks = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_bad",
                                    "function": {"name": "bad_func", "arguments": "not valid json"},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        runner.model.create_chat_completion.return_value = iter(incremental_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Test"):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.tool_calls is not None
        assert len(final_chunk.tool_calls) == 1
        assert final_chunk.tool_calls[0]["tool_name"] == "bad_func"
        assert final_chunk.tool_calls[0]["args"] == {}

    @pytest.mark.asyncio
    async def test_generate_stream_tool_calls_empty_name_skipped(self, runner):
        """generate_stream() skips tool calls with empty name."""
        incremental_chunks = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_empty",
                                    "function": {"name": "", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        runner.model.create_chat_completion.return_value = iter(incremental_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Test"):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.tool_calls is None or len(final_chunk.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_generate_stream_tool_calls_empty_args_string(self, runner):
        """generate_stream() handles empty arguments string."""
        incremental_chunks = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_empty_args",
                                    "function": {"name": "no_args_func", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        runner.model.create_chat_completion.return_value = iter(incremental_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Test"):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.tool_calls is not None
        assert len(final_chunk.tool_calls) == 1
        assert final_chunk.tool_calls[0]["tool_name"] == "no_args_func"
        assert final_chunk.tool_calls[0]["args"] == {}


class TestLlamaCppRunnerStreamToolCallProcessing:
    """Direct tests for stream tool call processing to ensure coverage."""

    @pytest.fixture
    def runner(self):
        """Create a LlamaCppRunner instance for testing."""
        handle = create_mock_model_handle(backend="llama_cpp")
        return LlamaCppRunner(handle)

    @pytest.fixture
    def sample_tools(self):
        """Sample tools to pass to generate_stream."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_stream_tool_call_with_id_field(self, runner, sample_tools):
        """Covers line 329: if 'id' in tc - when id IS present."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_with_id_123",
                                    "function": {"name": "my_tool", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            },
        ]
        runner.model.create_chat_completion.return_value = iter(chunks)

        result_chunks = []
        async for chunk in runner.generate_stream(prompt="test", tools=sample_tools):
            result_chunks.append(chunk)

        final = result_chunks[-1]
        assert final.tool_calls is not None
        assert final.tool_calls[0]["tool_call_id"] == "call_with_id_123"

    @pytest.mark.asyncio
    async def test_stream_tool_call_processes_accumulated_calls(self, runner, sample_tools):
        """Covers lines 350-353: processing accumulated_tool_calls loop."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "func_a", "arguments": '{"x": 1}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "function": {"name": "func_b", "arguments": '{"y": 2}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            },
        ]
        runner.model.create_chat_completion.return_value = iter(chunks)

        result_chunks = []
        async for chunk in runner.generate_stream(prompt="test", tools=sample_tools):
            result_chunks.append(chunk)

        final = result_chunks[-1]
        assert final.tool_calls is not None
        assert len(final.tool_calls) == 2
        assert final.tool_calls[0]["tool_name"] == "func_a"
        assert final.tool_calls[0]["args"] == {"x": 1}
        assert final.tool_calls[1]["tool_name"] == "func_b"
        assert final.tool_calls[1]["args"] == {"y": 2}

    @pytest.mark.asyncio
    async def test_stream_tool_call_appends_when_name_present(self, runner, sample_tools):
        """Covers line 361: if tool_name - appending when name is truthy."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_valid",
                                    "function": {"name": "valid_tool", "arguments": '{"param": "value"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            },
        ]
        runner.model.create_chat_completion.return_value = iter(chunks)

        result_chunks = []
        async for chunk in runner.generate_stream(prompt="test", tools=sample_tools):
            result_chunks.append(chunk)

        final = result_chunks[-1]
        assert final.tool_calls is not None
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0]["tool_name"] == "valid_tool"
        assert final.tool_calls[0]["args"]["param"] == "value"
        assert final.tool_calls[0]["tool_call_id"] == "call_valid"

    @pytest.mark.asyncio
    async def test_stream_tool_call_skipped_when_name_empty(self, runner, sample_tools):
        """Covers line 361 else branch: tool_name empty, not appended."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_no_name",
                                    "function": {"name": "", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            },
        ]
        runner.model.create_chat_completion.return_value = iter(chunks)

        result_chunks = []
        async for chunk in runner.generate_stream(prompt="test", tools=sample_tools):
            result_chunks.append(chunk)

        final = result_chunks[-1]
        assert final.tool_calls is None or len(final.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_stream_tool_call_json_decode_error(self, runner, sample_tools):
        """Covers lines 350-353: JSONDecodeError branch in processing."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_bad_json",
                                    "function": {"name": "tool_x", "arguments": "invalid json{"},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            },
        ]
        runner.model.create_chat_completion.return_value = iter(chunks)

        result_chunks = []
        async for chunk in runner.generate_stream(prompt="test", tools=sample_tools):
            result_chunks.append(chunk)

        final = result_chunks[-1]
        assert final.tool_calls is not None
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0]["tool_name"] == "tool_x"
        assert final.tool_calls[0]["args"] == {}

    @pytest.mark.asyncio
    async def test_stream_skips_empty_choices(self, runner, sample_tools):
        """Covers line 361: if not choices: continue."""
        chunks = [
            {
                "choices": [],
            },
            {
                "choices": [
                    {
                        "delta": {"content": "Hello"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
        ]
        runner.model.create_chat_completion.return_value = iter(chunks)

        result_chunks = []
        async for chunk in runner.generate_stream(prompt="test", tools=sample_tools):
            result_chunks.append(chunk)

        text_chunks = [c for c in result_chunks if c.text]
        assert len(text_chunks) == 1
        assert text_chunks[0].text == "Hello"