"""
Tests for TransformersRunner.generate_stream() method.
"""

import asyncio

import pytest
from unittest.mock import MagicMock, patch

import torch

from practorflow.llm.transformers_runner import TransformersRunner
from practorflow.llm.tools.base import ToolResult
from tests.practorflow.common.fixtures import (
    mock_knowledge_store,
    sample_tool_definitions,
)
from tests.practorflow.llm.common_runner import (
    create_mock_model_handle,
    create_transformers_generate_output,
)


class TestTransformersRunnerGenerateStream:
    """Tests for TransformersRunner.generate_stream()"""

    @pytest.fixture
    def runner(self):
        """Create a TransformersRunner instance for testing."""
        handle = create_mock_model_handle(backend="transformers")
        handle.model.generate.return_value = create_transformers_generate_output(
            input_length=5, output_length=15
        )
        return TransformersRunner(handle)

    @pytest.fixture
    def mock_streamer_chunks(self):
        """Create mock streamer that yields text chunks."""
        return ["Hello", " ", "world", "!"]

    @pytest.mark.asyncio
    async def test_generate_stream_yields_chunks(self, runner, mock_streamer_chunks):
        """generate_stream() yields text chunks from streaming response."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(mock_streamer_chunks))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Say hello"):
                chunks.append(chunk)
            
            text_chunks = [c for c in chunks if c.text and not c.finished]
            assert len(text_chunks) == 4
            assert text_chunks[0].text == "Hello"
            assert text_chunks[1].text == " "
            assert text_chunks[2].text == "world"
            assert text_chunks[3].text == "!"

    @pytest.mark.asyncio
    async def test_generate_stream_final_chunk_has_finished_true(self, runner):
        """generate_stream() final chunk has finished=True."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Response"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello"):
                chunks.append(chunk)
            
            final_chunk = chunks[-1]
            assert final_chunk.finished is True

    @pytest.mark.asyncio
    async def test_generate_stream_final_chunk_has_finish_reason(self, runner):
        """generate_stream() final chunk has finish_reason."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Response"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello"):
                chunks.append(chunk)
            
            final_chunk = chunks[-1]
            assert final_chunk.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_stream_final_chunk_has_latency(self, runner):
        """generate_stream() final chunk has latency_seconds."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Response"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello"):
                chunks.append(chunk)
            
            final_chunk = chunks[-1]
            assert final_chunk.latency_seconds is not None
            assert final_chunk.latency_seconds >= 0

    @pytest.mark.asyncio
    async def test_generate_stream_final_chunk_has_usage(self, runner):
        """generate_stream() final chunk has usage statistics."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Response"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello"):
                chunks.append(chunk)
            
            final_chunk = chunks[-1]
            assert final_chunk.usage is not None
            assert "prompt_tokens" in final_chunk.usage
            assert "completion_tokens" in final_chunk.usage
            assert "total_tokens" in final_chunk.usage

    @pytest.mark.asyncio
    async def test_generate_stream_with_context(self, runner):
        """generate_stream() includes context in final chunk when pending."""
        runner._pending_context = "Document context for the query."
        
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Answer"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="What does the doc say?"):
                chunks.append(chunk)
            
            final_chunk = chunks[-1]
            assert final_chunk.context_used == "Document context for the query."
            assert runner._pending_context is None

    @pytest.mark.asyncio
    async def test_generate_stream_clears_pending_context(self, runner):
        """generate_stream() clears pending context after use."""
        runner._pending_context = "Some context"
        
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Response"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
            
            assert runner._pending_context is None

    @pytest.mark.asyncio
    async def test_generate_stream_with_context_metadata(self, runner):
        """generate_stream() includes search_metadata in final chunk when available."""
        runner._pending_context = "Document content."
        mock_result = ToolResult(
            success=True,
            data="search results",
            metadata={"source": "doc1.txt", "score": 0.95},
        )
        runner.tool_registry._last_result = mock_result
        
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Answer"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
            
            final_chunk = chunks[-1]
            assert final_chunk.search_metadata is not None
            assert final_chunk.search_metadata["source"] == "doc1.txt"
            assert final_chunk.search_metadata["score"] == 0.95

    @pytest.mark.asyncio
    async def test_generate_stream_with_prompt(self, runner):
        """generate_stream() works with prompt parameter."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["OK"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Test prompt"):
                chunks.append(chunk)
            
            runner.tokenizer.apply_chat_template.assert_called_once()
            call_args = runner.tokenizer.apply_chat_template.call_args[0][0]
            user_messages = [msg for msg in call_args if msg["role"] == "user"]
            assert any("Test prompt" in msg["content"] for msg in user_messages)

    @pytest.mark.asyncio
    async def test_generate_stream_with_messages(self, runner):
        """generate_stream() works with messages parameter."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]
        
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["I'm good"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(messages=messages):
                chunks.append(chunk)
            
            runner.tokenizer.apply_chat_template.assert_called_once()
            call_args = runner.tokenizer.apply_chat_template.call_args[0][0]
            assert len(call_args) == 4

    @pytest.mark.asyncio
    async def test_generate_stream_with_instructions(self, runner):
        """generate_stream() includes instructions in system message."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Response"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(
                prompt="Hello",
                instructions="Be concise and helpful.",
            ):
                chunks.append(chunk)
            
            call_args = runner.tokenizer.apply_chat_template.call_args[0][0]
            system_messages = [msg for msg in call_args if msg["role"] == "system"]
            assert len(system_messages) > 0
            assert "concise and helpful" in system_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_generate_stream_uses_temperature(self, runner):
        """generate_stream() passes temperature to generation kwargs."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["OK"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello", temperature=0.5):
                chunks.append(chunk)
            
            call_kwargs = runner.model.generate.call_args[1]
            assert call_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_stream_uses_top_p(self, runner):
        """generate_stream() passes top_p to generation kwargs."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["OK"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello", top_p=0.8):
                chunks.append(chunk)
            
            call_kwargs = runner.model.generate.call_args[1]
            assert call_kwargs["top_p"] == 0.8

    @pytest.mark.asyncio
    async def test_generate_stream_non_finished_chunks_have_finished_false(self, runner):
        """generate_stream() intermediate chunks have finished=False."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["One", "Two", "Three"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Count"):
                chunks.append(chunk)
            
            non_final_chunks = [c for c in chunks if not c.finished]
            for chunk in non_final_chunks:
                assert chunk.finished is False

    @pytest.mark.asyncio
    async def test_generate_stream_with_tools_logs_warning(self, runner, sample_tool_definitions):
        """generate_stream() logs warning when tools provided (not implemented)."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Response"]))
            mock_streamer_class.return_value = mock_streamer
            
            with patch("practorflow.llm.transformers_runner.logger") as mock_logger:
                chunks = []
                async for chunk in runner.generate_stream(prompt="Hello", tools=sample_tool_definitions):
                    chunks.append(chunk)
                
                mock_logger.warning.assert_called()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "tool calling not implemented" in warning_msg.lower()

    @pytest.mark.asyncio
    async def test_generate_stream_creates_streamer_with_tokenizer(self, runner):
        """generate_stream() creates TextIteratorStreamer with tokenizer."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Response"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello"):
                chunks.append(chunk)
            
            mock_streamer_class.assert_called_once()
            call_args = mock_streamer_class.call_args
            assert call_args[0][0] is runner.tokenizer

    @pytest.mark.asyncio
    async def test_generate_stream_creates_streamer_with_skip_prompt(self, runner):
        """generate_stream() creates TextIteratorStreamer with skip_prompt=True."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Response"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello"):
                chunks.append(chunk)
            
            call_kwargs = mock_streamer_class.call_args[1]
            assert call_kwargs["skip_prompt"] is True

    @pytest.mark.asyncio
    async def test_generate_stream_creates_streamer_with_skip_special_tokens(self, runner):
        """generate_stream() creates TextIteratorStreamer with skip_special_tokens=True."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Response"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello"):
                chunks.append(chunk)
            
            call_kwargs = mock_streamer_class.call_args[1]
            assert call_kwargs["skip_special_tokens"] is True

    @pytest.mark.asyncio
    async def test_generate_stream_passes_streamer_to_generate(self, runner):
        """generate_stream() passes streamer to model.generate()."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Response"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello"):
                chunks.append(chunk)
            
            call_kwargs = runner.model.generate.call_args[1]
            assert call_kwargs["streamer"] is mock_streamer

    @pytest.mark.asyncio
    async def test_generate_stream_handles_empty_text_chunks(self, runner):
        """generate_stream() handles empty text chunks gracefully."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Hello", "", "World"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Test"):
                chunks.append(chunk)
            
            assert any(c.text == "Hello" for c in chunks)
            assert any(c.text == "World" for c in chunks)

    @pytest.mark.asyncio
    async def test_generate_stream_error_sets_finish_reason(self, runner):
        """generate_stream() sets error finish_reason when generation fails."""
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter([]))
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.side_effect = RuntimeError("Generation failed")
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello"):
                chunks.append(chunk)
            
            final_chunk = chunks[-1]
            assert final_chunk.finished is True
            assert "error" in final_chunk.finish_reason

    @pytest.mark.asyncio
    async def test_generate_stream_uses_default_temperature(self, runner):
        """generate_stream() uses config temperature when not specified."""
        runner.config.temperature = 0.8
        
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["OK"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello"):
                chunks.append(chunk)
            
            call_kwargs = runner.model.generate.call_args[1]
            assert call_kwargs["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_generate_stream_uses_default_top_p(self, runner):
        """generate_stream() uses config top_p when not specified."""
        runner.config.top_p = 0.95
        
        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["OK"]))
            mock_streamer_class.return_value = mock_streamer
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Hello"):
                chunks.append(chunk)
            
            call_kwargs = runner.model.generate.call_args[1]
            assert call_kwargs["top_p"] == 0.95