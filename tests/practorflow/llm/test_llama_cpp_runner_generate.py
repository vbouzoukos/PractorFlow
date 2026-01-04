"""
Tests for LlamaCppRunner.generate() method.
"""

import pytest
from unittest.mock import MagicMock, patch

from practorflow.llm.llama_cpp_runner import LlamaCppRunner
from tests.practorflow.common.fixtures import (
    mock_knowledge_store,
    sample_chat_messages,
    sample_tool_definitions,
)
from tests.practorflow.llm.common_runner import (
    create_mock_model_handle,
    create_llama_completion_response,
    create_tool_call,
)


class TestLlamaCppRunnerGenerate:
    """Tests for LlamaCppRunner.generate()"""

    @pytest.fixture
    def runner(self):
        """Create a LlamaCppRunner instance for testing."""
        handle = create_mock_model_handle(backend="llama_cpp")
        handle.model.create_chat_completion.return_value = create_llama_completion_response(
            content="Test response"
        )
        return LlamaCppRunner(handle)

    @pytest.mark.asyncio
    async def test_generate_with_prompt(self, runner):
        """generate() works with a simple prompt."""
        result = await runner.generate(prompt="Hello, world!")
        
        assert "reply" in result
        assert "latency_seconds" in result
        runner.model.create_chat_completion.assert_called_once()
        
        call_kwargs = runner.model.create_chat_completion.call_args[1]
        messages = call_kwargs["messages"]
        assert any(msg["role"] == "user" and "Hello, world!" in msg["content"] for msg in messages)

    @pytest.mark.asyncio
    async def test_generate_with_messages(self, runner, sample_chat_messages):
        """generate() works with a list of messages."""
        result = await runner.generate(messages=sample_chat_messages)
        
        assert "reply" in result
        runner.model.create_chat_completion.assert_called_once()
        
        call_kwargs = runner.model.create_chat_completion.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == len(sample_chat_messages)

    @pytest.mark.asyncio
    async def test_generate_with_instructions(self, runner):
        """generate() includes system instructions in messages."""
        result = await runner.generate(
            prompt="What is Python?",
            instructions="You are a programming expert.",
        )
        
        call_kwargs = runner.model.create_chat_completion.call_args[1]
        messages = call_kwargs["messages"]
        
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        assert len(system_messages) > 0
        assert "programming expert" in system_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_generate_with_context(self, runner):
        """generate() includes pending context in messages."""
        runner._pending_context = "Relevant document content here."
        
        result = await runner.generate(prompt="What does the document say?")
        
        call_kwargs = runner.model.create_chat_completion.call_args[1]
        messages = call_kwargs["messages"]
        
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        assert any("Relevant document content here" in msg["content"] for msg in system_messages)
        assert "context_used" in result
        assert runner._pending_context is None

    @pytest.mark.asyncio
    async def test_generate_with_tools(self, runner, sample_tool_definitions):
        """generate() passes tools to create_chat_completion."""
        result = await runner.generate(
            prompt="What is the weather?",
            tools=sample_tool_definitions,
        )
        
        call_kwargs = runner.model.create_chat_completion.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_generate_extracts_tool_calls(self, runner):
        """generate() extracts tool calls from response."""
        tool_calls = [
            create_tool_call("get_weather", {"location": "San Francisco, CA"}),
        ]
        runner.model.create_chat_completion.return_value = create_llama_completion_response(
            content="",
            tool_calls=tool_calls,
        )
        
        result = await runner.generate(prompt="What is the weather in SF?")
        
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool_name"] == "get_weather"
        assert result["tool_calls"][0]["args"]["location"] == "San Francisco, CA"

    @pytest.mark.asyncio
    async def test_generate_returns_response_structure(self, runner):
        """generate() returns expected response structure."""
        result = await runner.generate(prompt="Hello")
        
        assert "reply" in result
        assert "latency_seconds" in result
        assert isinstance(result["latency_seconds"], float)
        assert result["latency_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_generate_with_both_messages_and_prompt_raises(self, runner, sample_chat_messages):
        """generate() raises ValueError when both messages and prompt provided."""
        with pytest.raises(ValueError, match="Cannot provide both messages and prompt"):
            await runner.generate(
                messages=sample_chat_messages,
                prompt="Hello",
            )

    @pytest.mark.asyncio
    async def test_generate_with_neither_messages_nor_prompt_raises(self, runner):
        """generate() raises ValueError when neither messages nor prompt provided."""
        with pytest.raises(ValueError, match="Must provide either messages or prompt"):
            await runner.generate()

    @pytest.mark.asyncio
    async def test_generate_uses_config_defaults_for_temperature_and_top_p(self, runner):
        """generate() uses config defaults when temperature and top_p not provided."""
        await runner.generate(prompt="Hello")
        
        call_kwargs = runner.model.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == runner.config.temperature
        assert call_kwargs["top_p"] == runner.config.top_p

    @pytest.mark.asyncio
    async def test_generate_uses_provided_temperature_and_top_p(self, runner):
        """generate() uses provided temperature and top_p values."""
        await runner.generate(prompt="Hello", temperature=0.5, top_p=0.8)
        
        call_kwargs = runner.model.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.8

    @pytest.mark.asyncio
    async def test_generate_includes_max_tokens(self, runner):
        """generate() includes max_tokens from config."""
        await runner.generate(prompt="Hello")
        
        call_kwargs = runner.model.create_chat_completion.call_args[1]
        assert call_kwargs["max_tokens"] == runner.max_new_tokens

    @pytest.mark.asyncio
    async def test_generate_includes_stop_tokens_when_configured(self, runner):
        """generate() includes stop tokens when configured."""
        runner.config.stop_tokens = ["</s>", "<|end|>"]
        
        await runner.generate(prompt="Hello")
        
        call_kwargs = runner.model.create_chat_completion.call_args[1]
        assert call_kwargs["stop"] == ["</s>", "<|end|>"]

    @pytest.mark.asyncio
    async def test_generate_without_stop_tokens(self, runner):
        """generate() does not include stop key when stop_tokens is None."""
        runner.config.stop_tokens = None
        
        await runner.generate(prompt="Hello")
        
        call_kwargs = runner.model.create_chat_completion.call_args[1]
        assert "stop" not in call_kwargs

    @pytest.mark.asyncio
    async def test_generate_with_context_and_search_metadata(self, runner):
        """generate() includes search_metadata in response when available."""
        from practorflow.llm.tools.base import ToolResult

        runner._pending_context = "Relevant document content."
        mock_result = ToolResult(
            success=True,
            data="search results",
            metadata={"filename": "report.pdf", "relevance": 0.92},
        )
        runner.tool_registry._last_result = mock_result

        result = await runner.generate(prompt="What does the report say?")

        assert "context_used" in result
        assert result["context_used"] == "Relevant document content."
        assert "search_metadata" in result
        assert result["search_metadata"]["filename"] == "report.pdf"
        assert result["search_metadata"]["relevance"] == 0.92

    @pytest.mark.asyncio
    async def test_generate_with_context_but_no_metadata(self, runner):
        """generate() handles context without metadata gracefully."""
        from practorflow.llm.tools.base import ToolResult

        runner._pending_context = "Document content."
        mock_result = ToolResult(
            success=True,
            data="search results",
            metadata=None,
        )
        runner.tool_registry._last_result = mock_result

        result = await runner.generate(prompt="Question?")

        assert "context_used" in result
        assert result["context_used"] == "Document content."
        assert "search_metadata" not in result

    @pytest.mark.asyncio
    async def test_generate_with_context_clears_tool_registry(self, runner):
        """generate() clears last_result from tool registry after using context."""
        from practorflow.llm.tools.base import ToolResult

        runner._pending_context = "Context."
        mock_result = ToolResult(success=True, data="data", metadata={"key": "value"})
        runner.tool_registry._last_result = mock_result

        await runner.generate(prompt="Test")

        assert runner.tool_registry._last_result is None