"""
Tests for TransformersRunner.generate() method.
"""

import pytest
from unittest.mock import MagicMock, patch

import torch

from practorflow.llm.transformers_runner import TransformersRunner
from practorflow.llm.tools.base import ToolResult
from tests.practorflow.common.fixtures import (
    mock_knowledge_store,
    sample_chat_messages,
    sample_tool_definitions,
)
from tests.practorflow.llm.common_runner import (
    create_mock_model_handle,
    create_transformers_generate_output,
    MockTokenizerOutput,
)


class TestTransformersRunnerGenerate:
    """Tests for TransformersRunner.generate()"""

    @pytest.fixture
    def runner(self):
        """Create a TransformersRunner instance for testing."""
        handle = create_mock_model_handle(backend="transformers")
        handle.model.generate.return_value = create_transformers_generate_output(
            input_length=5, output_length=15
        )
        return TransformersRunner(handle)

    @pytest.mark.asyncio
    async def test_generate_with_prompt(self, runner):
        """generate() works with a simple prompt."""
        result = await runner.generate(prompt="Hello, world!")
        
        assert "reply" in result
        assert "latency_seconds" in result
        assert "usage" in result
        runner.model.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_messages(self, runner, sample_chat_messages):
        """generate() works with a list of messages."""
        result = await runner.generate(messages=sample_chat_messages)
        
        assert "reply" in result
        runner.model.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_instructions(self, runner):
        """generate() includes system instructions in messages."""
        result = await runner.generate(
            prompt="What is Python?",
            instructions="You are a programming expert.",
        )
        
        assert "reply" in result
        runner.tokenizer.apply_chat_template.assert_called_once()
        call_args = runner.tokenizer.apply_chat_template.call_args[0][0]
        system_messages = [msg for msg in call_args if msg["role"] == "system"]
        assert len(system_messages) > 0
        assert "programming expert" in system_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_generate_with_context(self, runner):
        """generate() includes pending context in messages."""
        runner._pending_context = "Relevant document content here."
        
        result = await runner.generate(prompt="What does the document say?")
        
        assert "reply" in result
        assert "context_used" in result
        assert result["context_used"] == "Relevant document content here."
        assert runner._pending_context is None

    @pytest.mark.asyncio
    async def test_generate_clears_pending_context(self, runner):
        """generate() clears pending context after use."""
        runner._pending_context = "Some context"
        
        await runner.generate(prompt="Question")
        
        assert runner._pending_context is None

    @pytest.mark.asyncio
    async def test_generate_with_context_metadata(self, runner):
        """generate() includes search_metadata when available."""
        runner._pending_context = "Document content."
        mock_result = ToolResult(
            success=True,
            data="search results",
            metadata={"source": "doc1.txt", "score": 0.95},
        )
        runner.tool_registry._last_result = mock_result
        
        result = await runner.generate(prompt="Question")
        
        assert "search_metadata" in result
        assert result["search_metadata"]["source"] == "doc1.txt"
        assert result["search_metadata"]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_generate_returns_usage_stats(self, runner):
        """generate() returns token usage statistics."""
        result = await runner.generate(prompt="Hello")
        
        assert "usage" in result
        assert "prompt_tokens" in result["usage"]
        assert "completion_tokens" in result["usage"]
        assert "total_tokens" in result["usage"]

    @pytest.mark.asyncio
    async def test_generate_returns_latency(self, runner):
        """generate() returns latency in seconds."""
        result = await runner.generate(prompt="Hello")
        
        assert "latency_seconds" in result
        assert isinstance(result["latency_seconds"], float)
        assert result["latency_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_generate_uses_default_temperature(self, runner):
        """generate() uses config temperature when not specified."""
        runner.config.temperature = 0.8
        
        await runner.generate(prompt="Hello")
        
        call_kwargs = runner.model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_generate_uses_provided_temperature(self, runner):
        """generate() uses provided temperature over config default."""
        runner.config.temperature = 0.8
        
        await runner.generate(prompt="Hello", temperature=0.5)
        
        call_kwargs = runner.model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_uses_default_top_p(self, runner):
        """generate() uses config top_p when not specified."""
        runner.config.top_p = 0.95
        
        await runner.generate(prompt="Hello")
        
        call_kwargs = runner.model.generate.call_args[1]
        assert call_kwargs["top_p"] == 0.95

    @pytest.mark.asyncio
    async def test_generate_uses_provided_top_p(self, runner):
        """generate() uses provided top_p over config default."""
        runner.config.top_p = 0.95
        
        await runner.generate(prompt="Hello", top_p=0.8)
        
        call_kwargs = runner.model.generate.call_args[1]
        assert call_kwargs["top_p"] == 0.8

    @pytest.mark.asyncio
    async def test_generate_with_zero_temperature_disables_sampling(self, runner):
        """generate() sets do_sample=False when temperature is 0."""
        await runner.generate(prompt="Hello", temperature=0)
        
        call_kwargs = runner.model.generate.call_args[1]
        assert call_kwargs["do_sample"] is False

    @pytest.mark.asyncio
    async def test_generate_with_positive_temperature_enables_sampling(self, runner):
        """generate() sets do_sample=True when temperature > 0."""
        await runner.generate(prompt="Hello", temperature=0.7)
        
        call_kwargs = runner.model.generate.call_args[1]
        assert call_kwargs["do_sample"] is True

    @pytest.mark.asyncio
    async def test_generate_passes_max_new_tokens(self, runner):
        """generate() passes max_new_tokens to model."""
        runner.max_new_tokens = 256
        
        await runner.generate(prompt="Hello")
        
        call_kwargs = runner.model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 256

    @pytest.mark.asyncio
    async def test_generate_passes_pad_token_id(self, runner):
        """generate() passes pad_token_id to model."""
        runner._pad_token_id = 42
        
        await runner.generate(prompt="Hello")
        
        call_kwargs = runner.model.generate.call_args[1]
        assert call_kwargs["pad_token_id"] == 42

    @pytest.mark.asyncio
    async def test_generate_passes_eos_token_id(self, runner):
        """generate() passes eos_token_id to model."""
        runner._eos_token_id = 99
        
        await runner.generate(prompt="Hello")
        
        call_kwargs = runner.model.generate.call_args[1]
        assert call_kwargs["eos_token_id"] == 99

    @pytest.mark.asyncio
    async def test_generate_with_tools_logs_warning(self, runner, sample_tool_definitions):
        """generate() logs warning when tools provided (not implemented)."""
        with patch("practorflow.llm.transformers_runner.logger") as mock_logger:
            await runner.generate(prompt="Hello", tools=sample_tool_definitions)
            
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "tool calling not implemented" in warning_msg.lower()

    @pytest.mark.asyncio
    async def test_generate_uses_apply_chat_template(self, runner):
        """generate() uses tokenizer.apply_chat_template when available."""
        await runner.generate(prompt="Hello")
        
        runner.tokenizer.apply_chat_template.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_uses_fallback_when_no_chat_template(self, runner):
        """generate() uses fallback formatting when apply_chat_template unavailable."""
        del runner.tokenizer.apply_chat_template
        
        with patch.object(runner, "_format_messages_fallback", return_value="formatted") as mock_fallback:
            await runner.generate(prompt="Hello")
            
            mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_decodes_output_tokens(self, runner):
        """generate() decodes generated tokens using tokenizer."""
        runner.tokenizer.decode.return_value = "Decoded response"
        
        result = await runner.generate(prompt="Hello")
        
        runner.tokenizer.decode.assert_called_once()
        assert result["reply"] == "Decoded response"


class TestTransformersRunnerPrepareInputs:
    """Tests for TransformersRunner._prepare_inputs()"""

    @pytest.fixture
    def runner(self):
        """Create a TransformersRunner instance for testing."""
        handle = create_mock_model_handle(backend="transformers")
        return TransformersRunner(handle)

    def test_prepare_inputs_returns_tuple(self, runner):
        """_prepare_inputs() returns tuple of (inputs, input_length)."""
        result = runner._prepare_inputs(prompt="Hello")
        
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_prepare_inputs_tokenizes_text(self, runner):
        """_prepare_inputs() tokenizes the input text."""
        runner._prepare_inputs(prompt="Hello")
        
        runner.tokenizer.assert_called()

    def test_prepare_inputs_returns_dict_with_input_ids(self, runner):
        """_prepare_inputs() returns inputs with input_ids."""
        inputs, _ = runner._prepare_inputs(prompt="Hello")
        
        assert "input_ids" in inputs

    def test_prepare_inputs_returns_input_length(self, runner):
        """_prepare_inputs() returns correct input length."""
        _, input_length = runner._prepare_inputs(prompt="Hello")
        
        assert input_length == 5


class TestTransformersRunnerBuildGenerationKwargs:
    """Tests for TransformersRunner._build_generation_kwargs()"""

    @pytest.fixture
    def runner(self):
        """Create a TransformersRunner instance for testing."""
        handle = create_mock_model_handle(backend="transformers")
        return TransformersRunner(handle)

    @pytest.fixture
    def mock_inputs(self):
        """Create mock tokenized inputs."""
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        return MockTokenizerOutput(input_ids, attention_mask)

    def test_build_generation_kwargs_includes_input_ids(self, runner, mock_inputs):
        """_build_generation_kwargs() includes input_ids from inputs."""
        result = runner._build_generation_kwargs(mock_inputs, 0.7, 0.9)
        
        assert "input_ids" in result

    def test_build_generation_kwargs_includes_attention_mask(self, runner, mock_inputs):
        """_build_generation_kwargs() includes attention_mask from inputs."""
        result = runner._build_generation_kwargs(mock_inputs, 0.7, 0.9)
        
        assert "attention_mask" in result

    def test_build_generation_kwargs_includes_max_new_tokens(self, runner, mock_inputs):
        """_build_generation_kwargs() includes max_new_tokens."""
        runner.max_new_tokens = 512
        
        result = runner._build_generation_kwargs(mock_inputs, 0.7, 0.9)
        
        assert result["max_new_tokens"] == 512

    def test_build_generation_kwargs_do_sample_true_when_temp_positive(self, runner, mock_inputs):
        """_build_generation_kwargs() sets do_sample=True when temperature > 0."""
        result = runner._build_generation_kwargs(mock_inputs, 0.7, 0.9)
        
        assert result["do_sample"] is True

    def test_build_generation_kwargs_do_sample_false_when_temp_zero(self, runner, mock_inputs):
        """_build_generation_kwargs() sets do_sample=False when temperature = 0."""
        result = runner._build_generation_kwargs(mock_inputs, 0.0, 0.9)
        
        assert result["do_sample"] is False

    def test_build_generation_kwargs_includes_temperature_when_sampling(self, runner, mock_inputs):
        """_build_generation_kwargs() includes temperature when do_sample=True."""
        result = runner._build_generation_kwargs(mock_inputs, 0.7, 0.9)
        
        assert result["temperature"] == 0.7

    def test_build_generation_kwargs_includes_top_p_when_sampling(self, runner, mock_inputs):
        """_build_generation_kwargs() includes top_p when do_sample=True."""
        result = runner._build_generation_kwargs(mock_inputs, 0.7, 0.9)
        
        assert result["top_p"] == 0.9

    def test_build_generation_kwargs_excludes_sampling_params_when_not_sampling(self, runner, mock_inputs):
        """_build_generation_kwargs() excludes temperature/top_p when do_sample=False."""
        result = runner._build_generation_kwargs(mock_inputs, 0.0, 0.9)
        
        assert "temperature" not in result
        assert "top_p" not in result

    def test_build_generation_kwargs_includes_pad_token_id(self, runner, mock_inputs):
        """_build_generation_kwargs() includes pad_token_id."""
        runner._pad_token_id = 42
        
        result = runner._build_generation_kwargs(mock_inputs, 0.7, 0.9)
        
        assert result["pad_token_id"] == 42

    def test_build_generation_kwargs_includes_eos_token_id(self, runner, mock_inputs):
        """_build_generation_kwargs() includes eos_token_id."""
        runner._eos_token_id = 99
        
        result = runner._build_generation_kwargs(mock_inputs, 0.7, 0.9)
        
        assert result["eos_token_id"] == 99

    def test_build_generation_kwargs_includes_streamer_when_provided(self, runner, mock_inputs):
        """_build_generation_kwargs() includes streamer when provided."""
        mock_streamer = MagicMock()
        
        result = runner._build_generation_kwargs(mock_inputs, 0.7, 0.9, streamer=mock_streamer)
        
        assert result["streamer"] is mock_streamer

    def test_build_generation_kwargs_excludes_streamer_when_none(self, runner, mock_inputs):
        """_build_generation_kwargs() excludes streamer when None."""
        result = runner._build_generation_kwargs(mock_inputs, 0.7, 0.9, streamer=None)
        
        assert "streamer" not in result


class TestTransformersRunnerGenerateSync:
    """Tests for TransformersRunner._generate_sync()"""

    @pytest.fixture
    def runner(self):
        """Create a TransformersRunner instance for testing."""
        handle = create_mock_model_handle(backend="transformers")
        handle.model.generate.return_value = create_transformers_generate_output(
            input_length=5, output_length=15
        )
        return TransformersRunner(handle)

    @pytest.fixture
    def mock_inputs(self):
        """Create mock tokenized inputs."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        return MockTokenizerOutput(input_ids, attention_mask)

    def test_generate_sync_returns_tuple(self, runner, mock_inputs):
        """_generate_sync() returns tuple of (text, token_count)."""
        result = runner._generate_sync(mock_inputs, 5, 0.7, 0.9)
        
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generate_sync_returns_decoded_text(self, runner, mock_inputs):
        """_generate_sync() returns decoded response text."""
        runner.tokenizer.decode.return_value = "Generated text"
        
        text, _ = runner._generate_sync(mock_inputs, 5, 0.7, 0.9)
        
        assert text == "Generated text"

    def test_generate_sync_returns_token_count(self, runner, mock_inputs):
        """_generate_sync() returns number of generated tokens."""
        _, token_count = runner._generate_sync(mock_inputs, 5, 0.7, 0.9)
        
        assert isinstance(token_count, int)
        assert token_count == 10

    def test_generate_sync_calls_model_generate(self, runner, mock_inputs):
        """_generate_sync() calls model.generate()."""
        runner._generate_sync(mock_inputs, 5, 0.7, 0.9)
        
        runner.model.generate.assert_called_once()

    def test_generate_sync_decodes_with_skip_special_tokens(self, runner, mock_inputs):
        """_generate_sync() decodes with skip_special_tokens=True."""
        runner._generate_sync(mock_inputs, 5, 0.7, 0.9)
        
        call_kwargs = runner.tokenizer.decode.call_args[1]
        assert call_kwargs["skip_special_tokens"] is True