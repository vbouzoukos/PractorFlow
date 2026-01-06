"""
Tests for TransformersRunner thinking/reasoning trace parsing.

Tests the _parse_thinking_response() method, _chat_template_starts_with_think()
detection, and integration with generate() and generate_stream() for models 
like Nemotron 3 that use <think>...</think> tags for chain-of-thought reasoning.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import torch

from practorflow.llm.transformers_runner import TransformersRunner
from practorflow.llm.tools.base import ToolResult
from tests.practorflow.llm.common_runner import create_mock_model_handle


def create_mock_transformers_handle(
    model_name: str = "test-model",
    max_context_length: int = 4096,
    chat_template: str = None,
) -> MagicMock:
    """
    Create a mock ModelHandle configured for transformers backend.
    
    Args:
        model_name: Model name for the config.
        max_context_length: Maximum context length.
        chat_template: Optional chat template string.
    
    Returns:
        MagicMock configured as a transformers ModelHandle.
    """
    handle = create_mock_model_handle(
        backend="transformers",
        model_name=model_name,
        max_context_length=max_context_length,
        chat_template=chat_template,
    )
    
    # Create mock transformers model
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate = MagicMock()
    
    # Create mock tokenizer
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.chat_template = chat_template
    tokenizer.apply_chat_template = MagicMock(return_value="formatted input")
    tokenizer.decode = MagicMock(return_value="decoded output")
    
    # Mock tokenizer call to return tensor-like object
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(return_value=mock_inputs)
    mock_inputs.to = MagicMock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    mock_inputs_result = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_inputs_result_obj = MagicMock()
    mock_inputs_result_obj.__getitem__ = lambda self, key: mock_inputs_result[key]
    mock_inputs_result_obj.to = MagicMock(return_value=mock_inputs_result)
    tokenizer.return_value = mock_inputs_result_obj
    
    handle.model = model
    handle.tokenizer = tokenizer
    
    return handle


def create_mock_generation_output(response_text: str, input_length: int = 3):
    """
    Create mock output tensor from model.generate().
    
    Args:
        response_text: The text that will be decoded.
        input_length: Length of input tokens.
    
    Returns:
        Mock tensor representing generation output.
    """
    # Create output tensor with input + generated tokens
    total_length = input_length + 10  # Assume 10 generated tokens
    output_tensor = torch.tensor([[i for i in range(total_length)]])
    return output_tensor


class TestThinkingModelDetection:
    """Tests for TransformersRunner._chat_template_starts_with_think()"""

    def test_detects_generation_prompt_with_think(self):
        """Detects thinking model when generation_prompt ends with <think>."""
        chat_template = """
        {% if add_generation_prompt %}
        {{ '<|assistant|>' }}<think>
        {%- endif %}
        """
        handle = create_mock_transformers_handle(chat_template=chat_template)
        runner = TransformersRunner(handle)
        
        assert runner._thinking_model is True

    def test_detects_assistant_prefix_with_think(self):
        """Detects thinking model when assistant prefix contains <think>."""
        chat_template = """
        {% for message in messages %}
        {% if message.role == 'assistant' %}<think>{{ message.content }}
        {% endif %}
        {% endfor %}
        """
        handle = create_mock_transformers_handle(chat_template=chat_template)
        runner = TransformersRunner(handle)
        
        assert runner._thinking_model is True

    def test_detects_template_ending_with_think(self):
        """Detects thinking model when template ends with <think>."""
        chat_template = "{{ messages }}{{ '<|assistant|>' }}<think>"
        handle = create_mock_transformers_handle(chat_template=chat_template)
        runner = TransformersRunner(handle)
        
        assert runner._thinking_model is True

    def test_no_detection_for_regular_template(self):
        """Returns False for templates without <think> in generation prompt."""
        chat_template = """
        {% for message in messages %}
        {{ message.role }}: {{ message.content }}
        {% endfor %}
        """
        handle = create_mock_transformers_handle(chat_template=chat_template)
        runner = TransformersRunner(handle)
        
        assert runner._thinking_model is False

    def test_no_detection_for_none_template(self):
        """Returns False when chat_template is None."""
        handle = create_mock_transformers_handle(chat_template=None)
        runner = TransformersRunner(handle)
        
        assert runner._thinking_model is False

    def test_no_detection_for_empty_template(self):
        """Returns False when chat_template is empty string."""
        handle = create_mock_transformers_handle(chat_template="")
        runner = TransformersRunner(handle)
        
        assert runner._thinking_model is False

    def test_think_far_from_assistant_not_detected(self):
        """Does not detect if <think> is more than 50 chars from assistant."""
        # <think> is more than 50 characters after "assistant"
        chat_template = """
        {% if message.role == 'assistant' %}
        This is a very long string that goes on and on and on for more than fifty characters
        <think>
        {% endif %}
        """
        handle = create_mock_transformers_handle(chat_template=chat_template)
        runner = TransformersRunner(handle)
        
        assert runner._thinking_model is False

    def test_detects_template_ending_with_think(self):
        """
        Explicitly covers:
            template_str.rstrip().endswith('<think>')
        """
        chat_template = "Some content before\n   <think>   \n"
        handle = create_mock_transformers_handle(chat_template=chat_template)

        runner = TransformersRunner(handle)

        assert runner._thinking_model is True

    def test_exception_in_chat_template_detection_returns_false(self, caplog):
        """
        Explicitly covers:
            except Exception as e:
                return False
        """
        tokenizer = MagicMock()
        tokenizer.chat_template = MagicMock(side_effect=RuntimeError("boom"))

        handle = create_mock_transformers_handle(chat_template=None)
        handle.tokenizer = tokenizer

        with caplog.at_level("WARNING"):
            runner = TransformersRunner(handle)
            result = runner._chat_template_starts_with_think()

        assert result is False
        assert any(
            "Error detecting thinking model" in record.message
            for record in caplog.records
        )
        
    def test_exception_in_chat_template_detection_returns_false(self):
        class BadTemplate:
            def __str__(self):
                raise RuntimeError("boom")

        handle = create_mock_transformers_handle(chat_template=None)
        handle.tokenizer.chat_template = BadTemplate()

        runner = TransformersRunner(handle)

        assert runner._chat_template_starts_with_think() is False
        assert runner._thinking_model is False
            
class TestTransformersRunnerParseThinkingResponse:
    """Tests for TransformersRunner._parse_thinking_response()"""

    @pytest.fixture
    def runner(self):
        """Create a TransformersRunner instance for testing."""
        handle = create_mock_transformers_handle()
        return TransformersRunner(handle)

    def test_parse_thinking_response_with_think_tags(self, runner):
        """Extracts thinking content from <think> tags."""
        text = "<think>Let me analyze this step by step.</think>The answer is 42."
        
        thinking, reply = runner._parse_thinking_response(text)
        
        assert thinking == "Let me analyze this step by step."
        assert reply == "The answer is 42."

    def test_parse_thinking_response_without_think_tags(self, runner):
        """Returns None for thinking when no tags present."""
        text = "The answer is simply 42."
        
        thinking, reply = runner._parse_thinking_response(text)
        
        assert thinking is None
        assert reply == "The answer is simply 42."

    def test_parse_thinking_response_multiline_thinking(self, runner):
        """Handles multiline content inside think tags."""
        text = """<think>
First, I need to consider the problem.
Then, I'll calculate the result.
Finally, I'll verify my answer.
</think>The answer is 42."""
        
        thinking, reply = runner._parse_thinking_response(text)
        
        assert "First, I need to consider the problem." in thinking
        assert "Then, I'll calculate the result." in thinking
        assert "Finally, I'll verify my answer." in thinking
        assert reply == "The answer is 42."

    def test_parse_thinking_response_empty_think_tags(self, runner):
        """Handles empty think tags."""
        text = "<think></think>The answer is 42."
        
        thinking, reply = runner._parse_thinking_response(text)
        
        assert thinking is None
        assert reply == "The answer is 42."

    def test_parse_thinking_response_multiple_think_blocks(self, runner):
        """Combines multiple think blocks."""
        text = "<think>First thought.</think>Middle text.<think>Second thought.</think>Final answer."
        
        thinking, reply = runner._parse_thinking_response(text)
        
        assert "First thought." in thinking
        assert "Second thought." in thinking
        assert reply == "Middle text.Final answer."

    def test_parse_thinking_response_empty_text(self, runner):
        """Handles empty input text."""
        thinking, reply = runner._parse_thinking_response("")
        
        assert thinking is None
        assert reply == ""

    def test_parse_thinking_response_none_text(self, runner):
        """Handles None input text."""
        thinking, reply = runner._parse_thinking_response(None)
        
        assert thinking is None
        assert reply == ""

    def test_parse_thinking_response_only_think_tags(self, runner):
        """Handles response with only thinking content."""
        text = "<think>Just reasoning, no final answer.</think>"
        
        thinking, reply = runner._parse_thinking_response(text)
        
        assert thinking == "Just reasoning, no final answer."
        assert reply == ""

    def test_parse_thinking_response_whitespace_handling(self, runner):
        """Strips whitespace from thinking and reply."""
        text = "  <think>  Some thinking  </think>  Some reply  "
        
        thinking, reply = runner._parse_thinking_response(text)
        
        assert thinking == "Some thinking"
        assert reply == "Some reply"

    def test_parse_thinking_response_nested_angle_brackets(self, runner):
        """Handles content with other angle brackets inside."""
        text = "<think>Compare a < b and c > d.</think>Result: a < b is true."
        
        thinking, reply = runner._parse_thinking_response(text)
        
        assert thinking == "Compare a < b and c > d."
        assert reply == "Result: a < b is true."

    def test_parse_thinking_response_code_in_thinking(self, runner):
        """Handles code snippets inside think tags."""
        text = "<think>```python\nprint('hello')\n```</think>Here's the code explanation."
        
        thinking, reply = runner._parse_thinking_response(text)
        
        assert "```python" in thinking
        assert "print('hello')" in thinking
        assert reply == "Here's the code explanation."

class TestTransformersRunnerGenerateWithThinking:
    """Tests for generate() with thinking/reasoning traces."""

    @pytest.fixture
    def runner(self):
        """Create a TransformersRunner instance for testing."""
        handle = create_mock_transformers_handle()
        runner = TransformersRunner(handle)
        return runner

    @pytest.mark.asyncio
    async def test_generate_extracts_thinking_from_response(self, runner):
        """generate() extracts thinking and returns it separately."""
        response_text = "<think>Step 1: Analyze. Step 2: Calculate.</think>The answer is 42."
        runner.tokenizer.decode.return_value = response_text
        runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        result = await runner.generate(prompt="What is the answer?")
        
        assert "thinking" in result
        assert "Step 1: Analyze" in result["thinking"]
        assert "Step 2: Calculate" in result["thinking"]
        assert result["reply"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_generate_no_thinking_key_when_absent(self, runner):
        """generate() does not include thinking key when no think tags."""
        response_text = "The answer is 42."
        runner.tokenizer.decode.return_value = response_text
        runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        result = await runner.generate(prompt="What is the answer?")
        
        assert "thinking" not in result
        assert result["reply"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_generate_handles_empty_thinking(self, runner):
        """generate() handles empty think tags correctly."""
        response_text = "<think></think>Direct answer."
        runner.tokenizer.decode.return_value = response_text
        runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        result = await runner.generate(prompt="Quick question")
        
        assert "thinking" not in result
        assert result["reply"] == "Direct answer."

    @pytest.mark.asyncio
    async def test_generate_with_thinking_and_context(self, runner):
        """generate() handles both thinking and context correctly."""
        response_text = "<think>Analyzing the document.</think>Based on the doc, the answer is 42."
        runner.tokenizer.decode.return_value = response_text
        runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        runner._pending_context = "Document content here."
        runner.tool_registry._last_result = ToolResult(
            success=True,
            data="test",
            metadata={"source": "test_doc.pdf"}
        )
        
        result = await runner.generate(prompt="What does the doc say?")
        
        assert "thinking" in result
        assert "Analyzing the document" in result["thinking"]
        assert result["reply"] == "Based on the doc, the answer is 42."
        assert result["context_used"] == "Document content here."
        assert result["search_metadata"]["source"] == "test_doc.pdf"

    @pytest.mark.asyncio
    async def test_generate_multiline_thinking(self, runner):
        """generate() handles multiline thinking content."""
        response_text = """<think>
Step 1: Read the question.
Step 2: Think about it.
Step 3: Formulate answer.
</think>The final answer is 42."""
        runner.tokenizer.decode.return_value = response_text
        runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        result = await runner.generate(prompt="Complex question")
        
        assert "thinking" in result
        assert "Step 1" in result["thinking"]
        assert "Step 2" in result["thinking"]
        assert "Step 3" in result["thinking"]
        assert result["reply"] == "The final answer is 42."

    @pytest.mark.asyncio
    async def test_generate_multiple_think_blocks(self, runner):
        """generate() combines multiple think blocks."""
        response_text = "<think>First analysis.</think>Intermediate.<think>Second analysis.</think>Final answer."
        runner.tokenizer.decode.return_value = response_text
        runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        result = await runner.generate(prompt="Question")
        
        assert "thinking" in result
        assert "First analysis" in result["thinking"]
        assert "Second analysis" in result["thinking"]
        assert result["reply"] == "Intermediate.Final answer."


class TestTransformersRunnerGenerateThinkingModel:
    """Tests for generate() when _thinking_model=True (detected thinking model)."""

    @pytest.fixture
    def thinking_runner(self):
        """Create a TransformersRunner with _thinking_model=True."""
        chat_template = "{{ messages }}{{ '<|assistant|>' }}<think>"
        handle = create_mock_transformers_handle(chat_template=chat_template)
        runner = TransformersRunner(handle)
        assert runner._thinking_model is True
        return runner

    @pytest.mark.asyncio
    async def test_thinking_model_extracts_thinking_with_close_tag(self, thinking_runner):
        """Thinking model extracts thinking when </think> is present."""
        response_text = "Reasoning step by step.</think>The answer is 42."
        thinking_runner.tokenizer.decode.return_value = response_text
        thinking_runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        result = await thinking_runner.generate(prompt="Question")
        
        assert "thinking" in result
        assert "Reasoning step by step" in result["thinking"]
        assert result["reply"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_thinking_model_with_leading_think_tag(self, thinking_runner):
        """Thinking model strips leading <think> tag."""
        response_text = "<think>Reasoning here.</think>Final answer."
        thinking_runner.tokenizer.decode.return_value = response_text
        thinking_runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        result = await thinking_runner.generate(prompt="Question")
        
        assert "thinking" in result
        assert result["thinking"] == "Reasoning here."
        assert result["reply"] == "Final answer."

    @pytest.mark.asyncio
    async def test_thinking_model_no_close_tag_treats_as_normal(self, thinking_runner):
        """Thinking model without </think> treats response as normal text."""
        response_text = "This is just a normal response without closing tag."
        thinking_runner.tokenizer.decode.return_value = response_text
        thinking_runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        result = await thinking_runner.generate(prompt="Question")
        
        assert "thinking" not in result
        assert result["reply"] == "This is just a normal response without closing tag."

    @pytest.mark.asyncio
    async def test_thinking_model_no_close_tag_strips_leading_think(self, thinking_runner):
        """Thinking model without </think> strips leading <think> from reply."""
        response_text = "<think>Normal response that template prepended think to."
        thinking_runner.tokenizer.decode.return_value = response_text
        thinking_runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        result = await thinking_runner.generate(prompt="Question")
        
        assert "thinking" not in result
        assert result["reply"] == "Normal response that template prepended think to."

    @pytest.mark.asyncio
    async def test_thinking_model_empty_thinking_content(self, thinking_runner):
        """Thinking model with empty thinking content."""
        response_text = "</think>Direct answer."
        thinking_runner.tokenizer.decode.return_value = response_text
        thinking_runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        result = await thinking_runner.generate(prompt="Question")
        
        assert "thinking" not in result
        assert result["reply"] == "Direct answer."

    @pytest.mark.asyncio
    async def test_thinking_model_only_thinking_no_reply(self, thinking_runner):
        """Thinking model with only thinking content and no reply."""
        response_text = "Just reasoning content.</think>"
        thinking_runner.tokenizer.decode.return_value = response_text
        thinking_runner.model.generate.return_value = create_mock_generation_output(response_text)
        
        result = await thinking_runner.generate(prompt="Question")
        
        assert "thinking" in result
        assert result["thinking"] == "Just reasoning content."
        assert result["reply"] == ""


class TestTransformersRunnerGenerateStreamWithThinking:
    """Tests for generate_stream() with thinking/reasoning traces."""

    @pytest.fixture
    def runner(self):
        """Create a TransformersRunner instance for testing."""
        handle = create_mock_transformers_handle()
        runner = TransformersRunner(handle)
        return runner

    @pytest.mark.asyncio
    async def test_generate_stream_filters_thinking_content(self, runner):
        """generate_stream() filters out thinking content from streamed chunks."""
        text_chunks = ["<think>", "Reasoning here.", "</think>", "Final answer."]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert "<think>" not in text_content
        assert "</think>" not in text_content
        assert "Reasoning here." not in text_content
        assert "Final answer." in text_content

    @pytest.mark.asyncio
    async def test_generate_stream_includes_thinking_in_final_chunk(self, runner):
        """generate_stream() includes thinking in final chunk search_metadata."""
        text_chunks = ["<think>", "Step by step reasoning.", "</think>", "The answer."]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        final_chunk = chunks[-1]
        assert final_chunk.finished is True
        assert final_chunk.search_metadata is not None
        assert "thinking" in final_chunk.search_metadata
        assert "Step by step reasoning" in final_chunk.search_metadata["thinking"]

    @pytest.mark.asyncio
    async def test_generate_stream_no_thinking_metadata_when_absent(self, runner):
        """generate_stream() final chunk has no thinking in metadata when tags absent."""
        text_chunks = ["Just", " a", " response."]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert "Just a response." in text_content
        
        final_chunk = chunks[-1]
        assert final_chunk.search_metadata is None or "thinking" not in final_chunk.search_metadata

    @pytest.mark.asyncio
    async def test_generate_stream_with_context_and_thinking(self, runner):
        """generate_stream() handles both context metadata and thinking."""
        text_chunks = ["<think>Analyzing document.</think>", "The answer from doc."]
        
        runner._pending_context = "Document content."
        runner.tool_registry._last_result = ToolResult(
            success=True,
            data="test",
            metadata={"source": "test_doc.pdf"}
        )
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        final_chunk = chunks[-1]
        assert final_chunk.context_used == "Document content."
        assert final_chunk.search_metadata is not None
        assert "thinking" in final_chunk.search_metadata
        assert "source" in final_chunk.search_metadata

    @pytest.mark.asyncio
    async def test_generate_stream_inline_thinking_tags(self, runner):
        """generate_stream() handles thinking tags inline with content."""
        text_chunks = ["Start <think>reasoning</think> end."]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert "<think>" not in text_content
        assert "</think>" not in text_content
        assert "reasoning" not in text_content
        assert "Start" in text_content
        assert "end." in text_content
        
        final_chunk = chunks[-1]
        assert final_chunk.search_metadata is not None
        assert final_chunk.search_metadata["thinking"] == "reasoning"

    @pytest.mark.asyncio
    async def test_generate_stream_buffers_partial_think_tag(self, runner):
        """generate_stream() buffers text when last 6 chars might be start of <think> tag."""
        text_chunks = ["A<think", "er>"]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert "A<thinker>" in text_content

    @pytest.mark.asyncio
    async def test_generate_stream_yields_pending_text_incomplete_thinking(self, runner):
        """generate_stream() yields pending_text when stream ends mid-thinking block.
        
        This tests the post-loop code that yields remaining pending_text
        when the stream ends with an incomplete <think> block (no closing </think>).
        """
        text_chunks = ["<think>", "Incomplete reasoning without closing tag"]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert "Incomplete reasoning without closing tag" in text_content

    @pytest.mark.asyncio
    async def test_generate_stream_preserves_context_metadata_without_thinking(self, runner):
        """generate_stream() preserves context_metadata when no thinking present."""
        text_chunks = ["Just a regular response."]
        
        runner._pending_context = "Document content."
        runner.tool_registry._last_result = ToolResult(
            success=True,
            data="test",
            metadata={"source": "test_doc.pdf", "score": 0.95}
        )
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        final_chunk = chunks[-1]
        assert final_chunk.context_used == "Document content."
        assert final_chunk.search_metadata is not None
        assert final_chunk.search_metadata["source"] == "test_doc.pdf"
        assert final_chunk.search_metadata["score"] == 0.95
        assert "thinking" not in final_chunk.search_metadata

    @pytest.mark.asyncio
    async def test_generate_stream_multiline_thinking_across_chunks(self, runner):
        """generate_stream() handles multiline thinking split across chunks."""
        text_chunks = [
            "<think>",
            "Line 1 of reasoning.\n",
            "Line 2 of reasoning.\n",
            "Line 3 of reasoning.",
            "</think>",
            "Final answer here.",
        ]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert "Line 1" not in text_content
        assert "Line 2" not in text_content
        assert "Line 3" not in text_content
        assert "Final answer here." in text_content
        
        final_chunk = chunks[-1]
        assert final_chunk.search_metadata is not None
        assert "Line 1" in final_chunk.search_metadata["thinking"]
        assert "Line 2" in final_chunk.search_metadata["thinking"]
        assert "Line 3" in final_chunk.search_metadata["thinking"]

    @pytest.mark.asyncio
    async def test_generate_stream_only_thinking_no_reply(self, runner):
        """generate_stream() handles response with only thinking content."""
        text_chunks = ["<think>Just reasoning, no answer.</think>"]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert text_content == ""
        
        final_chunk = chunks[-1]
        assert final_chunk.search_metadata is not None
        assert final_chunk.search_metadata["thinking"] == "Just reasoning, no answer."

    @pytest.mark.asyncio
    async def test_generate_stream_code_in_thinking(self, runner):
        """generate_stream() handles code blocks inside thinking tags."""
        text_chunks = [
            "<think>```python\nprint('hello')\n```</think>",
            "Here's the explanation.",
        ]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert "```python" not in text_content
        assert "Here's the explanation." in text_content
        
        final_chunk = chunks[-1]
        assert "```python" in final_chunk.search_metadata["thinking"]


class TestTransformersRunnerGenerateStreamThinkingModel:
    """Tests for generate_stream() when _thinking_model=True (detected thinking model)."""

    @pytest.fixture
    def thinking_runner(self):
        """Create a TransformersRunner with _thinking_model=True."""
        chat_template = "{{ messages }}{{ '<|assistant|>' }}<think>"
        handle = create_mock_transformers_handle(chat_template=chat_template)
        runner = TransformersRunner(handle)
        assert runner._thinking_model is True
        return runner

    @pytest.mark.asyncio
    async def test_thinking_model_stream_starts_in_thinking_mode(self, thinking_runner):
        """Thinking model stream starts in_thinking=True, filters initial content."""
        text_chunks = ["Reasoning content.", "</think>", "Final answer."]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            thinking_runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in thinking_runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert "Reasoning content." not in text_content
        assert "Final answer." in text_content
        
        final_chunk = chunks[-1]
        assert final_chunk.search_metadata is not None
        assert "Reasoning content" in final_chunk.search_metadata["thinking"]

    @pytest.mark.asyncio
    async def test_thinking_model_stream_no_close_tag_yields_all_as_text(self, thinking_runner):
        """Thinking model without </think> yields all accumulated content as text."""
        text_chunks = ["This is normal content without thinking close tag."]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            thinking_runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in thinking_runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert "This is normal content without thinking close tag." in text_content
        
        final_chunk = chunks[-1]
        assert final_chunk.search_metadata is None or "thinking" not in final_chunk.search_metadata

    @pytest.mark.asyncio
    async def test_thinking_model_stream_with_proper_thinking_block(self, thinking_runner):
        """Thinking model with proper </think> captures thinking in metadata."""
        text_chunks = ["Step 1. Step 2.", "</think>", "The answer is 42."]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            thinking_runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in thinking_runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert "Step 1" not in text_content
        assert "The answer is 42." in text_content
        
        final_chunk = chunks[-1]
        assert final_chunk.search_metadata is not None
        assert "Step 1. Step 2." in final_chunk.search_metadata["thinking"]

    @pytest.mark.asyncio
    async def test_thinking_model_stream_empty_thinking_then_reply(self, thinking_runner):
        """Thinking model with immediate </think> yields reply correctly."""
        text_chunks = ["</think>", "Direct answer."]
        
        streamer_values = iter(text_chunks)
        
        with patch('practorflow.llm.transformers_runner.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer_class.return_value = mock_streamer
            
            thinking_runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            chunks = []
            async for chunk in thinking_runner.generate_stream(prompt="Question"):
                chunks.append(chunk)
        
        text_content = "".join(c.text for c in chunks if c.text)
        assert "Direct answer." in text_content
        
        final_chunk = chunks[-1]
        assert final_chunk.search_metadata is None or "thinking" not in final_chunk.search_metadata

    @pytest.mark.asyncio
    async def test_stream_enters_partial_close_tag_buffer_branch(self, thinking_runner):
        # IMPORTANT:
        # We explicitly enter thinking via <think> in the stream,
        # and end a chunk with EXACTLY 7 chars starting with "<" => "</think"

        text_chunks = [
            "<think>ABC</think",  # pending_text[-7:] == "</think"  ✅
            ">",                  # completes </think>
            "DONE",
        ]

        streamer_values = iter(text_chunks)

        with patch("practorflow.llm.transformers_runner.TextIteratorStreamer") as mock_streamer:
            streamer = MagicMock()
            streamer.__iter__ = MagicMock(side_effect=lambda: streamer_values)
            mock_streamer.return_value = streamer

            thinking_runner.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

            chunks = []
            async for chunk in thinking_runner.generate_stream(prompt="Q"):
                chunks.append(chunk)

        # Minimal assertion: stream completed
        assert chunks[-1].finished is True
