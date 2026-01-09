"""
Tests for LlamaCppRunner thinking/reasoning trace parsing.

Tests the _parse_thinking_response() method, _chat_template_starts_with_think()
detection, and integration with generate()/generate_stream() for models like
DeepSeek-R1 and QwQ that use <think>...</think> tags for chain-of-thought reasoning.
"""

import pytest

from practorflow.llm.llama_cpp_runner import LlamaCppRunner
from tests.practorflow.llm.common_runner import (
    create_mock_model_handle,
    create_llama_completion_response,
    create_llama_stream_chunks,
    create_tool_call,
)


class TestLlamaCppRunnerChatTemplateStartsWithThink:
    """Tests for LlamaCppRunner._chat_template_starts_with_think()"""

    def test_detects_generation_prompt_with_think(self):
        """Detects thinking model when generation_prompt ends with <think>."""
        chat_template = """
{%- for message in messages %}
{{ message['content'] }}
{%- endfor %}
{%- if add_generation_prompt %}
<|assistant|>
<think>{%- endif %}
"""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template=chat_template,
        )
        runner = LlamaCppRunner(handle)

        assert runner._thinking_model is True

    def test_detects_assistant_prefix_with_think(self):
        """Detects thinking model when assistant prefix contains <think>."""
        chat_template = """
{%- if messages[0]['role'] == 'system' %}
<|system|>{{ messages[0]['content'] }}
{%- endif %}
<|assistant|>
<think>
"""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template=chat_template,
        )
        runner = LlamaCppRunner(handle)

        assert runner._thinking_model is True

    def test_detects_template_ending_with_think(self):
        """Detects thinking model when template ends with <think>."""
        chat_template = "{{ messages }}}}}<think>"
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template=chat_template,
        )
        runner = LlamaCppRunner(handle)

        assert runner._thinking_model is True

    def test_returns_false_for_normal_template(self):
        """Returns False for templates without <think> in generation prompt."""
        chat_template = """
{%- for message in messages %}
<|{{ message.role }}|>{{ message.content }}
{%- endfor %}
{%- if add_generation_prompt %}<|assistant|>{%- endif %}
"""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template=chat_template,
        )
        runner = LlamaCppRunner(handle)

        assert runner._thinking_model is False

    def test_returns_false_when_no_template(self):
        """Returns False when no chat template is available."""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template=None,
        )
        runner = LlamaCppRunner(handle)

        assert runner._thinking_model is False

    def test_returns_false_for_template_with_tool_but_no_think(self):
        """Returns False for tool-calling templates without <think>."""
        chat_template = """
{%- if tools %}Available tools: {{ tools }}{%- endif %}
<|assistant|>
"""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template=chat_template,
        )
        runner = LlamaCppRunner(handle)

        assert runner._thinking_model is False

    def test_think_far_from_assistant_not_detected(self):
        """Does not detect thinking when <think> is far from assistant marker."""
        chat_template = "<|assistant|>" + ("x" * 60) + "<think>" + ("y" * 20)
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template=chat_template,
        )
        runner = LlamaCppRunner(handle)

        assert runner._thinking_model is False


class TestLlamaCppRunnerParseThinkingResponse:
    """Tests for LlamaCppRunner._parse_thinking_response()"""

    @pytest.fixture
    def runner(self):
        """Create a LlamaCppRunner instance for testing."""
        handle = create_mock_model_handle(backend="llama_cpp")
        return LlamaCppRunner(handle)

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
        text = (
            "<think>```python\nprint('hello')\n```</think>Here's the code explanation."
        )

        thinking, reply = runner._parse_thinking_response(text)

        assert "```python" in thinking
        assert "print('hello')" in thinking
        assert reply == "Here's the code explanation."


class TestLlamaCppRunnerGenerateWithThinking:
    """Tests for generate() with thinking/reasoning traces."""

    @pytest.fixture
    def runner(self):
        """Create a LlamaCppRunner instance for testing."""
        handle = create_mock_model_handle(backend="llama_cpp")
        return LlamaCppRunner(handle)

    @pytest.mark.asyncio
    async def test_generate_extracts_thinking_from_response(self, runner):
        """generate() extracts thinking and returns it separately."""
        runner.model.create_chat_completion.return_value = create_llama_completion_response(
            content="<think>Step 1: Analyze. Step 2: Calculate.</think>The answer is 42."
        )

        result = await runner.generate(prompt="What is the answer?")

        assert "thinking" in result
        assert "Step 1: Analyze" in result["thinking"]
        assert "Step 2: Calculate" in result["thinking"]
        assert result["reply"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_generate_no_thinking_key_when_absent(self, runner):
        """generate() does not include thinking key when no think tags."""
        runner.model.create_chat_completion.return_value = (
            create_llama_completion_response(content="The answer is 42.")
        )

        result = await runner.generate(prompt="What is the answer?")

        assert "thinking" not in result
        assert result["reply"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_generate_handles_empty_thinking(self, runner):
        """generate() handles empty think tags correctly."""
        runner.model.create_chat_completion.return_value = (
            create_llama_completion_response(content="<think></think>Direct answer.")
        )

        result = await runner.generate(prompt="Quick question")

        assert "thinking" not in result
        assert result["reply"] == "Direct answer."

    @pytest.mark.asyncio
    async def test_generate_with_thinking_and_tool_calls(self, runner):
        """generate() handles both thinking and tool calls."""
        tool_calls = [create_tool_call("get_weather", {"location": "Paris"})]
        runner.model.create_chat_completion.return_value = (
            create_llama_completion_response(
                content="<think>I need to check the weather.</think>",
                tool_calls=tool_calls,
            )
        )

        result = await runner.generate(prompt="What's the weather in Paris?")

        assert "thinking" in result
        assert "I need to check the weather" in result["thinking"]
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1


class TestLlamaCppRunnerGenerateThinkingModelNoCloseTag:
    """Tests for generate() with thinking models that don't return </think>."""

    @pytest.fixture
    def thinking_runner(self):
        """Create a LlamaCppRunner for a thinking model."""
        chat_template = """
{%- if add_generation_prompt %}<|assistant|><think>{%- endif %}
"""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template=chat_template,
        )
        return LlamaCppRunner(handle)

    @pytest.mark.asyncio
    async def test_thinking_model_no_close_tag_treats_as_normal(self, thinking_runner):
        """Thinking model without </think> treats response as normal text."""
        thinking_runner.model.create_chat_completion.return_value = (
            create_llama_completion_response(
                content="This is just a normal response without thinking tags."
            )
        )

        result = await thinking_runner.generate(prompt="Hello")

        assert "thinking" not in result
        assert (
            result["reply"] == "This is just a normal response without thinking tags."
        )

    @pytest.mark.asyncio
    async def test_thinking_model_with_close_tag_extracts_thinking(
        self, thinking_runner
    ):
        """Thinking model with </think> properly extracts thinking."""
        thinking_runner.model.create_chat_completion.return_value = (
            create_llama_completion_response(
                content="Step by step reasoning here.</think>The final answer."
            )
        )

        result = await thinking_runner.generate(prompt="Question")

        assert "thinking" in result
        assert "Step by step reasoning here." in result["thinking"]
        assert result["reply"] == "The final answer."

    @pytest.mark.asyncio
    async def test_thinking_model_strips_leading_think_tag(self, thinking_runner):
        """Thinking model strips leading <think> tag from normal response."""
        thinking_runner.model.create_chat_completion.return_value = (
            create_llama_completion_response(
                content="<think>This is text without a closing tag."
            )
        )

        result = await thinking_runner.generate(prompt="Hello")

        assert "thinking" not in result
        assert result["reply"] == "This is text without a closing tag."


class TestLlamaCppRunnerExtractContentFromResponse:
    """Tests for LlamaCppRunner._extract_content_from_response()"""

    @pytest.fixture
    def runner(self):
        """Create a LlamaCppRunner instance for testing."""
        handle = create_mock_model_handle(backend="llama_cpp")
        return LlamaCppRunner(handle)

    def test_extract_content_from_valid_response(self, runner):
        """Extracts content from standard response structure."""
        response = create_llama_completion_response(content="Hello, world!")

        content = runner._extract_content_from_response(response)

        assert content == "Hello, world!"

    def test_extract_content_from_empty_choices(self, runner):
        """Returns empty string when choices is empty."""
        response = {"choices": []}

        content = runner._extract_content_from_response(response)

        assert content == ""

    def test_extract_content_from_missing_content(self, runner):
        """Returns empty string when content is missing."""
        response = {"choices": [{"message": {}}]}

        content = runner._extract_content_from_response(response)

        assert content == ""

    def test_extract_content_from_none_content(self, runner):
        """Returns empty string when content is None."""
        response = {"choices": [{"message": {"content": None}}]}

        content = runner._extract_content_from_response(response)

        assert content == ""

    def test_extract_content_with_thinking_tags(self, runner):
        """Extracts content including thinking tags (parsing done separately)."""
        response = create_llama_completion_response(
            content="<think>Reasoning</think>Answer"
        )

        content = runner._extract_content_from_response(response)

        assert content == "<think>Reasoning</think>Answer"


class TestLlamaCppRunnerGenerateStreamWithThinking:
    """Tests for generate_stream() with thinking/reasoning traces."""

    @pytest.fixture
    def runner(self):
        """Create a LlamaCppRunner instance for testing."""
        handle = create_mock_model_handle(backend="llama_cpp")
        return LlamaCppRunner(handle)

    @pytest.mark.asyncio
    async def test_generate_stream_filters_thinking_from_chunks(self, runner):
        """generate_stream() does not yield thinking content in text chunks."""
        stream_chunks = create_llama_stream_chunks(
            [
                "<think>",
                "Reasoning here.",
                "</think>",
                "Final answer.",
            ]
        )
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Question"):
            chunks.append(chunk)

        text_content = "".join(c.text for c in chunks if c.text)
        assert "Reasoning here." not in text_content
        assert "Final answer." in text_content

    @pytest.mark.asyncio
    async def test_generate_stream_includes_thinking_in_final_chunk(self, runner):
        """generate_stream() includes thinking in final chunk search_metadata."""
        stream_chunks = create_llama_stream_chunks(
            [
                "<think>",
                "Step by step reasoning.",
                "</think>",
                "The answer.",
            ]
        )
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

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
        stream_chunks = create_llama_stream_chunks(["Just", " a", " response."])
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Question"):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        if final_chunk.search_metadata:
            assert "thinking" not in final_chunk.search_metadata

    @pytest.mark.asyncio
    async def test_generate_stream_thinking_with_existing_context_metadata(
        self, runner
    ):
        """generate_stream() merges thinking with existing context metadata."""
        from practorflow.llm.tools.base import ToolResult

        runner._pending_context = "Document content."
        runner.tool_registry._last_result = ToolResult(
            success=True, data="test", metadata={"source": "test_doc.pdf"}
        )

        stream_chunks = create_llama_stream_chunks(
            [
                "<think>Analyzing document.</think>",
                "The answer from doc.",
            ]
        )
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

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
        stream_chunks = create_llama_stream_chunks(
            [
                "Start <think>reasoning</think> end.",
            ]
        )
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Question"):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.search_metadata is not None
        assert final_chunk.search_metadata["thinking"] == "reasoning"

    @pytest.mark.asyncio
    async def test_generate_stream_buffers_partial_think_tag(self, runner):
        """generate_stream() buffers text when token might be start of <think> tag."""
        stream_chunks = create_llama_stream_chunks(
            [
                "A<think",
                "er>",
            ]
        )
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Question"):
            chunks.append(chunk)

        text_content = "".join(c.text for c in chunks if c.text)
        assert "A" in text_content
        assert "thinker>" in text_content

    @pytest.mark.asyncio
    async def test_generate_stream_yields_pending_text_incomplete_thinking(
        self, runner
    ):
        """generate_stream() yields pending_text when stream ends mid-thinking block."""
        stream_chunks = create_llama_stream_chunks(
            [
                "<think>",
                "Incomplete reasoning without closing tag",
            ]
        )
        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Question"):
            chunks.append(chunk)

        text_content = "".join(c.text for c in chunks if c.text)
        assert "Incomplete reasoning without closing tag" in text_content

    @pytest.mark.asyncio
    async def test_stream_flushes_pending_text_at_end(self, runner):
        stream_chunks = create_llama_stream_chunks([
            "Hello"
        ])

        runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in runner.generate_stream(prompt="Q"):
            chunks.append(chunk)

        text = "".join(c.text for c in chunks if c.text)
        assert text == "Hello"


class TestLlamaCppRunnerGenerateStreamThinkingModel:
    """Tests for generate_stream() with thinking models (template prepends <think>)."""

    @pytest.fixture
    def thinking_runner(self):
        """Create a LlamaCppRunner for a thinking model."""
        chat_template = """
{%- if add_generation_prompt %}<|assistant|><think>{%- endif %}
"""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template=chat_template,
        )
        return LlamaCppRunner(handle)

    @pytest.mark.asyncio
    async def test_thinking_model_starts_in_thinking_mode(self, thinking_runner):
        """Thinking model starts with in_thinking=True, filters initial content."""
        stream_chunks = create_llama_stream_chunks(
            [
                "This is thinking content",
                "</think>",
                "This is the answer.",
            ]
        )
        thinking_runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in thinking_runner.generate_stream(prompt="Question"):
            chunks.append(chunk)

        text_content = "".join(c.text for c in chunks if c.text)
        assert "This is thinking content" not in text_content
        assert "This is the answer." in text_content

        final_chunk = chunks[-1]
        assert final_chunk.search_metadata is not None
        assert "thinking" in final_chunk.search_metadata
        assert "This is thinking content" in final_chunk.search_metadata["thinking"]

    @pytest.mark.asyncio
    async def test_thinking_model_no_close_tag_yields_all_as_text(
        self, thinking_runner
    ):
        """Thinking model without </think> yields all content as regular text."""
        stream_chunks = create_llama_stream_chunks(
            [
                "This is actually a normal response",
                " from a model that chose not to think.",
            ]
        )
        thinking_runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in thinking_runner.generate_stream(prompt="Hello"):
            chunks.append(chunk)

        text_content = "".join(c.text for c in chunks if c.text)
        assert "This is actually a normal response" in text_content
        assert "from a model that chose not to think." in text_content

        final_chunk = chunks[-1]
        if final_chunk.search_metadata:
            assert "thinking" not in final_chunk.search_metadata

    @pytest.mark.asyncio
    async def test_thinking_model_with_proper_thinking_block(self, thinking_runner):
        """Thinking model with proper </think> extracts thinking correctly."""
        stream_chunks = create_llama_stream_chunks(
            [
                "Let me analyze step by step.",
                " First, consider X.",
                " Then, Y.",
                "</think>",
                "The answer is Z.",
            ]
        )
        thinking_runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in thinking_runner.generate_stream(prompt="Question"):
            chunks.append(chunk)

        text_content = "".join(c.text for c in chunks if c.text)
        assert "Let me analyze" not in text_content
        assert "First, consider X" not in text_content
        assert "The answer is Z." in text_content

        final_chunk = chunks[-1]
        assert final_chunk.search_metadata is not None
        assert "thinking" in final_chunk.search_metadata
        assert "Let me analyze step by step" in final_chunk.search_metadata["thinking"]
        assert "First, consider X" in final_chunk.search_metadata["thinking"]

    @pytest.mark.asyncio
    async def test_thinking_model_multiple_think_blocks(self, thinking_runner):
        """Thinking model handles multiple thinking blocks correctly."""
        stream_chunks = create_llama_stream_chunks(
            [
                "First thought.",
                "</think>",
                "Middle answer.",
                "<think>",
                "Second thought.",
                "</think>",
                "Final answer.",
            ]
        )
        thinking_runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in thinking_runner.generate_stream(prompt="Question"):
            chunks.append(chunk)

        text_content = "".join(c.text for c in chunks if c.text)
        assert "First thought." not in text_content
        assert "Second thought." not in text_content
        assert "Middle answer." in text_content
        assert "Final answer." in text_content

    @pytest.mark.asyncio
    async def test_stream_enters_partial_close_tag_buffer_branch(self, thinking_runner):
        # IMPORTANT:
        # We must explicitly enter thinking via <think> in the stream,
        # and end a chunk with EXACTLY 7 chars starting with "<" => "</think"

        stream_chunks = create_llama_stream_chunks([
            "<think>ABC</think",   # pending_text[-7:] == "</think"  ✅
            ">",                   # completes </think>
            "DONE"
        ])

        thinking_runner.model.create_chat_completion.return_value = iter(stream_chunks)

        chunks = []
        async for chunk in thinking_runner.generate_stream(prompt="Q"):
            chunks.append(chunk)

        # Minimal assertion: stream completed
        assert chunks[-1].finished is True


class TestLlamaCppRunnerThinkingModelDuplicateThinkTag:
    """Covers case where a thinking model redundantly emits <think>."""

    @pytest.fixture
    def thinking_runner(self):
        chat_template = """
{%- if add_generation_prompt %}<|assistant|><think>{%- endif %}
"""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            chat_template=chat_template,
        )
        return LlamaCppRunner(handle)

    @pytest.mark.asyncio
    async def test_thinking_model_strips_redundant_leading_think(self, thinking_runner):
        """
        Covers the branch:

            if thinking_part.startswith('<think>'):

        by forcing:
        - thinking model enabled via template
        - response that begins with <think> and includes </think>
        """
        thinking_runner.model.create_chat_completion.return_value = (
            create_llama_completion_response(
                content="<think>internal reasoning</think>final answer"
            )
        )

        result = await thinking_runner.generate(prompt="q")

        assert result["thinking"] == "internal reasoning"
        assert result["reply"] == "final answer"
