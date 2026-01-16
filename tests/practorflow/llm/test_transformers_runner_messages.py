"""
Tests for TransformersRunner message building methods.
"""

import pytest

from practorflow.llm.transformers_runner import TransformersRunner
from tests.practorflow.common.fixtures import sample_chat_messages
from tests.practorflow.llm.common_runner import create_mock_model_handle


class TestTransformersRunnerBuildChatMessages:
    """Tests for TransformersRunner._build_chat_messages()"""

    @pytest.fixture
    def runner(self):
        """Create a TransformersRunner instance for testing."""
        handle = create_mock_model_handle(backend="transformers")
        return TransformersRunner(handle)

    def test_build_chat_messages_with_prompt_only(self, runner):
        """Builds messages with just a prompt."""
        result = runner._build_chat_messages(prompt="What is Python?")

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "What is Python?" in result[0]["content"]

    def test_build_chat_messages_with_messages_only(self, runner, sample_chat_messages):
        """Builds messages from provided message list."""
        result = runner._build_chat_messages(messages=sample_chat_messages)

        # plus the system message for not repeating answer
        assert len(result) == len(sample_chat_messages) + 1
        # next we remove it
        result = [m for m in result if not m["role"] == "system"]
        for i, msg in enumerate(sample_chat_messages):
            assert result[i]["role"] == msg["role"]
            assert result[i]["content"] == msg["content"]

    def test_build_chat_messages_with_instructions(self, runner):
        """Includes instructions as system message."""
        result = runner._build_chat_messages(
            prompt="Hello",
            instructions="You are a helpful assistant.",
        )

        system_messages = [m for m in result if m["role"] == "system"]
        assert len(system_messages) == 1
        assert "helpful assistant" in system_messages[0]["content"]

    def test_build_chat_messages_with_context(self, runner):
        """Includes context in system message with instructions."""
        result = runner._build_chat_messages(
            prompt="What does the document say?",
            context="The document contains important information about AI.",
        )

        system_messages = [m for m in result if m["role"] == "system"]
        assert len(system_messages) == 1
        assert "important information about AI" in system_messages[0]["content"]
        assert "REFERENCE DOCUMENTS" in system_messages[0]["content"]

    def test_build_chat_messages_with_instructions_and_context(self, runner):
        """Combines instructions and context in system message."""
        result = runner._build_chat_messages(
            prompt="Summarize the doc",
            instructions="Be concise.",
            context="Document content here.",
        )

        system_messages = [m for m in result if m["role"] == "system"]
        assert len(system_messages) == 1
        content = system_messages[0]["content"]
        assert "Be concise" in content
        assert "Document content here" in content

    def test_build_chat_messages_with_all_params(self, runner):
        """Builds messages with all parameters provided."""
        messages = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        result = runner._build_chat_messages(
            messages=messages,
            instructions="Be helpful.",
            context="Relevant context.",
        )

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert "Be helpful" in result[0]["content"]
        assert "Relevant context" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_build_chat_messages_prompt_with_context_adds_reminder(self, runner):
        """Adds reminder to use reference documents when context provided with prompt."""
        result = runner._build_chat_messages(
            prompt="Answer my question",
            context="Context info",
        )

        user_messages = [m for m in result if m["role"] == "user"]
        assert len(user_messages) == 1
        assert "reference documents" in user_messages[0]["content"].lower()

    def test_build_chat_messages_raises_with_both_messages_and_prompt(self, runner):
        """Raises ValueError when both messages and prompt provided."""
        with pytest.raises(ValueError, match="Cannot provide both messages and prompt"):
            runner._build_chat_messages(
                messages=[{"role": "user", "content": "Hello"}],
                prompt="Hello",
            )

    def test_build_chat_messages_raises_with_neither_messages_nor_prompt(self, runner):
        """Raises ValueError when neither messages nor prompt provided."""
        with pytest.raises(ValueError, match="Must provide either messages or prompt"):
            runner._build_chat_messages()

    def test_build_chat_messages_preserves_message_order(self, runner):
        """Preserves order of messages in conversation."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
            {"role": "assistant", "content": "Fourth"},
        ]
        result = runner._build_chat_messages(messages=messages)
        # system removed 
        result = [m for m in result if not m["role"] == "system"]
        assert result[0]["content"] == "First"
        assert result[1]["content"] == "Second"
        assert result[2]["content"] == "Third"
        assert result[3]["content"] == "Fourth"

    def test_build_chat_messages_system_message_first(self, runner):
        """System message appears before user messages."""
        result = runner._build_chat_messages(
            prompt="Question",
            instructions="Instructions here",
        )

        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_build_chat_messages_context_formatting(self, runner):
        """Context is properly formatted with document markers."""
        result = runner._build_chat_messages(
            prompt="Query",
            context="Document text content",
        )

        system_content = result[0]["content"]
        assert "REFERENCE DOCUMENTS:" in system_content
        assert "Document text content" in system_content
        assert "END OF DOCUMENTS" in system_content

    def test_build_chat_messages_empty_instructions(self, runner):
        """Handles empty string instructions."""
        result = runner._build_chat_messages(
            prompt="Hello",
            instructions="",
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_build_chat_messages_none_context(self, runner):
        """Handles None context correctly."""
        result = runner._build_chat_messages(
            prompt="Hello",
            context=None,
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "REFERENCE DOCUMENTS" not in result[0]["content"]


class TestTransformersRunnerFormatMessagesFallback:
    """Tests for TransformersRunner._format_messages_fallback()"""

    @pytest.fixture
    def runner(self):
        """Create a TransformersRunner instance for testing."""
        handle = create_mock_model_handle(backend="transformers")
        return TransformersRunner(handle)

    def test_format_messages_fallback_system_message(self, runner):
        """Formats system message with 'System:' prefix."""
        messages = [{"role": "system", "content": "You are helpful."}]
        
        result = runner._format_messages_fallback(messages)
        
        assert "System: You are helpful." in result

    def test_format_messages_fallback_user_message(self, runner):
        """Formats user message with 'User:' prefix."""
        messages = [{"role": "user", "content": "Hello there"}]
        
        result = runner._format_messages_fallback(messages)
        
        assert "User: Hello there" in result

    def test_format_messages_fallback_assistant_message(self, runner):
        """Formats assistant message with 'Assistant:' prefix."""
        messages = [{"role": "assistant", "content": "Hi!"}]
        
        result = runner._format_messages_fallback(messages)
        
        assert "Assistant: Hi!" in result

    def test_format_messages_fallback_ends_with_assistant_prompt(self, runner):
        """Ends with 'Assistant:' prompt for generation."""
        messages = [{"role": "user", "content": "Question"}]
        
        result = runner._format_messages_fallback(messages)
        
        assert result.endswith("Assistant:")

    def test_format_messages_fallback_multiple_messages(self, runner):
        """Formats multiple messages in sequence."""
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        
        result = runner._format_messages_fallback(messages)
        
        assert "System: Be helpful" in result
        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result
        assert "User: How are you?" in result

    def test_format_messages_fallback_separates_with_newlines(self, runner):
        """Messages are separated by double newlines."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
        ]
        
        result = runner._format_messages_fallback(messages)
        
        assert "\n\n" in result