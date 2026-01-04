"""
Tests for TransformersRunner tool/function calling support.
"""

import pytest
from unittest.mock import MagicMock

from practorflow.llm.transformers_runner import TransformersRunner
from tests.practorflow.llm.common_runner import create_mock_model_handle


class TestTransformersRunnerSupportsFunctionCalling:
    """Tests for TransformersRunner.supports_function_calling()"""

    def test_supports_function_calling_true_with_tool_in_template(self):
        """supports_function_calling() returns True when chat template contains 'tool'."""
        handle = create_mock_model_handle(
            backend="transformers",
            chat_template="{% if tools %}Use tools: {{ tools }}{% endif %}",
        )
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_true_with_function_in_template(self):
        """supports_function_calling() returns True when chat template contains 'function'."""
        handle = create_mock_model_handle(
            backend="transformers",
            chat_template="{% if functions %}Call function: {{ functions }}{% endif %}",
        )
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_true_with_tool_call_tag(self):
        """supports_function_calling() returns True when template has <tool_call> tag."""
        handle = create_mock_model_handle(
            backend="transformers",
            chat_template="<tool_call>{{ tool }}</tool_call>",
        )
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_true_with_function_call_tag(self):
        """supports_function_calling() returns True when template has <function_call> tag."""
        handle = create_mock_model_handle(
            backend="transformers",
            chat_template="<function_call>{{ func }}</function_call>",
        )
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_true_with_tools_keyword(self):
        """supports_function_calling() returns True when template contains 'tools'."""
        handle = create_mock_model_handle(
            backend="transformers",
            chat_template="Available tools: {{ tools }}",
        )
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_true_with_functions_keyword(self):
        """supports_function_calling() returns True when template contains 'functions'."""
        handle = create_mock_model_handle(
            backend="transformers",
            chat_template="Available functions: {{ functions }}",
        )
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_true_with_tool_use_keyword(self):
        """supports_function_calling() returns True when template contains 'tool_use'."""
        handle = create_mock_model_handle(
            backend="transformers",
            chat_template="{% if tool_use %}{{ tool_use }}{% endif %}",
        )
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_true_with_function_use_keyword(self):
        """supports_function_calling() returns True when template contains 'function_use'."""
        handle = create_mock_model_handle(
            backend="transformers",
            chat_template="{% if function_use %}{{ function_use }}{% endif %}",
        )
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_false_without_keywords(self):
        """supports_function_calling() returns False when no tool keywords in template."""
        handle = create_mock_model_handle(
            backend="transformers",
            chat_template="Standard chat template without special keywords",
        )
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_false_with_no_template(self):
        """supports_function_calling() returns False when tokenizer has no chat_template."""
        handle = create_mock_model_handle(backend="transformers")
        handle.tokenizer.chat_template = None
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_false_with_empty_template(self):
        """supports_function_calling() returns False when chat_template is empty."""
        handle = create_mock_model_handle(backend="transformers")
        handle.tokenizer.chat_template = ""
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_true_with_config_flag(self):
        """supports_function_calling() returns True when config has supports_function_calling=True."""
        handle = create_mock_model_handle(backend="transformers")
        handle.tokenizer.chat_template = None
        handle.model.config.to_dict.return_value = {"supports_function_calling": True}
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_false_with_config_flag_false(self):
        """supports_function_calling() returns False when config has supports_function_calling=False."""
        handle = create_mock_model_handle(backend="transformers")
        handle.tokenizer.chat_template = None
        handle.model.config.to_dict.return_value = {"supports_function_calling": False}
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_checks_template_case_insensitive(self):
        """supports_function_calling() checks template keywords case-insensitively."""
        handle = create_mock_model_handle(
            backend="transformers",
            chat_template="TOOL_CALL and FUNCTION keywords",
        )
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is True

    def test_supports_function_calling_handles_missing_tokenizer(self):
        """supports_function_calling() returns False when tokenizer is None."""
        handle = create_mock_model_handle(backend="transformers")
        handle.tokenizer = None
        runner = TransformersRunner.__new__(TransformersRunner)
        runner.tokenizer = None
        runner.model = handle.model
        handle.model.config.to_dict.return_value = {}

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_handles_missing_chat_template_attr(self):
        """supports_function_calling() returns False when chat_template attribute missing."""
        handle = create_mock_model_handle(backend="transformers")
        del handle.tokenizer.chat_template
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_handles_missing_config(self):
        """supports_function_calling() handles missing model config gracefully."""
        handle = create_mock_model_handle(backend="transformers")
        handle.tokenizer.chat_template = None
        del handle.model.config
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_handles_missing_to_dict(self):
        """supports_function_calling() handles missing config.to_dict gracefully."""
        handle = create_mock_model_handle(backend="transformers")
        handle.tokenizer.chat_template = None
        del handle.model.config.to_dict
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_handles_exception(self):
        """supports_function_calling() returns False on exception."""
        handle = create_mock_model_handle(backend="transformers")
        runner = TransformersRunner(handle)
        
        # Make tokenizer raise exception when accessed in the check
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = MagicMock()
        mock_tokenizer.chat_template.__str__ = MagicMock(side_effect=RuntimeError("Test error"))
        runner.tokenizer = mock_tokenizer

        assert runner.supports_function_calling() is False

    def test_supports_function_calling_template_takes_precedence_over_config(self):
        """supports_function_calling() checks template before config flag."""
        handle = create_mock_model_handle(
            backend="transformers",
            chat_template="{% if tools %}{{ tools }}{% endif %}",
        )
        handle.model.config.to_dict.return_value = {"supports_function_calling": False}
        runner = TransformersRunner(handle)

        assert runner.supports_function_calling() is True