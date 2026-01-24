"""
Unit tests for LLM runner factory.

Tests cover create_runner factory function for 100% code coverage.
"""

import pytest
from unittest.mock import patch, MagicMock

from practorflow.llm.factory import create_runner
from practorflow.llm.llama_cpp_runner import LlamaCppRunner
from practorflow.llm.transformers_runner import TransformersRunner
from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.llm.common_runner import create_mock_model_handle


class TestCreateRunner:
    """Tests for create_runner factory function."""

    def test_create_runner_llama_cpp_backend(self):
        """create_runner returns LlamaCppRunner for llama_cpp backend."""
        handle = create_mock_model_handle(backend="llama_cpp")

        runner = create_runner(handle)

        assert isinstance(runner, LlamaCppRunner)
        assert runner.handle is handle

    def test_create_runner_transformers_backend(self):
        """create_runner returns TransformersRunner for transformers backend."""
        handle = create_mock_model_handle(backend="transformers")
        handle.model.device = "cpu"

        runner = create_runner(handle)

        assert isinstance(runner, TransformersRunner)
        assert runner.handle is handle

    def test_create_runner_with_knowledge_store(self, mock_knowledge_store):
        """create_runner passes knowledge_store to runner."""
        handle = create_mock_model_handle(backend="llama_cpp")

        runner = create_runner(handle, knowledge_store=mock_knowledge_store)

        assert runner.knowledge_store is mock_knowledge_store

    def test_create_runner_without_knowledge_store(self):
        """create_runner works without knowledge_store."""
        handle = create_mock_model_handle(backend="llama_cpp")

        runner = create_runner(handle, knowledge_store=None)

        assert runner.knowledge_store is None

    def test_create_runner_unsupported_backend_raises_error(self):
        """create_runner raises ValueError for unsupported backend."""
        handle = create_mock_model_handle(backend="unsupported_backend")

        with pytest.raises(ValueError, match="Unsupported backend: unsupported_backend"):
            create_runner(handle)

    def test_create_runner_error_message_lists_supported_backends(self):
        """create_runner error message lists supported backends."""
        handle = create_mock_model_handle(backend="invalid")

        with pytest.raises(ValueError) as exc_info:
            create_runner(handle)

        assert "llama_cpp" in str(exc_info.value)
        assert "transformers" in str(exc_info.value)