"""
Tests for LlamaCppRunner initialization.
"""

import pytest
from unittest.mock import MagicMock, patch

from practorflow.llm.llama_cpp_runner import LlamaCppRunner
from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.llm.common_runner import create_mock_model_handle


class TestLlamaCppRunnerInit:
    """Tests for LlamaCppRunner.__init__"""

    def test_init_with_valid_llama_cpp_handle(self):
        """LlamaCppRunner initializes successfully with llama_cpp backend handle."""
        handle = create_mock_model_handle(backend="llama_cpp")
        
        runner = LlamaCppRunner(handle)
        
        assert runner.handle is handle
        assert runner.model is handle.model
        assert runner.model_name == handle.config.model_name

    def test_init_with_transformers_handle_raises_error(self):
        """LlamaCppRunner raises ValueError when given transformers backend handle."""
        handle = create_mock_model_handle(backend="transformers")
        
        with pytest.raises(ValueError, match="Expected llama_cpp handle"):
            LlamaCppRunner(handle)

    def test_init_with_knowledge_store_registers_tool(self, mock_knowledge_store):
        """LlamaCppRunner registers knowledge_search tool when knowledge_store provided."""
        handle = create_mock_model_handle(backend="llama_cpp")
        
        runner = LlamaCppRunner(handle, knowledge_store=mock_knowledge_store)
        
        assert runner.knowledge_store is mock_knowledge_store
        assert "knowledge_search" in runner.tool_registry.list_tools()

    def test_init_without_knowledge_store(self):
        """LlamaCppRunner does not register knowledge_search tool when knowledge_store is None."""
        handle = create_mock_model_handle(backend="llama_cpp")
        
        runner = LlamaCppRunner(handle, knowledge_store=None)
        
        assert runner.knowledge_store is None
        assert "knowledge_search" not in runner.tool_registry.list_tools()

    def test_init_sets_model_attributes(self):
        """LlamaCppRunner correctly sets model attributes from handle."""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            model_name="test-gguf-model",
            max_context_length=8192,
        )
        handle.config.device = "cuda"
        handle.config.dtype = "float16"
        handle.config.max_new_tokens = 1024
        
        runner = LlamaCppRunner(handle)
        
        assert runner.model_name == "test-gguf-model"
        assert runner.max_context_length == 8192
        assert runner.device == "cuda"
        assert runner.dtype == "float16"
        assert runner.max_new_tokens == 1024

    def test_init_sets_config_from_handle(self):
        """LlamaCppRunner sets config reference from handle."""
        handle = create_mock_model_handle(backend="llama_cpp")
        
        runner = LlamaCppRunner(handle)
        
        assert runner.config is handle.config

    def test_init_tokenizer_is_none(self):
        """LlamaCppRunner has None tokenizer (llama.cpp doesn't use separate tokenizer)."""
        handle = create_mock_model_handle(backend="llama_cpp")
        handle.tokenizer = None
        
        runner = LlamaCppRunner(handle)
        
        assert runner.tokenizer is None