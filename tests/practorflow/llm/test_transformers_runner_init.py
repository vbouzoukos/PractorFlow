"""
Tests for TransformersRunner initialization.
"""

import pytest
from unittest.mock import MagicMock

import torch

from practorflow.llm.transformers_runner import TransformersRunner
from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.llm.common_runner import create_mock_model_handle


class TestTransformersRunnerInit:
    """Tests for TransformersRunner.__init__"""

    def test_init_with_valid_transformers_handle(self):
        """TransformersRunner initializes successfully with transformers backend handle."""
        handle = create_mock_model_handle(backend="transformers")
        
        runner = TransformersRunner(handle)
        
        assert runner.handle is handle
        assert runner.model is handle.model
        assert runner.model_name == handle.config.model_name

    def test_init_with_llama_cpp_handle_raises_error(self):
        """TransformersRunner raises ValueError when given llama_cpp backend handle."""
        handle = create_mock_model_handle(backend="llama_cpp")
        
        with pytest.raises(ValueError, match="Expected transformers handle"):
            TransformersRunner(handle)

    def test_init_with_knowledge_store_registers_tool(self, mock_knowledge_store):
        """TransformersRunner registers knowledge_search tool when knowledge_store provided."""
        handle = create_mock_model_handle(backend="transformers")
        
        runner = TransformersRunner(handle, knowledge_store=mock_knowledge_store)
        
        assert runner.knowledge_store is mock_knowledge_store
        assert "knowledge_search" in runner.tool_registry.list_tools()

    def test_init_without_knowledge_store(self):
        """TransformersRunner does not register knowledge_search tool when knowledge_store is None."""
        handle = create_mock_model_handle(backend="transformers")
        
        runner = TransformersRunner(handle, knowledge_store=None)
        
        assert runner.knowledge_store is None
        assert "knowledge_search" not in runner.tool_registry.list_tools()

    def test_init_sets_model_attributes(self):
        """TransformersRunner correctly sets model attributes from handle."""
        handle = create_mock_model_handle(
            backend="transformers",
            model_name="test-hf-model",
            max_context_length=8192,
        )
        handle.config.device = "cuda"
        handle.config.dtype = "float16"
        handle.config.max_new_tokens = 1024
        
        runner = TransformersRunner(handle)
        
        assert runner.model_name == "test-hf-model"
        assert runner.max_context_length == 8192
        assert runner.device == "cuda"
        assert runner.dtype == "float16"
        assert runner.max_new_tokens == 1024

    def test_init_sets_config_from_handle(self):
        """TransformersRunner sets config reference from handle."""
        handle = create_mock_model_handle(backend="transformers")
        
        runner = TransformersRunner(handle)
        
        assert runner.config is handle.config

    def test_init_sets_tokenizer_from_handle(self):
        """TransformersRunner sets tokenizer from handle."""
        handle = create_mock_model_handle(backend="transformers")
        
        runner = TransformersRunner(handle)
        
        assert runner.tokenizer is handle.tokenizer

    def test_init_sets_pad_token_id(self):
        """TransformersRunner sets _pad_token_id from tokenizer."""
        handle = create_mock_model_handle(backend="transformers")
        handle.tokenizer.pad_token_id = 42
        
        runner = TransformersRunner(handle)
        
        assert runner._pad_token_id == 42

    def test_init_sets_eos_token_id(self):
        """TransformersRunner sets _eos_token_id from tokenizer."""
        handle = create_mock_model_handle(backend="transformers")
        handle.tokenizer.eos_token_id = 99
        
        runner = TransformersRunner(handle)
        
        assert runner._eos_token_id == 99

    def test_init_gets_device_from_model_device_attribute(self):
        """TransformersRunner gets device from model.device when available."""
        handle = create_mock_model_handle(backend="transformers")
        handle.model.device = torch.device("cuda:0")
        
        runner = TransformersRunner(handle)
        
        assert runner._device == torch.device("cuda:0")

    def test_init_gets_device_from_parameters_when_no_device_attr(self):
        """TransformersRunner gets device from model parameters when model.device not available."""
        handle = create_mock_model_handle(backend="transformers")
        del handle.model.device
        
        mock_param = MagicMock()
        mock_param.device = torch.device("cuda:1")
        handle.model.parameters = MagicMock(return_value=iter([mock_param]))
        
        runner = TransformersRunner(handle)
        
        assert runner._device == torch.device("cuda:1")

    def test_init_initializes_tool_registry(self):
        """TransformersRunner initializes tool_registry."""
        handle = create_mock_model_handle(backend="transformers")
        
        runner = TransformersRunner(handle)
        
        assert runner.tool_registry is not None

    def test_init_initializes_pending_context_to_none(self):
        """TransformersRunner initializes _pending_context to None."""
        handle = create_mock_model_handle(backend="transformers")
        
        runner = TransformersRunner(handle)
        
        assert runner._pending_context is None