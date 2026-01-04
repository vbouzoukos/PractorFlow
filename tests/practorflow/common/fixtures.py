"""
Shared fixtures for PractorFlow tests.

This module provides common fixtures used across multiple test suites
including chat_service, llm runners, and other components.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.llm_config import LLMConfig


@pytest.fixture
def mock_knowledge_store():
    """
    Mock KnowledgeStore for testing.
    
    Includes common method mocks used across tests.
    """
    store = MagicMock(spec=KnowledgeStore)
    store.add_document_from_stream = MagicMock(
        return_value={
            "id": "doc-123",
            "filename": "test.txt",
            "content": "test content",
        }
    )
    store.delete_document = MagicMock()
    store.search_scoped = MagicMock(return_value=[])
    return store


@pytest.fixture
def mock_llm_config():
    """
    Mock LLMConfig for testing.
    
    Provides default values for common configuration options.
    """
    config = MagicMock(spec=LLMConfig)
    config.model_name = "test-model"
    config.backend = "llama_cpp"
    config.device = "cpu"
    config.dtype = "float16"
    config.temperature = 0.7
    config.top_p = 0.9
    config.max_new_tokens = 512
    config.max_search_results = 5
    config.stop_tokens = None
    config.n_ctx = 4096
    config.n_gpu_layers = 0
    config.n_batch = 512
    config.quantization = None
    return config


@pytest.fixture
def sample_tool_definitions() -> List[Dict[str, Any]]:
    """
    Sample tool definitions for testing function calling.
    
    Returns tool definitions in OpenAI function calling format.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search for relevant documents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


@pytest.fixture
def sample_chat_messages() -> List[Dict[str, str]]:
    """
    Sample chat messages for testing generation.
    
    Returns a typical conversation history.
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
        {"role": "user", "content": "What is the capital of France?"},
    ]