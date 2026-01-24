"""
Common fixtures and helpers for ChatService tests.

This module provides shared test infrastructure used across all
test_chat_service_* test files to reduce duplication.
"""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from practorflow.llm import ModelPool
from practorflow.llm.base.session import Message, Session
from practorflow.llm.base.session_store import SessionStore
from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.tools.base_web_search import DuckDuckGoSearchTool
from practorflow.services.chat.chat_service import ChatService
from practorflow.services.dto.chat_file import ChatFile
from tests.practorflow.common.stream_mock import MockStreamText


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_model_pool():
    """Mock ModelPool for testing."""
    return MagicMock(spec=ModelPool)


@pytest.fixture
def mock_model_config():
    """Mock LLMConfig for testing."""
    config = MagicMock(spec=LLMConfig)
    config.model_name = "test-model"
    config.temperature = 0.7
    config.top_p = 0.9
    return config


@pytest.fixture
def mock_session_store():
    """
    Mock SessionStore for testing.
    
    Includes common method mocks used across tests.
    """
    store = MagicMock(spec=SessionStore)
    store.exists = MagicMock(return_value=False)
    store.get = MagicMock()
    store.save = MagicMock()
    store.delete = MagicMock()
    return store


@pytest.fixture
def mock_web_search_tool():
    """Mock DuckDuckGoSearchTool for testing."""
    mock = MagicMock(spec=DuckDuckGoSearchTool)
    mock.search = MagicMock(return_value="Mock search results")
    return mock


# ============================================================================
# ChatService Fixture
# ============================================================================


@pytest.fixture
def chat_service(
    mock_model_pool,
    mock_model_config,
    mock_knowledge_store,
    mock_session_store,
    mock_web_search_tool,
):
    """ChatService instance for testing."""
    return ChatService(
        model_pool=mock_model_pool,
        model_config=mock_model_config,
        knowledge_store=mock_knowledge_store,
        session_store=mock_session_store,
        web_search_tool=mock_web_search_tool,
    )


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_session():
    """Sample session for testing."""
    return Session(
        session_id="session_test123",
        instructions="Test instructions",
        user="test_user",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ],
        documents=[
            {"id": "doc-1", "filename": "file1.txt"},
            {"id": "doc-2", "filename": "file2.txt"},
        ],
    )


@pytest.fixture
def sample_chat_file():
    """Sample ChatFile for testing."""
    file_content = b"Test file content"
    file_stream = BytesIO(file_content)

    chat_file = MagicMock(spec=ChatFile)
    chat_file.file = file_stream
    chat_file.filename = "test_document.txt"
    chat_file.content_type = "text/plain"
    return chat_file


# ============================================================================
# Helper Functions for Streaming Tests (Legacy - for run_stream)
# ============================================================================


def create_mock_streaming_context(text_chunks, input_tokens=10, output_tokens=5):
    """
    Create a mock streaming context for testing agent.run_stream().
    
    DEPRECATED: Use create_mock_agent_iter_context for agent.iter() tests.
    
    Args:
        text_chunks: List of text strings to yield during streaming.
        input_tokens: Number of input tokens for usage stats.
        output_tokens: Number of output tokens for usage stats.
    
    Returns:
        AsyncMock configured as a streaming context manager.
    """
    mock_response = MagicMock()
    mock_response.stream_text.return_value = MockStreamText(text_chunks)
    mock_usage = MagicMock()
    mock_usage.input_tokens = input_tokens
    mock_usage.output_tokens = output_tokens
    mock_response.usage.return_value = mock_usage

    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    mock_stream_context.__aexit__.return_value = None

    return mock_stream_context


def create_mock_model_pool_context(mock_model_pool):
    """
    Configure mock model pool with async context manager.
    
    Args:
        mock_model_pool: The mock ModelPool instance to configure.
    
    Returns:
        The mock handle returned by the context manager.
    """
    mock_handle = MagicMock()
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_handle
    mock_context.__aexit__.return_value = None
    mock_model_pool.acquire_context.return_value = mock_context
    return mock_handle


# ============================================================================
# Helper Functions for Agentic Loop Tests (agent.iter())
# ============================================================================


class MockAsyncIterator:
    """Mock async iterator for agent.iter() node iteration."""
    
    def __init__(self, nodes):
        """
        Initialize with list of nodes to yield.
        
        Args:
            nodes: List of mock nodes to yield during iteration.
        """
        self.nodes = nodes
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index < len(self.nodes):
            node = self.nodes[self.index]
            self.index += 1
            return node
        raise StopAsyncIteration


def create_mock_agent_iter_context(text_chunks, input_tokens=10, output_tokens=5):
    """
    Create a mock agent.iter() context for testing agentic loop.
    
    This creates a mock that simulates the agent.iter() behavior by yielding
    an End node that contains the final output.
    
    Args:
        text_chunks: List of text strings that form the response.
        input_tokens: Number of input tokens for usage stats.
        output_tokens: Number of output tokens for usage stats.
    
    Returns:
        Tuple of (mock_iter_context, mock_agent_run, nodes_list)
    """
    full_text = "".join(text_chunks)
    
    # Create mock End node
    mock_end_node = MagicMock()
    mock_end_node.data = MagicMock()
    mock_end_node.data.output = full_text
    
    nodes = [mock_end_node]
    
    # Create mock usage
    mock_usage = MagicMock()
    mock_usage.input_tokens = input_tokens
    mock_usage.output_tokens = output_tokens
    
    # Create mock result
    mock_result = MagicMock()
    mock_result.output = full_text
    
    # Create mock agent run
    mock_agent_run = MagicMock()
    mock_agent_run.ctx = MagicMock()
    mock_agent_run.result = mock_result
    mock_agent_run.usage.return_value = mock_usage
    
    # Setup async iteration
    mock_agent_run.__aiter__ = lambda self: MockAsyncIterator(nodes)
    
    # Create context manager
    mock_iter_context = AsyncMock()
    mock_iter_context.__aenter__.return_value = mock_agent_run
    mock_iter_context.__aexit__.return_value = None
    
    return mock_iter_context, mock_agent_run, nodes


def setup_mock_agent_for_iter(mock_agent_class, text_chunks, input_tokens=10, output_tokens=5):
    """
    Setup a mock Agent class for agent.iter() testing.
    
    This configures the mock Agent class with all necessary node type checks
    and returns the configured mock agent.
    
    Args:
        mock_agent_class: The patched Agent class mock.
        text_chunks: List of text strings that form the response.
        input_tokens: Number of input tokens for usage stats.
        output_tokens: Number of output tokens for usage stats.
    
    Returns:
        Configured mock agent instance.
    """
    mock_agent = MagicMock()
    mock_agent_class.return_value = mock_agent
    
    # Create iter context
    mock_iter_context, mock_agent_run, nodes = create_mock_agent_iter_context(
        text_chunks, input_tokens, output_tokens
    )
    mock_agent.iter.return_value = mock_iter_context
    
    # Configure node type checks - End node returns True, others False
    mock_agent_class.is_end_node.return_value = True
    mock_agent_class.is_user_prompt_node.return_value = False
    mock_agent_class.is_model_request_node.return_value = False
    mock_agent_class.is_call_tools_node.return_value = False
    
    return mock_agent