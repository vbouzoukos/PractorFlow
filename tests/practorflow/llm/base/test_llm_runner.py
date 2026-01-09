"""
Unit tests for LLMRunner base class.

Tests cover LLMRunner methods that are not abstract.
Uses LlamaCppRunner as concrete implementation for testing.
"""


from practorflow.llm.llama_cpp_runner import LlamaCppRunner
from practorflow.llm.base.llm_runner import StreamChunk

from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.llm.common_runner import create_mock_model_handle


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_stream_chunk_default_values(self):
        """StreamChunk initializes with default values."""
        chunk = StreamChunk(text="Hello")

        assert chunk.text == "Hello"
        assert chunk.finished is False
        assert chunk.finish_reason is None
        assert chunk.latency_seconds is None
        assert chunk.usage is None
        assert chunk.context_used is None
        assert chunk.search_metadata is None
        assert chunk.tool_calls is None

    def test_stream_chunk_all_fields(self):
        """StreamChunk accepts all fields."""
        chunk = StreamChunk(
            text="Response",
            finished=True,
            finish_reason="stop",
            latency_seconds=1.5,
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            context_used="Some context",
            search_metadata={"query": "test"},
            tool_calls=[{"name": "search"}],
        )

        assert chunk.text == "Response"
        assert chunk.finished is True
        assert chunk.finish_reason == "stop"
        assert chunk.latency_seconds == 1.5
        assert chunk.usage == {"prompt_tokens": 10, "completion_tokens": 20}
        assert chunk.context_used == "Some context"
        assert chunk.search_metadata == {"query": "test"}
        assert chunk.tool_calls == [{"name": "search"}]


class TestLLMRunnerDocumentScope:
    """Tests for LLMRunner document scope methods."""

    def test_set_document_scope(self):
        """set_document_scope delegates to tool_registry."""
        handle = create_mock_model_handle(backend="llama_cpp")
        runner = LlamaCppRunner(handle)

        runner.set_document_scope({"doc-1", "doc-2"})

        assert runner.tool_registry.get_document_scope() == {"doc-1", "doc-2"}

    def test_set_document_scope_with_none(self):
        """set_document_scope with None clears scope."""
        handle = create_mock_model_handle(backend="llama_cpp")
        runner = LlamaCppRunner(handle)
        runner.set_document_scope({"doc-1"})

        runner.set_document_scope(None)

        assert runner.tool_registry.get_document_scope() is None

    def test_clear_document_scope(self):
        """clear_document_scope clears the scope."""
        handle = create_mock_model_handle(backend="llama_cpp")
        runner = LlamaCppRunner(handle)
        runner.set_document_scope({"doc-1"})

        runner.clear_document_scope()

        assert runner.tool_registry.get_document_scope() is None

    def test_get_document_scope(self):
        """get_document_scope returns current scope."""
        handle = create_mock_model_handle(backend="llama_cpp")
        runner = LlamaCppRunner(handle)

        assert runner.get_document_scope() is None

        runner.set_document_scope({"doc-a", "doc-b"})

        assert runner.get_document_scope() == {"doc-a", "doc-b"}


class TestLLMRunnerSearch:
    """Tests for LLMRunner.search() method."""

    def test_search_without_knowledge_store(self):
        """search() returns error when knowledge_store not configured."""
        handle = create_mock_model_handle(backend="llama_cpp")
        runner = LlamaCppRunner(handle, knowledge_store=None)

        result = runner.search("test query")

        assert result.success is False
        assert "Knowledge search tool not available" in result.error

    def test_search_with_knowledge_store(self, mock_knowledge_store):
        """search() executes knowledge_search tool."""
        mock_knowledge_store.search_scoped.return_value = [
            {"text": "Result text", "metadata": {"filename": "doc.txt"}, "similarity": 0.9}
        ]
        handle = create_mock_model_handle(backend="llama_cpp")
        runner = LlamaCppRunner(handle, knowledge_store=mock_knowledge_store)

        result = runner.search("test query")

        assert result.success is True
        mock_knowledge_store.search_scoped.assert_called_once()

    def test_search_uses_default_top_k(self, mock_knowledge_store):
        """search() uses config.max_search_results when top_k not provided."""
        mock_knowledge_store.search_scoped.return_value = []
        handle = create_mock_model_handle(backend="llama_cpp")
        handle.config.max_search_results = 7
        runner = LlamaCppRunner(handle, knowledge_store=mock_knowledge_store)

        runner.search("test query")

        mock_knowledge_store.search_scoped.assert_called_once_with(
            query="test query",
            top_k=7,
            document_ids=None,
        )

    def test_search_uses_provided_top_k(self, mock_knowledge_store):
        """search() uses provided top_k over default."""
        mock_knowledge_store.search_scoped.return_value = []
        handle = create_mock_model_handle(backend="llama_cpp")
        handle.config.max_search_results = 5
        runner = LlamaCppRunner(handle, knowledge_store=mock_knowledge_store)

        runner.search("test query", top_k=3)

        mock_knowledge_store.search_scoped.assert_called_once_with(
            query="test query",
            top_k=3,
            document_ids=None,
        )

    def test_search_stores_pending_context(self, mock_knowledge_store):
        """search() stores successful results as pending context."""
        mock_knowledge_store.search_scoped.return_value = [
            {"text": "Found content", "metadata": {"filename": "doc.txt"}, "similarity": 0.9}
        ]
        handle = create_mock_model_handle(backend="llama_cpp")
        runner = LlamaCppRunner(handle, knowledge_store=mock_knowledge_store)

        runner.search("test query")

        assert runner._pending_context is not None
        assert "Found content" in runner._pending_context

    def test_search_no_pending_context_on_empty_results(self, mock_knowledge_store):
        """search() does not store pending context when no results."""
        mock_knowledge_store.search_scoped.return_value = []
        handle = create_mock_model_handle(backend="llama_cpp")
        runner = LlamaCppRunner(handle, knowledge_store=mock_knowledge_store)

        runner.search("test query")

        assert runner._pending_context is None


class TestLLMRunnerPendingContext:
    """Tests for LLMRunner pending context methods."""

    def test_clear_pending_context(self):
        """clear_pending_context clears the pending context."""
        handle = create_mock_model_handle(backend="llama_cpp")
        runner = LlamaCppRunner(handle)
        runner._pending_context = "Some context"

        runner.clear_pending_context()

        assert runner._pending_context is None

    def test_has_pending_context_true(self):
        """has_pending_context returns True when context exists."""
        handle = create_mock_model_handle(backend="llama_cpp")
        runner = LlamaCppRunner(handle)
        runner._pending_context = "Some context"

        assert runner.has_pending_context() is True

    def test_has_pending_context_false(self):
        """has_pending_context returns False when no context."""
        handle = create_mock_model_handle(backend="llama_cpp")
        runner = LlamaCppRunner(handle)

        assert runner.has_pending_context() is False


class TestLLMRunnerGetChatReplyStructure:
    """Tests for LLMRunner.get_chat_reply_structure() method."""

    def test_get_chat_reply_structure_returns_template(self):
        """get_chat_reply_structure returns chat template from handle."""
        handle = create_mock_model_handle(backend="llama_cpp")
        handle.get_chat_template.return_value = "{% for message in messages %}..."
        runner = LlamaCppRunner(handle)

        result = runner.get_chat_reply_structure()

        assert result == "{% for message in messages %}..."
        handle.get_chat_template.assert_called_once()

    def test_get_chat_reply_structure_returns_none(self):
        """get_chat_reply_structure returns None when no template."""
        handle = create_mock_model_handle(backend="llama_cpp")
        handle.get_chat_template.return_value = None
        runner = LlamaCppRunner(handle)

        result = runner.get_chat_reply_structure()

        assert result is None


class TestLLMRunnerInit:
    """Tests for LLMRunner.__init__ via concrete implementation."""

    def test_init_sets_attributes_from_handle(self):
        """__init__ sets attributes from ModelHandle."""
        handle = create_mock_model_handle(
            backend="llama_cpp",
            model_name="test-model",
            max_context_length=8192,
        )
        handle.config.device = "cuda"
        handle.config.dtype = "float16"
        handle.config.max_new_tokens = 1024

        runner = LlamaCppRunner(handle)

        assert runner.handle is handle
        assert runner.config is handle.config
        assert runner.model_name == "test-model"
        assert runner.device == "cuda"
        assert runner.dtype == "float16"
        assert runner.max_new_tokens == 1024
        assert runner.max_context_length == 8192
        assert runner.model is handle.model
        assert runner.tokenizer is handle.tokenizer

    def test_init_creates_tool_registry(self):
        """__init__ creates empty tool registry."""
        handle = create_mock_model_handle(backend="llama_cpp")

        runner = LlamaCppRunner(handle)

        assert runner.tool_registry is not None

    def test_init_with_knowledge_store_registers_tool(self, mock_knowledge_store):
        """__init__ registers knowledge_search tool when knowledge_store provided."""
        handle = create_mock_model_handle(backend="llama_cpp")

        runner = LlamaCppRunner(handle, knowledge_store=mock_knowledge_store)

        assert runner.knowledge_store is mock_knowledge_store
        assert "knowledge_search" in runner.tool_registry.list_tools()

    def test_init_without_knowledge_store(self):
        """__init__ does not register tool when no knowledge_store."""
        handle = create_mock_model_handle(backend="llama_cpp")

        runner = LlamaCppRunner(handle, knowledge_store=None)

        assert runner.knowledge_store is None
        assert "knowledge_search" not in runner.tool_registry.list_tools()

    def test_init_pending_context_is_none(self):
        """__init__ initializes pending context to None."""
        handle = create_mock_model_handle(backend="llama_cpp")

        runner = LlamaCppRunner(handle)

        assert runner._pending_context is None