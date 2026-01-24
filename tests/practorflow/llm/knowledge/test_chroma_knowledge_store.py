"""
Unit tests for ChromaKnowledgeStore.

Tests cover all ChromaKnowledgeStore methods for 100% code coverage.
Uses mocks from chroma_mocks.py to avoid ChromaDB/embedding dependencies.
"""

import io
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from practorflow.llm.knowledge.chroma_knowledge_store import (
    ChromaKnowledgeStore,
    validate_vector_db_path,
    validate_model_path,
)
from practorflow.llm.knowledge.chroma_knowledge_config import ChromaKnowledgeStoreConfig
from tests.practorflow.llm.knowledge.chroma_mocks import (
    MockChromaClient,
    MockChromaCollection,
    MockEmbeddingModel,
    MockDocumentLoader,
    create_sample_document,
)


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock(spec=ChromaKnowledgeStoreConfig)
    config.persist_directory = "/tmp/test_chroma"
    config.embedding_model_dir = "/tmp/test_models"
    config.embedding_model_name = "test-model"
    config.retrieval_collection_name = "retrieval"
    config.context_collection_name = "context"
    config.documents_collection_name = "documents"
    config.distance_metric = "cosine"
    config.anonymized_telemetry = False
    config.allow_reset = True
    config.batch_size = 100
    config.retrieval_chunk_size = 256
    config.retrieval_chunk_overlap = 50
    config.context_chunk_size = 1024
    config.context_chunk_overlap = 100
    return config


@pytest.fixture
def mock_chroma_client():
    """Create a mock ChromaDB client."""
    return MockChromaClient()


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    return MockEmbeddingModel(dimension=384)


@pytest.fixture
def mock_document_loader():
    """Create a mock document loader."""
    return MockDocumentLoader()


@pytest.fixture
def chroma_store(mock_config, mock_chroma_client, mock_embedding_model, mock_document_loader):
    """Create a ChromaKnowledgeStore with mocked dependencies."""
    with patch("practorflow.llm.knowledge.chroma_knowledge_store.chromadb.PersistentClient", return_value=mock_chroma_client), \
         patch("practorflow.llm.knowledge.chroma_knowledge_store.SentenceTransformerEmbeddingModel", return_value=mock_embedding_model), \
         patch("practorflow.llm.knowledge.chroma_knowledge_store.DocumentLoader", return_value=mock_document_loader), \
         patch("practorflow.llm.knowledge.chroma_knowledge_store.os.makedirs"), \
         patch("practorflow.llm.knowledge.chroma_knowledge_store.os.path.exists", return_value=True):
        store = ChromaKnowledgeStore(config=mock_config)
        return store


class TestValidatePaths:
    """Tests for path validation functions."""

    def test_validate_vector_db_path_exists(self):
        """validate_vector_db_path does not warn when path exists."""
        with patch("practorflow.llm.knowledge.chroma_knowledge_store.os.path.exists", return_value=True):
            validate_vector_db_path("/existing/path")

    def test_validate_vector_db_path_not_exists(self):
        """validate_vector_db_path warns when path does not exist."""
        with patch("practorflow.llm.knowledge.chroma_knowledge_store.os.path.exists", return_value=False), \
             patch("practorflow.llm.knowledge.chroma_knowledge_store.logger") as mock_logger:
            validate_vector_db_path("/nonexistent/path")
            mock_logger.warning.assert_called_once()

    def test_validate_model_path_exists(self):
        """validate_model_path does not warn when path exists."""
        with patch("practorflow.llm.knowledge.chroma_knowledge_store.os.path.exists", return_value=True):
            validate_model_path("/existing/model/path")

    def test_validate_model_path_not_exists(self):
        """validate_model_path warns when path does not exist."""
        with patch("practorflow.llm.knowledge.chroma_knowledge_store.os.path.exists", return_value=False), \
             patch("practorflow.llm.knowledge.chroma_knowledge_store.logger") as mock_logger:
            validate_model_path("/nonexistent/model/path")
            mock_logger.warning.assert_called_once()


class TestChromaKnowledgeStoreInit:
    """Tests for ChromaKnowledgeStore.__init__"""

    def test_init_creates_collections(self, chroma_store):
        """__init__ creates retrieval, context, and documents collections."""
        assert chroma_store.retrieval_collection is not None
        assert chroma_store.context_collection is not None
        assert chroma_store.documents_collection is not None

    def test_init_sets_embedding_dimension(self, chroma_store):
        """__init__ sets embedding dimension from model."""
        assert chroma_store.dimension == 384

    def test_init_with_default_config(self, mock_chroma_client, mock_embedding_model, mock_document_loader):
        """__init__ uses default config when none provided."""
        with patch("practorflow.llm.knowledge.chroma_knowledge_store.chromadb.PersistentClient", return_value=mock_chroma_client), \
             patch("practorflow.llm.knowledge.chroma_knowledge_store.SentenceTransformerEmbeddingModel", return_value=mock_embedding_model), \
             patch("practorflow.llm.knowledge.chroma_knowledge_store.DocumentLoader", return_value=mock_document_loader), \
             patch("practorflow.llm.knowledge.chroma_knowledge_store.os.makedirs"), \
             patch("practorflow.llm.knowledge.chroma_knowledge_store.os.path.exists", return_value=True):
            store = ChromaKnowledgeStore(config=None)
            assert store.config is not None


class TestChromaKnowledgeStoreAddDocument:
    """Tests for add_document_from_* methods."""

    def test_add_document_from_file(self, chroma_store):
        """add_document_from_file stores document and chunks."""
        result = chroma_store.add_document_from_file("/path/to/test.txt")

        assert "id" in result
        assert result["filename"] == "test.txt"
        assert "retrieval_chunk_count" in result
        assert "context_chunk_count" in result

    def test_add_document_from_bytes(self, chroma_store):
        """add_document_from_bytes stores document from bytes."""
        content = b"Test document content from bytes."

        result = chroma_store.add_document_from_bytes(
            file_bytes=content,
            filename="bytes_doc.txt",
            mime_type="text/plain",
        )

        assert "id" in result
        assert result["filename"] == "bytes_doc.txt"

    def test_add_document_from_base64(self, chroma_store):
        """add_document_from_base64 stores document from base64."""
        import base64
        content = base64.b64encode(b"Base64 encoded content").decode()

        result = chroma_store.add_document_from_base64(
            base64_data=content,
            filename="base64_doc.txt",
            mime_type="text/plain",
        )

        assert "id" in result
        assert result["filename"] == "base64_doc.txt"

    def test_add_document_from_stream(self, chroma_store):
        """add_document_from_stream stores document from stream."""
        stream = io.BytesIO(b"Stream content for testing.")

        result = chroma_store.add_document_from_stream(
            file_stream=stream,
            filename="stream_doc.txt",
            mime_type="text/plain",
        )

        assert "id" in result
        assert result["filename"] == "stream_doc.txt"

    def test_add_document_with_metadata(self, chroma_store):
        """add_document_from_file accepts additional metadata."""
        result = chroma_store.add_document_from_file(
            "/path/to/test.txt",
            metadata={"source": "test", "author": "user"},
        )

        assert "id" in result


class TestChromaKnowledgeStoreSearch:
    """Tests for search methods."""

    def test_search_empty_collection(self, chroma_store):
        """search returns empty list when collection is empty."""
        result = chroma_store.search("test query")

        assert result == []

    def test_search_with_results(self, chroma_store):
        """search returns results when documents exist."""
        chroma_store.add_document_from_file("/path/to/test.txt")

        result = chroma_store.search("test query", top_k=5)

        assert isinstance(result, list)

    def test_search_with_filter(self, chroma_store):
        """search accepts filter_metadata parameter."""
        chroma_store.add_document_from_file("/path/to/test.txt")

        result = chroma_store.search(
            "test query",
            top_k=5,
            filter_metadata={"document_id": "doc_123"},
        )

        assert isinstance(result, list)

    def test_search_scoped_empty_collection(self, chroma_store):
        """search_scoped returns empty list when collection is empty."""
        result = chroma_store.search_scoped("test query")

        assert result == []

    def test_search_scoped_with_document_ids(self, chroma_store):
        """search_scoped filters by document IDs."""
        chroma_store.add_document_from_file("/path/to/test.txt")

        result = chroma_store.search_scoped(
            "test query",
            top_k=5,
            document_ids={"doc_123", "doc_456"},
        )

        assert isinstance(result, list)

    def test_search_scoped_without_document_ids(self, chroma_store):
        """search_scoped searches all documents when no IDs specified."""
        chroma_store.add_document_from_file("/path/to/test.txt")

        result = chroma_store.search_scoped("test query", top_k=5)

        assert isinstance(result, list)

    def test_search_by_vector_empty_collection(self, chroma_store):
        """search_by_vector returns empty list when collection is empty."""
        query_vector = np.random.rand(384).astype(np.float32)

        result = chroma_store.search_by_vector(query_vector)

        assert result == []

    def test_search_by_vector_with_results(self, chroma_store):
        """search_by_vector returns results when documents exist."""
        chroma_store.add_document_from_file("/path/to/test.txt")
        query_vector = np.random.rand(384).astype(np.float32)

        result = chroma_store.search_by_vector(query_vector, top_k=5)

        assert isinstance(result, list)

    def test_search_by_vector_with_single_filter(self, chroma_store):
        """search_by_vector handles single filter metadata."""
        chroma_store.add_document_from_file("/path/to/test.txt")
        query_vector = np.random.rand(384).astype(np.float32)

        result = chroma_store.search_by_vector(
            query_vector,
            top_k=5,
            filter_metadata={"document_id": "doc_123"},
        )

        assert isinstance(result, list)

    def test_search_by_vector_with_multiple_filters(self, chroma_store):
        """search_by_vector handles multiple filter metadata."""
        chroma_store.add_document_from_file("/path/to/test.txt")
        query_vector = np.random.rand(384).astype(np.float32)

        result = chroma_store.search_by_vector(
            query_vector,
            top_k=5,
            filter_metadata={"document_id": "doc_123", "file_type": "txt"},
        )

        assert isinstance(result, list)

    def test_search_by_vector_with_list_input(self, chroma_store):
        """search_by_vector accepts list as query vector."""
        chroma_store.add_document_from_file("/path/to/test.txt")
        query_vector = [0.1] * 384

        result = chroma_store.search_by_vector(query_vector, top_k=5)

        assert isinstance(result, list)


class TestChromaKnowledgeStoreGetDocument:
    """Tests for get_document method."""

    def test_get_document_existing(self, chroma_store):
        """get_document returns document by ID."""
        add_result = chroma_store.add_document_from_file("/path/to/test.txt")
        doc_id = add_result["id"]

        result = chroma_store.get_document(doc_id)

        assert result is not None
        assert result["id"] == doc_id

    def test_get_document_nonexistent(self, chroma_store):
        """get_document returns None for nonexistent ID."""
        result = chroma_store.get_document("nonexistent_id")

        assert result is None

    def test_get_document_handles_exception(self, chroma_store):
        """get_document returns None on exception."""
        chroma_store.documents_collection.get = MagicMock(side_effect=Exception("DB error"))

        result = chroma_store.get_document("doc_123")

        assert result is None


class TestChromaKnowledgeStoreGetChunk:
    """Tests for get_chunk and get_context_chunk methods."""

    def test_get_chunk_existing(self, chroma_store):
        """get_chunk returns chunk by ID."""
        add_result = chroma_store.add_document_from_file("/path/to/test.txt")
        doc_id = add_result["id"]
        chunk_id = f"{doc_id}_chunk_0"

        result = chroma_store.get_chunk(chunk_id)

        assert result is not None
        assert result["id"] == chunk_id

    def test_get_chunk_nonexistent(self, chroma_store):
        """get_chunk returns None for nonexistent ID."""
        result = chroma_store.get_chunk("nonexistent_chunk")

        assert result is None

    def test_get_chunk_handles_exception(self, chroma_store):
        """get_chunk returns None on exception."""
        chroma_store.retrieval_collection.get = MagicMock(side_effect=Exception("DB error"))

        result = chroma_store.get_chunk("chunk_123")

        assert result is None

    def test_get_context_chunk_existing(self, chroma_store):
        """get_context_chunk returns context chunk by key."""
        add_result = chroma_store.add_document_from_file("/path/to/test.txt")
        doc_id = add_result["id"]
        parent_key = f"{doc_id}_ctx_0"

        result = chroma_store.get_context_chunk(parent_key)

        assert result is not None
        assert result["id"] == parent_key

    def test_get_context_chunk_nonexistent(self, chroma_store):
        """get_context_chunk returns None for nonexistent key."""
        result = chroma_store.get_context_chunk("nonexistent_parent")

        assert result is None

    def test_get_context_chunk_handles_exception(self, chroma_store):
        """get_context_chunk returns None on exception."""
        chroma_store.context_collection.get = MagicMock(side_effect=Exception("DB error"))

        result = chroma_store.get_context_chunk("parent_123")

        assert result is None


class TestChromaKnowledgeStoreGetChunksByDocument:
    """Tests for get_*_chunks_by_document methods."""

    def test_get_retrieval_chunks_by_document(self, chroma_store):
        """get_retrieval_chunks_by_document returns chunks for document."""
        add_result = chroma_store.add_document_from_file("/path/to/test.txt")
        doc_id = add_result["id"]

        result = chroma_store.get_retrieval_chunks_by_document(doc_id)

        assert isinstance(result, list)

    def test_get_retrieval_chunks_by_document_empty(self, chroma_store):
        """get_retrieval_chunks_by_document returns empty list when no chunks."""
        result = chroma_store.get_retrieval_chunks_by_document("nonexistent_doc")

        assert result == []

    def test_get_retrieval_chunks_by_document_handles_exception(self, chroma_store):
        """get_retrieval_chunks_by_document returns empty list on exception."""
        chroma_store.retrieval_collection.get = MagicMock(side_effect=Exception("DB error"))

        result = chroma_store.get_retrieval_chunks_by_document("doc_123")

        assert result == []

    def test_get_context_chunks_by_document(self, chroma_store):
        """get_context_chunks_by_document returns context chunks for document."""
        add_result = chroma_store.add_document_from_file("/path/to/test.txt")
        doc_id = add_result["id"]

        result = chroma_store.get_context_chunks_by_document(doc_id)

        assert isinstance(result, list)

    def test_get_context_chunks_by_document_empty(self, chroma_store):
        """get_context_chunks_by_document returns empty list when no chunks."""
        result = chroma_store.get_context_chunks_by_document("nonexistent_doc")

        assert result == []

    def test_get_context_chunks_by_document_handles_exception(self, chroma_store):
        """get_context_chunks_by_document returns empty list on exception."""
        chroma_store.context_collection.get = MagicMock(side_effect=Exception("DB error"))

        result = chroma_store.get_context_chunks_by_document("doc_123")

        assert result == []


class TestChromaKnowledgeStoreDeleteDocument:
    """Tests for delete_document method."""

    def test_delete_document_existing(self, chroma_store):
        """delete_document removes document and returns True."""
        add_result = chroma_store.add_document_from_file("/path/to/test.txt")
        doc_id = add_result["id"]

        result = chroma_store.delete_document(doc_id)

        assert result is True
        assert chroma_store.get_document(doc_id) is None

    def test_delete_document_nonexistent(self, chroma_store):
        """delete_document returns False for nonexistent document."""
        result = chroma_store.delete_document("nonexistent_doc")

        assert result is False

    def test_delete_document_removes_chunks(self, chroma_store):
        """delete_document removes associated chunks."""
        add_result = chroma_store.add_document_from_file("/path/to/test.txt")
        doc_id = add_result["id"]

        chroma_store.delete_document(doc_id)

        retrieval_chunks = chroma_store.get_retrieval_chunks_by_document(doc_id)
        context_chunks = chroma_store.get_context_chunks_by_document(doc_id)

        assert retrieval_chunks == []
        assert context_chunks == []

    def test_delete_document_raises_on_exception(self, chroma_store):
        """delete_document raises exception on error."""
        add_result = chroma_store.add_document_from_file("/path/to/test.txt")
        doc_id = add_result["id"]
        chroma_store.retrieval_collection.get = MagicMock(side_effect=Exception("DB error"))

        with pytest.raises(Exception, match="DB error"):
            chroma_store.delete_document(doc_id)


class TestChromaKnowledgeStoreListDocuments:
    """Tests for list_documents method."""

    def test_list_documents_empty(self, chroma_store):
        """list_documents returns empty list when no documents."""
        result = chroma_store.list_documents()

        assert result == []

    def test_list_documents_with_documents(self, chroma_store):
        """list_documents returns document summaries."""
        chroma_store.add_document_from_file("/path/to/test1.txt")
        chroma_store.add_document_from_file("/path/to/test2.pdf")

        result = chroma_store.list_documents()

        assert len(result) == 2
        assert all("id" in doc for doc in result)
        assert all("filename" in doc for doc in result)

    def test_list_documents_raises_on_exception(self, chroma_store):
        """list_documents raises exception on error."""
        chroma_store.documents_collection.get = MagicMock(side_effect=Exception("DB error"))

        with pytest.raises(Exception, match="DB error"):
            chroma_store.list_documents()


class TestChromaKnowledgeStoreCountMethods:
    """Tests for count_documents and count_chunks methods."""

    def test_count_documents_empty(self, chroma_store):
        """count_documents returns 0 when empty."""
        assert chroma_store.count_documents() == 0

    def test_count_documents_with_documents(self, chroma_store):
        """count_documents returns correct count."""
        chroma_store.add_document_from_file("/path/to/test1.txt")
        chroma_store.add_document_from_file("/path/to/test2.txt")

        assert chroma_store.count_documents() == 2

    def test_count_chunks_empty(self, chroma_store):
        """count_chunks returns 0 when empty."""
        assert chroma_store.count_chunks() == 0

    def test_count_chunks_with_documents(self, chroma_store):
        """count_chunks returns correct count."""
        chroma_store.add_document_from_file("/path/to/test.txt")

        assert chroma_store.count_chunks() >= 1


class TestChromaKnowledgeStoreGetStats:
    """Tests for get_stats method."""

    def test_get_stats_returns_dict(self, chroma_store):
        """get_stats returns statistics dictionary."""
        result = chroma_store.get_stats()

        assert isinstance(result, dict)
        assert "documents" in result
        assert "retrieval_chunks" in result
        assert "context_chunks" in result
        assert "embedding_dimension" in result
        assert "embedding_model" in result
        assert "persist_directory" in result
        assert "distance_metric" in result
        assert "storage_type" in result


class TestChromaKnowledgeStoreClear:
    """Tests for clear method."""

    def test_clear_removes_all_data(self, chroma_store):
        """clear removes all documents and chunks."""
        chroma_store.add_document_from_file("/path/to/test1.txt")
        chroma_store.add_document_from_file("/path/to/test2.txt")

        chroma_store.clear()

        assert chroma_store.count_documents() == 0
        assert chroma_store.count_chunks() == 0

    def test_clear_raises_on_exception(self, chroma_store):
        """clear raises exception on error."""
        chroma_store.client.delete_collection = MagicMock(side_effect=Exception("DB error"))

        with pytest.raises(Exception, match="DB error"):
            chroma_store.clear()


class TestChromaKnowledgeStoreSearchScopedExceptions:
    """Tests for search_scoped exception handling and edge cases."""

    def test_search_scoped_with_single_document_id(self, chroma_store):
        """search_scoped uses single document filter when one ID provided."""
        add_result = chroma_store.add_document_from_file("/path/to/test.txt")
        doc_id = add_result["id"]

        result = chroma_store.search_scoped(
            "test query",
            top_k=5,
            document_ids={doc_id},
        )

        assert isinstance(result, list)

    def test_search_scoped_returns_empty_when_query_has_no_results(self, chroma_store):
        """search_scoped returns empty list when query returns no matching chunks."""
        chroma_store.add_document_from_file("/path/to/test.txt")

        chroma_store.retrieval_collection.query = MagicMock(return_value={
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
        })

        result = chroma_store.search_scoped("nonexistent query", top_k=5)

        assert result == []

    def test_search_scoped_returns_empty_when_ids_is_none(self, chroma_store):
        """search_scoped returns empty list when query returns None ids."""
        chroma_store.add_document_from_file("/path/to/test.txt")

        chroma_store.retrieval_collection.query = MagicMock(return_value={
            "ids": None,
            "distances": None,
            "documents": None,
            "metadatas": None,
        })

        result = chroma_store.search_scoped("test query", top_k=5)

        assert result == []

    def test_search_scoped_handles_parent_fetch_exception(self, chroma_store):
        """search_scoped continues when fetching parent chunk fails."""
        add_result = chroma_store.add_document_from_file("/path/to/test.txt")
        doc_id = add_result["id"]

        original_get = chroma_store.context_collection.get

        def failing_get(*args, **kwargs):
            ids = kwargs.get("ids", [])
            if ids and "ctx" in ids[0]:
                raise Exception("Parent fetch error")
            return original_get(*args, **kwargs)

        chroma_store.context_collection.get = MagicMock(side_effect=failing_get)

        result = chroma_store.search_scoped("test query", top_k=5)

        assert isinstance(result, list)

    def test_search_scoped_skips_chunks_without_parent_key(self, chroma_store):
        """search_scoped skips chunks that have no parent_key in metadata."""
        chroma_store.add_document_from_file("/path/to/test.txt")

        chroma_store.retrieval_collection._data["chunk_no_parent"] = {
            "id": "chunk_no_parent",
            "document": "orphan chunk",
            "metadata": {"document_id": "doc_123"},
            "embedding": [0.1] * 384,
        }

        result = chroma_store.search_scoped("test query", top_k=5)

        assert isinstance(result, list)


class TestChromaKnowledgeStoreRepr:
    """Tests for __repr__ method."""

    def test_repr_format(self, chroma_store):
        """__repr__ returns expected format."""
        result = repr(chroma_store)

        assert "ChromaKnowledgeStore" in result
        assert "documents=" in result
        assert "chunks=" in result
        assert "path=" in result


class TestChromaKnowledgeStoreTimestamp:
    """Tests for _get_current_timestamp method."""

    def test_get_current_timestamp_returns_iso_string(self, chroma_store):
        """_get_current_timestamp returns ISO format string."""
        result = chroma_store._get_current_timestamp()

        assert isinstance(result, str)
        assert "T" in result