"""
Mock classes and helpers for ChromaKnowledgeStore tests.

Provides mock implementations of ChromaDB client, collections,
embedding model, and document loader for unit testing.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock
import numpy as np


class MockChromaCollection:
    """Mock ChromaDB collection for testing."""

    def __init__(self, name: str = "test_collection"):
        self.name = name
        self._data: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        ids: List[str],
        documents: List[str] = None,
        metadatas: List[Dict[str, Any]] = None,
        embeddings: List[List[float]] = None,
    ) -> None:
        """Add items to the collection."""
        documents = documents or [""] * len(ids)
        metadatas = metadatas or [{}] * len(ids)
        embeddings = embeddings or [None] * len(ids)

        for i, id_ in enumerate(ids):
            self._data[id_] = {
                "id": id_,
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "embedding": embeddings[i] if i < len(embeddings) else None,
            }

    def get(
        self,
        ids: List[str] = None,
        where: Dict[str, Any] = None,
        include: List[str] = None,
    ) -> Dict[str, List]:
        """Get items from the collection."""
        include = include or []
        results = {"ids": []}

        if "documents" in include:
            results["documents"] = []
        if "metadatas" in include:
            results["metadatas"] = []
        if "embeddings" in include:
            results["embeddings"] = []

        if ids:
            for id_ in ids:
                if id_ in self._data:
                    results["ids"].append(id_)
                    if "documents" in include:
                        results["documents"].append(self._data[id_]["document"])
                    if "metadatas" in include:
                        results["metadatas"].append(self._data[id_]["metadata"])
                    if "embeddings" in include:
                        results["embeddings"].append(self._data[id_]["embedding"])
        elif where:
            for id_, item in self._data.items():
                if self._matches_where(item["metadata"], where):
                    results["ids"].append(id_)
                    if "documents" in include:
                        results["documents"].append(item["document"])
                    if "metadatas" in include:
                        results["metadatas"].append(item["metadata"])
                    if "embeddings" in include:
                        results["embeddings"].append(item["embedding"])
        else:
            for id_, item in self._data.items():
                results["ids"].append(id_)
                if "documents" in include:
                    results["documents"].append(item["document"])
                if "metadatas" in include:
                    results["metadatas"].append(item["metadata"])
                if "embeddings" in include:
                    results["embeddings"].append(item["embedding"])

        return results

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Dict[str, Any] = None,
        include: List[str] = None,
    ) -> Dict[str, List]:
        """Query the collection by embeddings."""
        include = include or []
        results = {"ids": [[]], "distances": [[]]}

        if "documents" in include:
            results["documents"] = [[]]
        if "metadatas" in include:
            results["metadatas"] = [[]]

        matching_items = []
        for id_, item in self._data.items():
            if where and not self._matches_where(item["metadata"], where):
                continue
            matching_items.append((id_, item))

        for id_, item in matching_items[:n_results]:
            results["ids"][0].append(id_)
            results["distances"][0].append(0.1)
            if "documents" in include:
                results["documents"][0].append(item["document"])
            if "metadatas" in include:
                results["metadatas"][0].append(item["metadata"])

        return results

    def delete(self, ids: List[str] = None, where: Dict[str, Any] = None) -> None:
        """Delete items from the collection."""
        if ids:
            for id_ in ids:
                self._data.pop(id_, None)
        elif where:
            to_delete = [
                id_ for id_, item in self._data.items()
                if self._matches_where(item["metadata"], where)
            ]
            for id_ in to_delete:
                del self._data[id_]

    def count(self) -> int:
        """Get the number of items in the collection."""
        return len(self._data)

    def _matches_where(self, metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
        """Check if metadata matches where clause."""
        if not where:
            return True

        if "$and" in where:
            return all(self._matches_where(metadata, cond) for cond in where["$and"])

        for key, condition in where.items():
            if key.startswith("$"):
                continue
            if isinstance(condition, dict):
                if "$eq" in condition:
                    if metadata.get(key) != condition["$eq"]:
                        return False
            elif metadata.get(key) != condition:
                return False

        return True


class MockChromaClient:
    """Mock ChromaDB PersistentClient for testing."""

    def __init__(self, path: str = None, settings: Any = None):
        self.path = path
        self.settings = settings
        self._collections: Dict[str, MockChromaCollection] = {}

    def get_or_create_collection(
        self, name: str, metadata: Dict[str, Any] = None
    ) -> MockChromaCollection:
        """Get or create a collection."""
        if name not in self._collections:
            self._collections[name] = MockChromaCollection(name)
        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        self._collections.pop(name, None)


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Generate mock embedding for single text."""
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(self._dimension).astype(np.float32)

    def embed_batch(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        """Generate mock embeddings for batch of texts."""
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % 2**32)
            embeddings.append(np.random.rand(self._dimension).astype(np.float32))
        return np.array(embeddings)


class MockDocumentLoader:
    """Mock document loader for testing."""

    def __init__(self, **kwargs):
        self.config = kwargs

    def load_file(self, filepath: str) -> Dict[str, Any]:
        """Load document from file path."""
        return self._create_mock_document(filepath.split("/")[-1])

    def load_from_bytes(
        self, file_bytes: bytes, filename: str, mime_type: str = None
    ) -> Dict[str, Any]:
        """Load document from bytes."""
        return self._create_mock_document(filename, content=file_bytes.decode("utf-8", errors="ignore"))

    def load_from_base64(
        self, base64_data: str, filename: str = None, mime_type: str = None
    ) -> Dict[str, Any]:
        """Load document from base64."""
        return self._create_mock_document(filename or "base64_doc.txt")

    def load_from_stream(
        self, file_stream: Any, filename: str, mime_type: str = None
    ) -> Dict[str, Any]:
        """Load document from stream."""
        content = file_stream.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        return self._create_mock_document(filename, content=content)

    def _create_mock_document(
        self, filename: str, content: str = None
    ) -> Dict[str, Any]:
        """Create a mock parsed document."""
        import uuid
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        content = content or f"Mock content for {filename}"
        file_type = filename.split(".")[-1] if "." in filename else "txt"

        return {
            "id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "content": content,
            "retrieval_chunks": [
                {
                    "id": "chunk_0",
                    "text": content[:100] if len(content) > 100 else content,
                    "parent_id": "ctx_0",
                    "metadata": {"chunk_index": 0},
                },
            ],
            "context_chunks": [
                {
                    "id": "ctx_0",
                    "text": content,
                    "metadata": {"context_index": 0},
                },
            ],
        }


def create_mock_chroma_store(
    persist_directory: str = "/tmp/test_chroma",
    with_documents: bool = False,
) -> MagicMock:
    """
    Create a mock ChromaKnowledgeStore with injected mocks.

    Args:
        persist_directory: Mock persist directory path.
        with_documents: If True, pre-populate with sample documents.

    Returns:
        Configured mock store ready for testing.
    """
    mock_client = MockChromaClient(path=persist_directory)
    mock_embedding = MockEmbeddingModel()
    mock_loader = MockDocumentLoader()

    return {
        "client": mock_client,
        "embedding_model": mock_embedding,
        "document_loader": mock_loader,
    }


def create_sample_document(
    doc_id: str = "doc_test123",
    filename: str = "test.txt",
    content: str = "Test document content for testing purposes.",
) -> Dict[str, Any]:
    """Create a sample document dict for testing."""
    file_type = filename.split(".")[-1] if "." in filename else "txt"

    return {
        "id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "content": content,
        "retrieval_chunks": [
            {
                "id": "chunk_0",
                "text": content[:50],
                "parent_id": "ctx_0",
                "metadata": {"chunk_index": 0},
            },
            {
                "id": "chunk_1",
                "text": content[50:] if len(content) > 50 else content,
                "parent_id": "ctx_0",
                "metadata": {"chunk_index": 1},
            },
        ],
        "context_chunks": [
            {
                "id": "ctx_0",
                "text": content,
                "metadata": {"context_index": 0},
            },
        ],
    }