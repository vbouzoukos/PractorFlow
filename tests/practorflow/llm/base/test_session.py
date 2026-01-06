"""
Unit tests for Session and Message classes.

Tests cover all methods for 100% code coverage.
"""

import pytest
from datetime import datetime

from practorflow.llm.base.session import Message, Session


class TestMessageInit:
    """Tests for Message.__init__"""

    def test_message_required_fields(self):
        """Message initializes with required fields."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_default_id(self):
        """Message generates unique ID by default."""
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="user", content="Hello")

        assert msg1.id.startswith("msg_")
        assert msg2.id.startswith("msg_")
        assert msg1.id != msg2.id

    def test_message_default_type(self):
        """Message defaults type to 'message'."""
        msg = Message(role="user", content="Hello")

        assert msg.type == "message"

    def test_message_default_status(self):
        """Message defaults status to 'completed'."""
        msg = Message(role="user", content="Hello")

        assert msg.status == "completed"

    def test_message_default_timestamp(self):
        """Message defaults timestamp to now."""
        before = datetime.now()
        msg = Message(role="user", content="Hello")
        after = datetime.now()

        assert before <= msg.timestamp <= after

    def test_message_custom_fields(self):
        """Message accepts custom field values."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        msg = Message(
            role="assistant",
            content="Response",
            id="custom-id",
            type="custom-type",
            status="pending",
            timestamp=timestamp,
        )

        assert msg.id == "custom-id"
        assert msg.type == "custom-type"
        assert msg.status == "pending"
        assert msg.timestamp == timestamp


class TestMessageToDict:
    """Tests for Message.to_dict()"""

    def test_to_dict_contains_all_fields(self):
        """to_dict returns dict with expected fields."""
        msg = Message(
            role="user",
            content="Hello",
            id="msg-123",
            type="message",
            status="completed",
        )

        result = msg.to_dict()

        assert result["id"] == "msg-123"
        assert result["role"] == "user"
        assert result["type"] == "message"
        assert result["status"] == "completed"
        assert result["content"] == "Hello"

    def test_to_dict_excludes_timestamp(self):
        """to_dict does not include timestamp."""
        msg = Message(role="user", content="Hello")

        result = msg.to_dict()

        assert "timestamp" not in result


class TestMessageGetTextContent:
    """Tests for Message.get_text_content()"""

    def test_get_text_content_string(self):
        """get_text_content returns string content directly."""
        msg = Message(role="user", content="Simple text message")

        result = msg.get_text_content()

        assert result == "Simple text message"

    def test_get_text_content_structured_list(self):
        """get_text_content extracts text from structured content list."""
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "First part"},
                {"type": "text", "text": "Second part"},
            ],
        )

        result = msg.get_text_content()

        assert result == "First part\nSecond part"

    def test_get_text_content_mixed_structured(self):
        """get_text_content handles mixed content types."""
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Text content"},
                {"type": "image", "url": "http://example.com/img.png"},
                {"type": "text", "text": "More text"},
            ],
        )

        result = msg.get_text_content()

        assert result == "Text content\nMore text"

    def test_get_text_content_empty_list(self):
        """get_text_content returns empty string for empty list."""
        msg = Message(role="user", content=[])

        result = msg.get_text_content()

        assert result == ""

    def test_get_text_content_other_type(self):
        """get_text_content converts other types to string."""
        msg = Message(role="user", content=12345)

        result = msg.get_text_content()

        assert result == "12345"


class TestSessionInit:
    """Tests for Session.__init__"""

    def test_session_required_fields(self):
        """Session initializes with required session_id."""
        session = Session(session_id="sess-123")

        assert session.session_id == "sess-123"

    def test_session_default_messages(self):
        """Session defaults messages to empty list."""
        session = Session(session_id="sess-123")

        assert session.messages == []

    def test_session_default_documents(self):
        """Session defaults documents to empty list."""
        session = Session(session_id="sess-123")

        assert session.documents == []

    def test_session_default_metadata(self):
        """Session defaults metadata to empty dict."""
        session = Session(session_id="sess-123")

        assert session.metadata == {}

    def test_session_default_timestamps(self):
        """Session defaults timestamps to now."""
        before = datetime.now()
        session = Session(session_id="sess-123")
        after = datetime.now()

        assert before <= session.created_at <= after
        assert before <= session.updated_at <= after

    def test_session_custom_fields(self):
        """Session accepts custom field values."""
        messages = [Message(role="user", content="Hello")]
        documents = [{"id": "doc-1", "filename": "test.txt"}]
        session = Session(
            session_id="sess-456",
            messages=messages,
            instructions="Be helpful",
            documents=documents,
            metadata={"key": "value"},
            user="test_user",
        )

        assert session.messages == messages
        assert session.instructions == "Be helpful"
        assert session.documents == documents
        assert session.metadata == {"key": "value"}
        assert session.user == "test_user"


class TestSessionAddDocument:
    """Tests for Session.add_document()"""

    def test_add_document_new(self):
        """add_document adds new document to session."""
        session = Session(session_id="sess-123")
        doc = {"id": "doc-1", "filename": "test.txt", "content": "Test content"}

        session.add_document(doc)

        assert len(session.documents) == 1
        assert session.documents[0] == doc

    def test_add_document_updates_existing(self):
        """add_document updates document with same ID."""
        session = Session(session_id="sess-123")
        doc1 = {"id": "doc-1", "filename": "old.txt", "content": "Old content"}
        doc2 = {"id": "doc-1", "filename": "new.txt", "content": "New content"}
        session.add_document(doc1)

        session.add_document(doc2)

        assert len(session.documents) == 1
        assert session.documents[0]["filename"] == "new.txt"
        assert session.documents[0]["content"] == "New content"

    def test_add_document_updates_timestamp(self):
        """add_document updates session timestamp."""
        session = Session(session_id="sess-123")
        original_time = session.updated_at
        doc = {"id": "doc-1", "filename": "test.txt"}

        session.add_document(doc)

        assert session.updated_at >= original_time


class TestSessionRemoveDocument:
    """Tests for Session.remove_document()"""

    def test_remove_document_existing(self):
        """remove_document removes document and returns True."""
        session = Session(session_id="sess-123")
        session.documents = [
            {"id": "doc-1", "filename": "file1.txt"},
            {"id": "doc-2", "filename": "file2.txt"},
        ]

        result = session.remove_document("doc-1")

        assert result is True
        assert len(session.documents) == 1
        assert session.documents[0]["id"] == "doc-2"

    def test_remove_document_nonexistent(self):
        """remove_document returns False for nonexistent document."""
        session = Session(session_id="sess-123")
        session.documents = [{"id": "doc-1", "filename": "file1.txt"}]

        result = session.remove_document("doc-999")

        assert result is False
        assert len(session.documents) == 1

    def test_remove_document_updates_timestamp(self):
        """remove_document updates session timestamp when document removed."""
        session = Session(session_id="sess-123")
        session.documents = [{"id": "doc-1", "filename": "file1.txt"}]
        original_time = session.updated_at

        session.remove_document("doc-1")

        assert session.updated_at >= original_time


class TestSessionGetDocument:
    """Tests for Session.get_document()"""

    def test_get_document_existing(self):
        """get_document returns document by ID."""
        session = Session(session_id="sess-123")
        doc = {"id": "doc-1", "filename": "test.txt", "content": "Content"}
        session.documents = [doc]

        result = session.get_document("doc-1")

        assert result == doc

    def test_get_document_nonexistent(self):
        """get_document returns None for nonexistent document."""
        session = Session(session_id="sess-123")
        session.documents = [{"id": "doc-1", "filename": "test.txt"}]

        result = session.get_document("doc-999")

        assert result is None

    def test_get_document_empty_session(self):
        """get_document returns None for empty session."""
        session = Session(session_id="sess-123")

        result = session.get_document("doc-1")

        assert result is None


class TestSessionClearDocuments:
    """Tests for Session.clear_documents()"""

    def test_clear_documents_removes_all(self):
        """clear_documents removes all documents."""
        session = Session(session_id="sess-123")
        session.documents = [
            {"id": "doc-1", "filename": "file1.txt"},
            {"id": "doc-2", "filename": "file2.txt"},
        ]

        session.clear_documents()

        assert session.documents == []

    def test_clear_documents_updates_timestamp(self):
        """clear_documents updates session timestamp."""
        session = Session(session_id="sess-123")
        session.documents = [{"id": "doc-1", "filename": "file1.txt"}]
        original_time = session.updated_at

        session.clear_documents()

        assert session.updated_at >= original_time

    def test_clear_documents_empty_session(self):
        """clear_documents does nothing for empty session."""
        session = Session(session_id="sess-123")
        original_time = session.updated_at

        session.clear_documents()

        assert session.documents == []
        # Timestamp should not change if already empty
        assert session.updated_at == original_time


class TestSessionGetDocumentCount:
    """Tests for Session.get_document_count()"""

    def test_get_document_count_empty(self):
        """get_document_count returns 0 for empty session."""
        session = Session(session_id="sess-123")

        assert session.get_document_count() == 0

    def test_get_document_count_with_documents(self):
        """get_document_count returns correct count."""
        session = Session(session_id="sess-123")
        session.documents = [
            {"id": "doc-1", "filename": "file1.txt"},
            {"id": "doc-2", "filename": "file2.txt"},
            {"id": "doc-3", "filename": "file3.txt"},
        ]

        assert session.get_document_count() == 3


class TestSessionListDocuments:
    """Tests for Session.list_documents()"""

    def test_list_documents_empty(self):
        """list_documents returns empty list for empty session."""
        session = Session(session_id="sess-123")

        result = session.list_documents()

        assert result == []

    def test_list_documents_returns_summaries(self):
        """list_documents returns document summaries."""
        session = Session(session_id="sess-123")
        session.documents = [
            {"id": "doc-1", "filename": "report.pdf", "file_type": "pdf", "content": "..."},
            {"id": "doc-2", "filename": "data.csv", "file_type": "csv", "content": "..."},
        ]

        result = session.list_documents()

        assert len(result) == 2
        assert result[0] == {"id": "doc-1", "filename": "report.pdf", "file_type": "pdf"}
        assert result[1] == {"id": "doc-2", "filename": "data.csv", "file_type": "csv"}

    def test_list_documents_missing_file_type(self):
        """list_documents uses 'unknown' for missing file_type."""
        session = Session(session_id="sess-123")
        session.documents = [{"id": "doc-1", "filename": "file.txt"}]

        result = session.list_documents()

        assert result[0]["file_type"] == "unknown"