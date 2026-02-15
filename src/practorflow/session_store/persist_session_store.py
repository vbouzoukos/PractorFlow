"""
TinyDB-based persistent session store.

Provides durable session storage using TinyDB as a JSON-based document store.
Sessions persist across application restarts.

Requirements:
    pip install tinydb

Usage:
    from persist_session_store import TinyDBSessionStore

    store = TinyDBSessionStore(db_path="./sessions.json")

    # Save session
    store.save(session)

    # Get session
    session = store.get("session_id")

    # Check existence
    if store.exists("session_id"):
        ...

    # Delete session
    store.delete("session_id")
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from tinydb import TinyDB, Query

from practorflow.llm.base.session import Message, Session
from practorflow.llm.base.session_store import SessionStore


class TinyDBSessionStore(SessionStore):
    """
    TinyDB-based persistent session store.

    Stores sessions as JSON documents in a file-based TinyDB database.
    Handles serialization and deserialization of Session and Message objects.
    """

    def __init__(self, db_path: str = "./sessions.json"):
        """
        Initialize TinyDB session store.

        Args:
            db_path: Path to the TinyDB JSON file.
                    Directory will be created if it doesn't exist.
        """
        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._db = TinyDB(db_path)
        self._sessions = self._db.table("sessions")
        self._query = Query()

    def get(self, session_id: str) -> Session:
        """
        Get a session by ID.

        Creates a new session if it doesn't exist.

        Args:
            session_id: Session ID to retrieve.

        Returns:
            Session object.
        """
        result = self._sessions.get(self._query.session_id == session_id)

        if result is None:
            session = Session(session_id=session_id)
            self.save(session)
            return session

        return self._deserialize_session(result)

    def save(self, session: Session) -> None:
        """
        Save a session.

        Updates existing session or inserts new one.

        Args:
            session: Session object to save.
        """
        session.updated_at = datetime.now()
        data = self._serialize_session(session)

        existing = self._sessions.get(self._query.session_id == session.session_id)

        if existing:
            self._sessions.update(data, self._query.session_id == session.session_id)
        else:
            self._sessions.insert(data)

    def delete(self, session_id: str) -> None:
        """
        Delete a session.

        Args:
            session_id: Session ID to delete.
        """
        self._sessions.remove(self._query.session_id == session_id)

    def exists(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: Session ID to check.

        Returns:
            True if session exists, False otherwise.
        """
        return self._sessions.contains(self._query.session_id == session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions with summary information.

        Returns:
            List of session summary dicts with session_id, created_at, updated_at, message_count.
        """
        sessions = []
        for doc in self._sessions.all():
            sessions.append(
                {
                    "session_id": doc.get("session_id"),
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                    "message_count": len(doc.get("messages", [])),
                    "document_count": len(doc.get("documents", [])),
                }
            )
        return sessions

    def clear_all(self) -> int:
        """
        Delete all sessions.

        Returns:
            Number of sessions deleted.
        """
        count = len(self._sessions)
        self._sessions.truncate()
        return count

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def _serialize_session(self, session: Session) -> Dict[str, Any]:
        """
        Serialize a Session object to a JSON-compatible dict.

        Args:
            session: Session object to serialize.

        Returns:
            Dictionary suitable for TinyDB storage.
        """
        return {
            "session_id": session.session_id,
            "messages": [self._serialize_message(msg) for msg in session.messages],
            "instructions": session.instructions,
            "documents": session.documents,
            "metadata": session.metadata,
            "created_at": self._serialize_datetime(session.created_at),
            "updated_at": self._serialize_datetime(session.updated_at),
            "user": session.user,
            "title": session.title,
        }

    def _deserialize_session(self, data: Dict[str, Any]) -> Session:
        """
        Deserialize a dict to a Session object.

        Args:
            data: Dictionary from TinyDB.

        Returns:
            Reconstructed Session object.
        """
        return Session(
            session_id=data.get("session_id", ""),
            messages=[
                self._deserialize_message(msg) for msg in data.get("messages", [])
            ],
            instructions=data.get("instructions"),
            documents=data.get("documents", []),
            metadata=data.get("metadata", {}),
            created_at=self._deserialize_datetime(data.get("created_at")),
            updated_at=self._deserialize_datetime(data.get("updated_at")),
            user=data.get("user"),
            title=data.get("title"),
        )

    def _serialize_message(self, message: Message) -> Dict[str, Any]:
        """
        Serialize a Message object to a JSON-compatible dict.

        Args:
            message: Message object to serialize.

        Returns:
            Dictionary suitable for JSON storage.
        """
        return {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "type": message.type,
            "status": message.status,
            "timestamp": self._serialize_datetime(message.timestamp),
        }

    def _deserialize_message(self, data: Dict[str, Any]) -> Message:
        """
        Deserialize a dict to a Message object.

        Args:
            data: Dictionary from storage.

        Returns:
            Reconstructed Message object.
        """
        return Message(
            id=data.get("id", ""),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            type=data.get("type", "message"),
            status=data.get("status", "completed"),
            timestamp=self._deserialize_datetime(data.get("timestamp")),
        )

    def _serialize_datetime(self, dt: Optional[datetime]) -> Optional[str]:
        """
        Serialize datetime to ISO format string.

        Args:
            dt: Datetime object or None.

        Returns:
            ISO format string or None.
        """
        if dt is None:
            return None
        return dt.isoformat()

    def _deserialize_datetime(self, value: Optional[str]) -> datetime:
        """
        Deserialize ISO format string to datetime.

        Args:
            value: ISO format string or None.

        Returns:
            Datetime object (defaults to now if None or invalid).
        """
        if value is None:
            return datetime.now()
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return datetime.now()
