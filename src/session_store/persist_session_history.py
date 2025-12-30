"""
TinyDB-based persistent session history implementation.

Provides session history access backed by TinyDB for durable storage.
Sessions persist across application restarts.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from tinydb import TinyDB, Query

from practorflow.llm.base.session import Message, Session
from session_store.session_history import SessionHistory


class PersistSessionHistory(SessionHistory):
    """
    TinyDB-based implementation of SessionHistory.

    Provides persistent session history storage using TinyDB.
    Sessions are stored as JSON documents and survive application restarts.
    """

    def __init__(self, db: Optional[TinyDB] = None, db_path: str = "./sessions.json"):
        """
        Initialize persistent session history.

        Args:
            db: Optional existing TinyDB instance to use.
                If None, creates a new instance at db_path.
            db_path: Path to the TinyDB JSON file (used only if db is None).
        """
        if db is not None:
            self._db = db
            self._owns_db = False
        else:
            self._db = TinyDB(db_path)
            self._owns_db = True

        self._sessions = self._db.table("sessions", cache_size=0)
        self._query = Query()

    def list_sessions(self, user: Optional[str] = None) -> List[Session]:
        """
        List all sessions, optionally filtered by user.

        Args:
            user: Optional user identifier to filter sessions.
                  If None, returns all sessions.

        Returns:
            List of Session objects sorted by updated_at descending.
        """
        if user is not None:
            results = self._sessions.search(self._query.user == user)
        else:
            results = self._sessions.all()

        sessions = [self._deserialize_session(data) for data in results]

        # Sort by updated_at descending (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return sessions

    def get_history(self, session_id: str) -> Optional[Session]:
        """
        Get full session with complete message history.

        Args:
            session_id: Session ID to retrieve.

        Returns:
            Session object with full message history,
            or None if session not found.
        """
        result = self._sessions.get(self._query.session_id == session_id)

        if result is None:
            return None

        return self._deserialize_session(result)

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
        )

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

    def close(self) -> None:
        """Close the database connection if owned by this instance."""
        if self._owns_db:
            self._db.close()

    def __repr__(self) -> str:
        return f"PersistSessionHistory(sessions={len(self._sessions)})"
