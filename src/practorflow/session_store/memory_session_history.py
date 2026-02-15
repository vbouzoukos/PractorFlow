"""
In-memory session history implementation.

Provides session history access backed by InMemorySessionStore.
Sessions are stored in memory and lost on application restart.
"""

from typing import Dict, List, Optional

from practorflow.llm.base.session import Session
from practorflow.session_store.session_history import SessionHistory


class InMemorySessionHistory(SessionHistory):
    """
    In-memory implementation of SessionHistory.

    Stores sessions in a dictionary. Suitable for development
    and testing, or single-session applications where persistence
    is not required.
    """

    def __init__(self, sessions: Optional[Dict[str, Session]] = None):
        """
        Initialize in-memory session history.

        Args:
            sessions: Optional existing sessions dict to use.
                      If None, creates an empty dict.
        """
        self._sessions: Dict[str, Session] = sessions if sessions is not None else {}

    def list_sessions(self, user: Optional[str] = None) -> List[Session]:
        """
        List all sessions, optionally filtered by user.

        Args:
            user: Optional user identifier to filter sessions.
                  If None, returns all sessions.

        Returns:
            List of Session objects sorted by updated_at descending.
        """
        sessions = list(self._sessions.values())

        if user is not None:
            sessions = [s for s in sessions if s.user == user]

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
        return self._sessions.get(session_id)

    def sessions_by_title(self, title: str, user: Optional[str] = None) -> List[Session]:
        """
        Get sessions where their titles contain the search term.

        Args:
            title: Search term (case-insensitive).
            user: Optional user identifier to filter sessions.

        Returns:
            List of Session objects sorted by relevance then updated_at.
        """
        search_term = title.lower()

        matching = [
            s
            for s in self._sessions.values()
            if s.title and search_term in s.title.lower() and (user is None or s.user == user)
        ]

        matching.sort(
            key=lambda s: (
                not s.title.lower().startswith(search_term),
                -s.updated_at.timestamp(),
            )
        )

        return matching
