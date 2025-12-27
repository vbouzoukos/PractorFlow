"""
In-memory session history implementation.

Provides session history access backed by InMemorySessionStore.
Sessions are stored in memory and lost on application restart.
"""

from typing import Dict, List, Optional

from practorflow.llm.base.session import Session
from session_store.session_history import SessionHistory


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
    
    def set_sessions(self, sessions: Dict[str, Session]) -> None:
        """
        Set the sessions dictionary reference.
        
        Allows sharing session storage with InMemorySessionStore.
        
        Args:
            sessions: Sessions dict to use.
        """
        self._sessions = sessions
    
    def __repr__(self) -> str:
        return f"InMemorySessionHistory(sessions={len(self._sessions)})"