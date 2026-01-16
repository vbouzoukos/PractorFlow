"""
Session history abstract class.

Defines the interface for listing sessions and retrieving full message history.
Enables GUI chat clients to display and select past conversations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from practorflow.llm.base.session import Session


class SessionHistory(ABC):
    """
    Abstract base class for session history access.
    
    Provides methods for listing all sessions and retrieving
    full session history including messages. Supports optional
    user filtering for multi-user scenarios.
    """
    
    @abstractmethod
    def list_sessions(self, user: Optional[str] = None) -> List[Session]:
        """
        List all sessions, optionally filtered by user.
        
        Args:
            user: Optional user identifier to filter sessions.
                  If None, returns all sessions.
        
        Returns:
            List of Session objects.
        """
        pass  # pragma: no cover
    
    @abstractmethod
    def get_history(self, session_id: str) -> Optional[Session]:
        """
        Get full session with complete message history.
        
        Args:
            session_id: Session ID to retrieve.
        
        Returns:
            Session object with full message history,
            or None if session not found.
        """
        pass  # pragma: no cover

    @abstractmethod
    def sessions_by_title(self, title: str) -> List[Session]:
        """
        Get session where their titles are like title argument.
        
        Args:
            title: Search term.
        
        Returns:
            List of Session objects.
        """
        pass  # pragma: no cover