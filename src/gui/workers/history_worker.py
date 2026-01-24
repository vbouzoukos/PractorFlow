"""
History worker - Background threads for session history operations.

Handles session listing, history retrieval, and session deletion
in background threads to keep the UI responsive.
"""

from typing import Optional

from PySide6.QtCore import QThread, Signal

from gui.api.session_client import SessionClient


class ListSessionsWorker(QThread):
    """
    Worker thread for listing sessions.
    
    Supports both listing all sessions and searching by term.
    
    Signals:
        sessions_loaded: Emitted with list of SessionSummary on success.
        error_occurred: Emitted with error message on failure.
    """
    
    sessions_loaded = Signal(list)
    error_occurred = Signal(str)
    
    def __init__(
        self,
        client: SessionClient,
        user: Optional[str] = None,
        search_term: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._client = client
        self._user = user
        self._search_term = search_term
    
    def run(self):
        try:
            if self._search_term:
                sessions = self._client.search_sessions(self._search_term)
            else:
                sessions = self._client.list_sessions()
            self.sessions_loaded.emit(sessions)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()


class GetHistoryWorker(QThread):
    """
    Worker thread for retrieving session history.
    
    Signals:
        history_loaded: Emitted with SessionHistory on success.
        not_found: Emitted when session is not found.
        error_occurred: Emitted with error message on failure.
    """
    
    history_loaded = Signal(object)
    not_found = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, client: SessionClient, session_id: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._session_id = session_id
    
    def run(self):
        try:
            history = self._client.get_history(self._session_id)
            
            if history is None:
                self.not_found.emit(self._session_id)
            else:
                self.history_loaded.emit(history)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()


class DeleteSessionWorker(QThread):
    """
    Worker thread for deleting a session.
    
    Signals:
        session_deleted: Emitted with session_id on success.
        error_occurred: Emitted with error message on failure.
    """
    
    session_deleted = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, client: SessionClient, session_id: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._session_id = session_id
    
    def run(self):
        try:
            success = self._client.delete_session(self._session_id)
            
            if success:
                self.session_deleted.emit(self._session_id)
            else:
                self.error_occurred.emit("Failed to delete session")
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()