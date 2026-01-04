"""
Session worker - Background threads for session management.

Handles session creation and deletion in background threads
to keep the UI responsive.
"""

from PySide6.QtCore import QThread, Signal

from gui.api.chat_client import ChatClient


class StartSessionWorker(QThread):
    """
    Worker thread for starting a chat session.
    
    Signals:
        session_started: Emitted with session_id on success.
        error_occurred: Emitted with error message on failure.
    """
    
    session_started = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, client: ChatClient, parent=None):
        super().__init__(parent)
        self._client = client
    
    def run(self):
        try:
            session_id = self._client.start_session()
            self.session_started.emit(session_id)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()


class DeleteSessionWorker(QThread):
    """
    Worker thread for deleting a chat session.
    
    Signals:
        session_deleted: Emitted on success.
        error_occurred: Emitted with error message on failure.
    """
    
    session_deleted = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, client: ChatClient, session_id: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._session_id = session_id
    
    def run(self):
        try:
            self._client.delete_session(self._session_id)
            self.session_deleted.emit()
        except Exception:
            # Ignore delete errors on cleanup
            self.session_deleted.emit()
    
    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()