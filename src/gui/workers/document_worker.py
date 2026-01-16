"""
Document worker - Background threads for document operations.

Handles document listing and deletion in background threads
to keep the UI responsive.
"""

from PySide6.QtCore import QThread, Signal

from gui.api.session_client import SessionClient


class ListDocumentsWorker(QThread):
    """
    Worker thread for listing session documents.
    
    Signals:
        documents_loaded: Emitted with list of DocumentInfo on success.
        error_occurred: Emitted with error message on failure.
    """
    
    documents_loaded = Signal(list)
    error_occurred = Signal(str)
    
    def __init__(self, client: SessionClient, session_id: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._session_id = session_id
    
    def run(self):
        try:
            documents = self._client.list_session_documents(self._session_id)
            
            if documents is None:
                self.error_occurred.emit("Session not found")
            else:
                self.documents_loaded.emit(documents)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()


class DeleteDocumentWorker(QThread):
    """
    Worker thread for deleting a session document.
    
    Signals:
        document_deleted: Emitted with document_id on success.
        error_occurred: Emitted with error message on failure.
    """
    
    document_deleted = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, client: SessionClient, session_id: str, document_id: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._session_id = session_id
        self._document_id = document_id
    
    def run(self):
        try:
            result = self._client.delete_session_document(self._session_id, self._document_id)
            
            if result is None:
                self.error_occurred.emit("Session or document not found")
            elif result:
                self.document_deleted.emit(self._document_id)
            else:
                self.error_occurred.emit("Failed to delete document")
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()