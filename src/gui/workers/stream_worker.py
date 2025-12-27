"""
Stream worker - Background thread for SSE streaming.

Handles streaming API responses in a separate thread to keep
the UI responsive during generation.
"""

from typing import Optional

from PySide6.QtCore import QThread, Signal

from api.chat_client import ChatClient


class StreamWorker(QThread):
    """
    Worker thread for streaming chat responses.
    
    Runs the SSE streaming in a background thread and emits
    signals for each chunk received.
    
    Signals:
        chunk_received: Emitted for each text chunk (str).
        stream_finished: Emitted when streaming completes (dict with usage).
        error_occurred: Emitted on error (str with error message).
    """
    
    chunk_received = Signal(str)
    stream_finished = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(
        self,
        client: ChatClient,
        session_id: str,
        message: str,
        file_paths: Optional[list] = None,
        parent=None
    ):
        """
        Initialize the stream worker.
        
        Args:
            client: ChatClient instance.
            session_id: Session ID for the chat.
            message: User message to send.
            file_paths: Optional list of file paths to upload.
            parent: Parent QObject.
        """
        super().__init__(parent)
        
        self._client = client
        self._session_id = session_id
        self._message = message
        self._file_paths = file_paths or []
        self._stop_requested = False
    
    def run(self):
        """Execute the streaming request."""
        try:
            usage = None
            
            for chunk in self._client.send_message_stream(
                session_id=self._session_id,
                message=self._message,
                file_paths=self._file_paths
            ):
                if self._stop_requested:
                    break
                
                if chunk.error:
                    self.error_occurred.emit(chunk.error)
                    return
                
                if chunk.text:
                    self.chunk_received.emit(chunk.text)
                
                if chunk.finished:
                    usage = chunk.usage or {}
                    break
            
            self.stream_finished.emit(usage or {})
        
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """Request the worker to stop."""
        self._stop_requested = True
