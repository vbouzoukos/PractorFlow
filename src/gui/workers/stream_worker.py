"""
Stream worker - Background thread for SSE streaming.

Handles streaming API responses in a separate thread to keep
the UI responsive during generation.
"""

from typing import Optional

from PySide6.QtCore import QThread, Signal

from gui.api.chat_client import ChatClient


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
    
    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self._stop_requested = True
            self.wait(2000)
        self.deleteLater()

class ChatEditResumeWorker(QThread):
    """
    Worker for chat edit → truncate → resume flow.

    This is ADDITIVE functionality:
    - Does NOT modify StreamWorker
    - Reuses ChatClient.truncate_messages
    - Reuses StreamWorker for streaming execution
    """

    error_occurred = Signal(str)
    stream_finished = Signal(dict)
    chunk_received = Signal(str)

    def __init__(
        self,
        client: ChatClient,
        session_id: str,
        from_index: int,
        updated_message: str,
        file_paths: Optional[list] = None,
        parent=None,
    ):
        super().__init__(parent)

        self._client = client
        self._session_id = session_id
        self._from_index = from_index
        self._updated_message = updated_message
        self._file_paths = file_paths or []

        self._stream_worker: Optional[StreamWorker] = None

    def run(self):
        try:
            # 1. Truncate backend history
            result = self._client.truncate_messages(
                self._session_id,
                self._from_index,
            )

            if result is None:
                self.error_occurred.emit("Session not found during truncate")
                return

            # 2. Resume streaming using EXISTING StreamWorker
            self._stream_worker = StreamWorker(
                client=self._client,
                session_id=self._session_id,
                message=self._updated_message,
                file_paths=self._file_paths,
            )

            self._stream_worker.chunk_received.connect(self.chunk_received)
            self._stream_worker.stream_finished.connect(self.stream_finished)
            self._stream_worker.error_occurred.connect(self.error_occurred)

            self._stream_worker.start()
            self._stream_worker.wait()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker and child worker."""
        if self._stream_worker:
            self._stream_worker.stop()
            self._stream_worker.safe_delete()

        if self.isRunning():
            self.wait(2000)

        self.deleteLater()
