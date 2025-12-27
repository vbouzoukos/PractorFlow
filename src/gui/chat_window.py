"""
Chat window - Main application window.

Provides the primary chat interface with:
- Message history display
- Message input with file attachment
- Session management (async)
"""

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QStatusBar,
)
from PySide6.QtCore import Slot

from widgets.chat_display import ChatDisplay
from widgets.input_widget import InputWidget
from api.chat_client import ChatClient
from workers.stream_worker import StreamWorker
from workers.session_worker import StartSessionWorker, DeleteSessionWorker


class ChatWindow(QMainWindow):
    """
    Main chat window.
    
    Manages chat sessions and coordinates between UI components
    and the API client. All API calls are async via worker threads.
    """
    
    def __init__(self, api_url: str, parent=None):
        super().__init__(parent)
        
        self._api_url = api_url
        self._client = ChatClient(api_url)
        self._session_id = None
        self._stream_worker = None
        self._session_worker = None
        self._pending_close = False
        
        self._setup_ui()
        self._connect_signals()
        
        # Start a new session on launch
        self._start_new_session()
    
    def _setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("PractorFlow Chat")
        self.setMinimumSize(600, 500)
        self.resize(800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header with session controls
        header_layout = QHBoxLayout()
        
        self._session_label = QLabel("Session: Connecting...")
        header_layout.addWidget(self._session_label)
        
        self._reconnect_btn = QPushButton("Reconnect")
        self._reconnect_btn.setToolTip("Reconnect to server")
        self._reconnect_btn.hide()
        header_layout.addWidget(self._reconnect_btn)
        
        header_layout.addStretch()
        
        self._new_session_btn = QPushButton("New Session")
        self._new_session_btn.setToolTip("Start a new chat session")
        header_layout.addWidget(self._new_session_btn)
        
        layout.addLayout(header_layout)
        
        # Chat display area
        self._chat_display = ChatDisplay()
        layout.addWidget(self._chat_display, stretch=1)
        
        # Input area
        self._input_widget = InputWidget()
        layout.addWidget(self._input_widget)
        
        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(f"API: {self._api_url}")
        
        # Disable input until session is ready
        self._input_widget.set_enabled(False)
    
    def _connect_signals(self):
        """Connect widget signals to slots."""
        self._new_session_btn.clicked.connect(self._on_new_session_clicked)
        self._input_widget.message_submitted.connect(self._on_message_submitted)
        self._reconnect_btn.clicked.connect(self._on_reconnect_clicked)
    
    def _start_new_session(self):
        """Start a new chat session asynchronously."""
        # Update UI state
        self._session_label.setText("Session: Connecting...")
        self._input_widget.set_enabled(False)
        self._new_session_btn.setEnabled(False)
        
        # Clear chat display
        self._chat_display.clear_messages()
        
        # Start worker
        self._session_worker = StartSessionWorker(self._client)
        self._session_worker.session_started.connect(self._on_session_started)
        self._session_worker.error_occurred.connect(self._on_session_error)
        self._session_worker.start()
    
    @Slot(str)
    def _on_session_started(self, session_id: str):
        """Handle successful session creation."""
        self._session_id = session_id
        self._session_label.setText(f"Session: {session_id[:16]}...")
        self._input_widget.set_enabled(True)
        self._new_session_btn.setEnabled(True)
        self._reconnect_btn.hide()
        self._status_bar.showMessage("Session started", 3000)
        self._session_worker = None
    
    @Slot(str)
    def _on_session_error(self, error: str):
        """Handle session creation error."""
        self._session_id = None
        self._session_label.setText("Session: Error")
        self._new_session_btn.setEnabled(True)
        self._reconnect_btn.show()
        self._session_worker = None
        
        QMessageBox.critical(
            self,
            "Connection Error",
            f"Failed to start session:\n{error}\n\nIs the API server running at {self._api_url}?"
        )
    
    def _delete_session_async(self, on_complete=None):
        """Delete current session asynchronously."""
        if not self._session_id:
            if on_complete:
                on_complete()
            return
        
        worker = DeleteSessionWorker(self._client, self._session_id)
        
        if on_complete:
            worker.session_deleted.connect(on_complete)
        
        worker.start()
        
        # Keep reference to prevent garbage collection
        self._delete_worker = worker
    
    @Slot()
    def _on_reconnect_clicked(self):
        """Handle reconnect button click."""
        self._reconnect_btn.hide()
        self._start_new_session()
    
    @Slot()
    def _on_new_session_clicked(self):
        """Handle new session button click."""
        reply = QMessageBox.question(
            self,
            "New Session",
            "Start a new session? Current conversation will be cleared.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Delete old session then start new one
            self._delete_session_async(on_complete=self._start_new_session)
    
    @Slot(str, list)
    def _on_message_submitted(self, message: str, file_paths: list):
        """
        Handle message submission from input widget.
        
        Args:
            message: User message text.
            file_paths: List of file paths to upload.
        """
        if not self._session_id:
            QMessageBox.warning(
                self,
                "No Session",
                "No active session. Please start a new session."
            )
            return
        
        if not message.strip():
            return
        
        # Add user message to display
        self._chat_display.add_user_message(message)
        
        # Show file info if files attached
        if file_paths:
            file_names = [p.split("/")[-1].split("\\")[-1] for p in file_paths]
            self._chat_display.add_system_message(
                f"Attached files: {', '.join(file_names)}"
            )
        
        # Disable input while processing
        self._input_widget.set_enabled(False)
        self._status_bar.showMessage("Generating response...")
        
        # Create placeholder for assistant response
        self._chat_display.add_assistant_message("")
        
        # Start streaming worker
        self._stream_worker = StreamWorker(
            client=self._client,
            session_id=self._session_id,
            message=message,
            file_paths=file_paths
        )
        
        self._stream_worker.chunk_received.connect(self._on_chunk_received)
        self._stream_worker.stream_finished.connect(self._on_stream_finished)
        self._stream_worker.error_occurred.connect(self._on_stream_error)
        
        self._stream_worker.start()
    
    @Slot(str)
    def _on_chunk_received(self, text: str):
        """Handle streaming chunk received."""
        self._chat_display.append_to_last_message(text)
    
    @Slot(dict)
    def _on_stream_finished(self, usage: dict):
        """Handle stream completion."""
        self._input_widget.set_enabled(True)
        self._input_widget.clear_input()
        
        if usage:
            tokens = usage.get("total_tokens", "N/A")
            self._status_bar.showMessage(f"Response complete. Tokens: {tokens}", 5000)
        else:
            self._status_bar.showMessage("Response complete", 3000)
        
        # Finalize the last message (render markdown)
        self._chat_display.finalize_last_message()
        
        self._stream_worker = None
    
    @Slot(str)
    def _on_stream_error(self, error: str):
        """Handle streaming error."""
        self._input_widget.set_enabled(True)
        self._status_bar.showMessage(f"Error: {error}", 5000)
        
        self._chat_display.add_system_message(f"Error: {error}")
        
        self._stream_worker = None
    
    def closeEvent(self, event):
        """Handle window close - cleanup session asynchronously."""
        if self._stream_worker and self._stream_worker.isRunning():
            self._stream_worker.stop()
            self._stream_worker.wait(1000)
        
        # Delete session in background and close
        if self._session_id:
            self._delete_session_async()
        
        event.accept()