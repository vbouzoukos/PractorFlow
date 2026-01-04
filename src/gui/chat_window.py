"""
Chat window - Main application window.

Provides the primary chat interface with:
- Message history display
- Message input with file attachment
- Session management (async)
- Collapsible history panel for past sessions
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
from PySide6.QtCore import Qt, Slot

from gui.widgets.chat_display import ChatDisplay
from gui.widgets.input_widget import InputWidget
from gui.widgets.history_panel import HistoryPanel
from gui.api.chat_client import ChatClient, SessionHistory
from gui.workers.stream_worker import StreamWorker
from gui.workers.session_worker import StartSessionWorker, DeleteSessionWorker


class ChatWindow(QMainWindow):
    """
    Main chat window.
    
    Manages chat sessions and coordinates between UI components
    and the API client. All API calls are async via worker threads.
    """
    
    def __init__(self, api_url: str, parent=None):
        super().__init__(parent)
        
        self._api_url = api_url
        self._client = ChatClient(base_url=api_url, username="practorFlowClient")
        self._session_id = None
        self._stream_worker = None
        self._session_worker = None
        self._delete_worker = None
        self._pending_close = False
        
        self._setup_ui()
        self._connect_signals()
        
        # Load history panel and start a new session
        self._history_panel.refresh_sessions()
        self._start_new_session()
    
    def _setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("PractorFlow Chat")
        self.setMinimumSize(700, 500)
        self.resize(1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        # History panel (collapsible)
        self._history_panel = HistoryPanel(self._client)
        main_layout.addWidget(self._history_panel)
        
        # Chat area container
        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(4, 4, 4, 4)
        chat_layout.setSpacing(8)
        
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
        
        chat_layout.addLayout(header_layout)
        
        # Chat display area
        self._chat_display = ChatDisplay()
        chat_layout.addWidget(self._chat_display, stretch=1)
        
        # Input area
        self._input_widget = InputWidget()
        chat_layout.addWidget(self._input_widget)
        
        main_layout.addWidget(chat_container, stretch=1)
        
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
        
        # History panel signals
        self._history_panel.session_selected.connect(self._on_session_selected)
        self._history_panel.session_deleted.connect(self._on_session_deleted_from_history)
    
    def _start_new_session(self):
        """Start a new chat session asynchronously."""
        try:
            # Update UI state
            self._session_label.setText("Session: Connecting...")
            self._input_widget.set_enabled(False)
            self._new_session_btn.setEnabled(False)
            
            # Clear chat display
            self._chat_display.clear_messages()
            
            # Start worker with parent to prevent premature garbage collection
            self._session_worker = StartSessionWorker(self._client, parent=self)
            self._session_worker.session_started.connect(self._on_session_started)
            self._session_worker.error_occurred.connect(self._on_session_error)
            self._session_worker.finished.connect(self._cleanup_session_worker)
            self._session_worker.start()
        except Exception as e:
            self._session_label.setText("Session: Error")
            self._new_session_btn.setEnabled(True)
            self._status_bar.showMessage(f"Error: {e}", 5000)
    
    @Slot()
    def _cleanup_session_worker(self):
        """Clean up session worker after it finishes."""
        try:
            if self._session_worker:
                self._session_worker.deleteLater()
                self._session_worker = None
        except Exception:
            self._session_worker = None
    
    @Slot(str)
    def _on_session_started(self, session_id: str):
        """Handle successful session creation."""
        try:
            self._session_id = session_id
            self._session_label.setText(f"Session: {session_id[:16]}...")
            self._input_widget.set_enabled(True)
            self._new_session_btn.setEnabled(True)
            self._reconnect_btn.hide()
            self._status_bar.showMessage("Session started", 3000)
            
            # Update history panel current session
            self._history_panel.set_current_session(session_id)
            
            # Refresh history to show the new session
            self._history_panel.refresh_sessions()
        except Exception as e:
            self._status_bar.showMessage(f"Error: {e}", 5000)
    
    @Slot(str)
    def _on_session_error(self, error: str):
        """Handle session creation error."""
        try:
            self._session_id = None
            self._session_label.setText("Session: Error")
            self._new_session_btn.setEnabled(True)
            self._reconnect_btn.show()
            
            QMessageBox.critical(
                self,
                "Connection Error",
                f"Failed to start session:\n{error}\n\nIs the API server running at {self._api_url}?"
            )
        except Exception:
            pass
    
    def _delete_session_async(self, on_complete=None):
        """Delete current session asynchronously."""
        try:
            if not self._session_id:
                if on_complete:
                    on_complete()
                return
            
            self._delete_worker = DeleteSessionWorker(self._client, self._session_id, parent=self)
            
            if on_complete:
                self._delete_worker.session_deleted.connect(on_complete)
            
            self._delete_worker.finished.connect(self._cleanup_delete_worker)
            self._delete_worker.start()
        except Exception:
            if on_complete:
                on_complete()
    
    @Slot()
    def _cleanup_delete_worker(self):
        """Clean up delete worker after it finishes."""
        try:
            if self._delete_worker:
                self._delete_worker.deleteLater()
                self._delete_worker = None
        except Exception:
            self._delete_worker = None
    
    @Slot()
    def _on_reconnect_clicked(self):
        """Handle reconnect button click."""
        try:
            self._reconnect_btn.hide()
            self._start_new_session()
        except Exception:
            pass
    
    @Slot()
    def _on_new_session_clicked(self):
        """Handle new session button click."""
        try:
            reply = QMessageBox.question(
                self,
                "New Session",
                "Start a new session? Current conversation will be cleared.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Start new session (don't delete old one - it stays in history)
                self._start_new_session()
        except Exception:
            pass
    
    @Slot(object)
    def _on_session_selected(self, history: SessionHistory):
        """Handle session selected from history panel."""
        try:
            # Switch to the selected session
            self._session_id = history.session_id
            self._session_label.setText(f"Session: {history.session_id[:16]}...")
            
            # Clear and reload chat display with history
            self._chat_display.clear_messages()
            
            for msg in history.messages:
                if msg.role == "user":
                    self._chat_display.add_user_message(msg.content)
                elif msg.role == "assistant":
                    self._chat_display.add_assistant_message(msg.content)
                    self._chat_display.finalize_last_message()
                elif msg.role == "system":
                    self._chat_display.add_system_message(msg.content)
            
            # Enable input
            self._input_widget.set_enabled(True)
            self._input_widget.clear_input()
            
            self._status_bar.showMessage(f"Loaded session with {len(history.messages)} messages", 3000)
        except Exception as e:
            self._status_bar.showMessage(f"Error loading session: {e}", 5000)
    
    @Slot(str)
    def _on_session_deleted_from_history(self, session_id: str):
        """Handle session deleted from history panel."""
        try:
            # If the deleted session is the current one, start a new session
            if session_id == self._session_id:
                self._start_new_session()
        except Exception:
            pass
    
    @Slot(str, list)
    def _on_message_submitted(self, message: str, file_paths: list):
        """
        Handle message submission from input widget.
        
        Args:
            message: User message text.
            file_paths: List of file paths to upload.
        """
        try:
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
            
            # Start streaming worker with parent
            self._stream_worker = StreamWorker(
                client=self._client,
                session_id=self._session_id,
                message=message,
                file_paths=file_paths,
                parent=self
            )
            
            self._stream_worker.chunk_received.connect(self._on_chunk_received)
            self._stream_worker.stream_finished.connect(self._on_stream_finished)
            self._stream_worker.error_occurred.connect(self._on_stream_error)
            self._stream_worker.finished.connect(self._cleanup_stream_worker)
            
            self._stream_worker.start()
        except Exception as e:
            self._input_widget.set_enabled(True)
            self._status_bar.showMessage(f"Error: {e}", 5000)
    
    @Slot()
    def _cleanup_stream_worker(self):
        """Clean up stream worker after it finishes."""
        try:
            if self._stream_worker:
                self._stream_worker.deleteLater()
                self._stream_worker = None
        except Exception:
            self._stream_worker = None
    
    @Slot(str)
    def _on_chunk_received(self, text: str):
        """Handle streaming chunk received."""
        try:
            self._chat_display.append_to_last_message(text)
        except Exception:
            pass
    
    @Slot(dict)
    def _on_stream_finished(self, usage: dict):
        """Handle stream completion."""
        try:
            self._input_widget.set_enabled(True)
            self._input_widget.clear_input()
            
            if usage:
                tokens = usage.get("total_tokens", "N/A")
                self._status_bar.showMessage(f"Response complete. Tokens: {tokens}", 5000)
            else:
                self._status_bar.showMessage("Response complete", 3000)
            
            # Finalize the last message (render markdown)
            self._chat_display.finalize_last_message()
            
            # Refresh history to update message counts
            self._history_panel.refresh_sessions()
        except Exception:
            pass
    
    @Slot(str)
    def _on_stream_error(self, error: str):
        """Handle streaming error."""
        try:
            self._input_widget.set_enabled(True)
            self._status_bar.showMessage(f"Error: {error}", 5000)
            
            self._chat_display.add_system_message(f"Error: {error}")
        except Exception:
            pass
    
    def closeEvent(self, event):
        """Handle window close - cleanup all workers."""
        try:
            # Stop and wait for stream worker
            if self._stream_worker:
                if self._stream_worker.isRunning():
                    self._stream_worker.stop()
                    self._stream_worker.wait(2000)
                self._stream_worker.deleteLater()
                self._stream_worker = None
            
            # Wait for session worker
            if self._session_worker:
                if self._session_worker.isRunning():
                    self._session_worker.wait(2000)
                self._session_worker.deleteLater()
                self._session_worker = None
            
            # Wait for delete worker
            if self._delete_worker:
                if self._delete_worker.isRunning():
                    self._delete_worker.wait(2000)
                self._delete_worker.deleteLater()
                self._delete_worker = None
            
            # Shutdown history panel workers
            self._history_panel.shutdown()
            
        except Exception:
            pass
        
        # Always accept close event
        event.accept()