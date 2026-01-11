"""
Chat window - Main application window.

Provides the primary chat interface with:
- Message history display
- Message input with file attachment
- Session management (async)
- Collapsible history panel for past sessions
- Foldable documents panel for session documents
- Agent mode toggle for multi-agent task execution
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
    QSplitter,
    QCheckBox,
)
from PySide6.QtCore import Qt, Slot, QSize
from PySide6.QtGui import QIcon

from gui.widgets.chat_display import ChatDisplay
from gui.widgets.input_widget import InputWidget
from gui.widgets.history_panel import HistoryPanel
from gui.widgets.documents_panel import DocumentsPanel
from gui.api.chat_client import ChatClient, SessionHistory
from gui.api.agent_client import AgentClient, AgentTaskResult
from gui.workers.stream_worker import StreamWorker
from gui.workers.session_worker import StartSessionWorker, DeleteSessionWorker
from gui.workers.agent_worker import AgentTaskWorker, StartAgentSessionWorker


class ChatWindow(QMainWindow):
    """
    Main chat window.
    
    Manages chat sessions and coordinates between UI components
    and the API client. All API calls are async via worker threads.
    Supports both chat mode (streaming) and agent mode (task execution).
    """
    
    def __init__(self, api_url: str, parent=None):
        super().__init__(parent)
        
        self._api_url = api_url
        self._client = ChatClient(base_url=api_url, username="practorFlowClient")
        self._agent_client = AgentClient(base_url=api_url, username="practorFlowClient")
        self._session_id = None
        self._agent_mode = False
        self._stream_worker = None
        self._agent_worker = None
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
        # Reconnect
        self._reconnect_btn = QPushButton("Reconnect")
        self._reconnect_btn.setToolTip("Reconnect to server")
        self._reconnect_btn.hide()
        header_layout.addWidget(self._reconnect_btn)

        # Agent mode toggle
        self._agent_mode_checkbox = QCheckBox("Agent Mode")
        self._agent_mode_checkbox.setToolTip(
            "Enable multi-agent task execution (plan → execute → verify)"
        )
        header_layout.addWidget(self._agent_mode_checkbox)
        
        # Documents button (icon)
        self._documents_btn = QPushButton()
        self._documents_btn.setFixedSize(28, 28)
        self._documents_btn.setToolTip("Show session documents")
        self._documents_btn.setCursor(Qt.PointingHandCursor)
        
        # Try to use document icon, fallback to emoji
        doc_icon = QIcon.fromTheme("folder-documents")
        if doc_icon.isNull():
            doc_icon = QIcon.fromTheme("document-multiple")
        if doc_icon.isNull():
            self._documents_btn.setText("📄")
        else:
            self._documents_btn.setIcon(doc_icon)
            self._documents_btn.setIconSize(QSize(18, 18))
        
        header_layout.addWidget(self._documents_btn)
        
        header_layout.addStretch()
        
        self._new_session_btn = QPushButton("New Session")
        self._new_session_btn.setToolTip("Start a new chat session")
        header_layout.addWidget(self._new_session_btn)
        
        chat_layout.addLayout(header_layout)
        
        # Documents panel (foldable, below header)
        self._documents_panel = DocumentsPanel(self._client)
        chat_layout.addWidget(self._documents_panel)
        
        # Create a vertical splitter so the user can resize widgets vertically
        splitter = QSplitter(Qt.Vertical)

        # Create the chat message display (top area)
        self._chat_display = ChatDisplay()

        # Create the input widget (bottom area)
        self._input_widget = InputWidget()

        # Add chat display as the first (top) splitter pane
        splitter.addWidget(self._chat_display)

        # Add input widget as the second (bottom) splitter pane
        splitter.addWidget(self._input_widget)

        # Set initial splitter sizes:
        # - first value = chat area (larger)
        # - second value = input area (~100px relative size)
        # These values are relative weights, not exact pixels
        splitter.setSizes([500, 100])

        # Add the splitter to the chat layout so it becomes visible
        chat_layout.addWidget(splitter)

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
        self._documents_btn.clicked.connect(self._on_documents_btn_clicked)
        self._agent_mode_checkbox.toggled.connect(self._on_agent_mode_toggled)
        
        # History panel signals
        self._history_panel.session_selected.connect(self._on_session_selected)
        self._history_panel.session_deleted.connect(self._on_session_deleted_from_history)
        
        # Documents panel signals
        self._documents_panel.document_deleted.connect(self._on_document_deleted)
    
    @Slot(bool)
    def _on_agent_mode_toggled(self, checked: bool):
        """Handle agent mode toggle."""
        self._agent_mode = checked
        mode_name = "Agent" if checked else "Chat"
        self._status_bar.showMessage(f"{mode_name} mode enabled", 3000)
        
        # Start new session when mode changes
        self._start_new_session()
    
    def _start_new_session(self):
        """Start a new chat or agent session asynchronously."""
        try:
            # Update UI state
            self._session_label.setText("Session: Connecting...")
            self._input_widget.set_enabled(False)
            self._new_session_btn.setEnabled(False)
            
            # Clear chat display
            self._chat_display.clear_messages()
            
            # Collapse documents panel and clear session
            self._documents_panel.collapse()
            self._documents_panel.set_session(None)
            
            if self._agent_mode:
                # Start agent session
                self._session_worker = StartAgentSessionWorker(self._agent_client, parent=self)
            else:
                # Start chat session
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
            mode_prefix = "Agent" if self._agent_mode else "Session"
            self._session_label.setText(f"{mode_prefix}: {session_id[:16]}...")
            self._input_widget.set_enabled(True)
            self._new_session_btn.setEnabled(True)
            self._reconnect_btn.hide()
            self._status_bar.showMessage("Session started", 3000)
            
            # Update history panel current session
            self._history_panel.set_current_session(session_id)
            
            # Update documents panel with new session
            self._documents_panel.set_session(session_id)
            
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
    
    @Slot()
    def _on_documents_btn_clicked(self):
        """Handle documents button click - toggle documents panel."""
        try:
            self._documents_panel.toggle_expanded()
        except Exception:
            pass
    
    @Slot(str)
    def _on_document_deleted(self, document_id: str):
        """Handle document deleted from documents panel."""
        try:
            self._status_bar.showMessage("Document deleted", 3000)
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
            
            # Update documents panel with selected session
            self._documents_panel.set_session(history.session_id)
            
            # Collapse documents panel when switching sessions
            self._documents_panel.collapse()
            
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
            
            if self._agent_mode:
                self._execute_agent_task(message, file_paths)
            else:
                self._execute_chat_stream(message, file_paths)
                
        except Exception as e:
            self._input_widget.set_enabled(True)
            self._status_bar.showMessage(f"Error: {e}", 5000)
    
    def _execute_chat_stream(self, message: str, file_paths: list):
        """Execute chat streaming request."""
        self._status_bar.showMessage("Generating response...")
        
        # Create placeholder for assistant response
        self._chat_display.add_assistant_message("")
        
        # Start streaming worker
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
    
    def _execute_agent_task(self, task: str, file_paths: list):
        """Execute agent task request."""
        self._status_bar.showMessage("Executing agent task...")
        
        # Start agent worker
        self._agent_worker = AgentTaskWorker(
            client=self._agent_client,
            session_id=self._session_id,
            task=task,
            file_paths=file_paths,
            parent=self
        )
        
        self._agent_worker.task_completed.connect(self._on_agent_task_completed)
        self._agent_worker.error_occurred.connect(self._on_agent_task_error)
        self._agent_worker.finished.connect(self._cleanup_agent_worker)
        
        self._agent_worker.start()
    
    @Slot()
    def _cleanup_stream_worker(self):
        """Clean up stream worker after it finishes."""
        try:
            if self._stream_worker:
                self._stream_worker.deleteLater()
                self._stream_worker = None
        except Exception:
            self._stream_worker = None
    
    @Slot()
    def _cleanup_agent_worker(self):
        """Clean up agent worker after it finishes."""
        try:
            if self._agent_worker:
                self._agent_worker.deleteLater()
                self._agent_worker = None
        except Exception:
            self._agent_worker = None
    
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
            
            # Refresh documents panel if expanded (new files may have been added)
            if self._documents_panel.is_expanded():
                self._documents_panel.refresh_documents()
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
    
    @Slot(object)
    def _on_agent_task_completed(self, result: AgentTaskResult):
        """Handle agent task completion."""
        try:
            self._input_widget.set_enabled(True)
            self._input_widget.clear_input()
            
            if result.success:
                # Add assistant message with output
                self._chat_display.add_assistant_message(result.output or "Task completed.")
                self._chat_display.finalize_last_message()
                self._status_bar.showMessage("Agent task completed", 3000)
            else:
                # Show error
                error_msg = result.error or "Task failed"
                self._chat_display.add_system_message(f"Agent error: {error_msg}")
                self._status_bar.showMessage("Agent task failed", 5000)
            
            # Refresh history to update message counts
            self._history_panel.refresh_sessions()
            
            # Refresh documents panel if expanded
            if self._documents_panel.is_expanded():
                self._documents_panel.refresh_documents()
        except Exception:
            pass
    
    @Slot(str)
    def _on_agent_task_error(self, error: str):
        """Handle agent task error."""
        try:
            self._input_widget.set_enabled(True)
            self._status_bar.showMessage(f"Error: {error}", 5000)
            
            self._chat_display.add_system_message(f"Agent error: {error}")
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
            
            # Wait for agent worker
            if self._agent_worker:
                if self._agent_worker.isRunning():
                    self._agent_worker.wait(5000)
                self._agent_worker.deleteLater()
                self._agent_worker = None
            
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
            
            # Shutdown documents panel workers
            self._documents_panel.shutdown()
            
        except Exception:
            pass
        
        # Always accept close event
        event.accept()