"""
Chat window - Chat view widget.

Provides the primary chat interface with:
- Message history display
- Message input with file attachment
- Session management (async)
- Collapsible history panel for past sessions
- Foldable documents panel for session documents
- Agent mode toggle for multi-agent task execution
- Manual connection via Connect button
- Reads settings on activation
"""

from PySide6.QtWidgets import QWidget, QMessageBox
from PySide6.QtCore import Slot

from gui.logger import get_logger

logger = get_logger(
    "practorflow-client", level="INFO", log_file="logs/practorflow-client.log"
)

from gui.settings.settings import AppSettings, load_settings
from gui.api.chat_client import ChatClient
from gui.api.agent_client import AgentClient, AgentTaskResult
from gui.api.session_client import SessionClient
from gui.api.client_data import SessionHistory
from gui.chat.chat_ui import ChatUI
from gui.chat.worker_manager import WorkerManager


class ChatWindow(QWidget):
    """
    Chat window widget.

    Manages chat sessions and coordinates between UI components
    and API clients. All API calls are async via worker threads.
    Supports both chat mode (streaming) and agent mode (task execution).
    Reads settings from settings.json on each activation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        try:
            self._settings: AppSettings = None
            self._client: ChatClient = None
            self._agent_client: AgentClient = None
            self._session_client: SessionClient = None
            self._session_id: str = None
            self._agent_mode: bool = False
            self._connected: bool = False

            # Load initial settings
            self._load_settings()

            # Initialize UI
            self._ui = ChatUI(self, self._session_client)

            # Initialize worker manager
            self._workers = WorkerManager(self)

            # Connect signals
            self._connect_signals()

            logger.info("ChatWindow initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ChatWindow: {e}")
            raise

    def _load_settings(self):
        """Load settings and initialize/update clients."""
        try:
            self._settings = load_settings()

            api_url = self._settings.api_url
            username = self._settings.username

            self._client = ChatClient(base_url=api_url, username=username)
            self._agent_client = AgentClient(base_url=api_url, username=username)
            self._session_client = SessionClient(base_url=api_url, username=username)

            logger.info(f"Settings loaded: api_url={api_url}, username={username}")

        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            raise

    def _connect_signals(self):
        """Connect widget signals to slots."""
        try:
            self._ui.connect_btn.clicked.connect(self._on_connect_clicked)
            self._ui.reconnect_btn.clicked.connect(self._on_reconnect_clicked)
            self._ui.new_session_btn.clicked.connect(self._on_new_session_clicked)
            self._ui.documents_btn.clicked.connect(self._on_documents_btn_clicked)
            self._ui.agent_mode_checkbox.toggled.connect(self._on_agent_mode_toggled)
            self._ui.input_widget.message_submitted.connect(self._on_message_submitted)

            # History panel signals
            self._ui.history_panel.session_selected.connect(self._on_session_selected)
            self._ui.history_panel.session_deleted.connect(self._on_session_deleted)

            # Documents panel signals
            self._ui.documents_panel.document_deleted.connect(self._on_document_deleted)

            # Chat display signals
            self._ui.chat_display.message_edit_requested.connect(
                self._on_message_edit_requested
            )

            logger.debug("Signals connected")

        except Exception as e:
            logger.error(f"Failed to connect signals: {e}")
            raise

    def showEvent(self, event):
        """Handle widget show - reload settings."""
        super().showEvent(event)
        try:
            old_url = self._settings.api_url if self._settings else None
            old_username = self._settings.username if self._settings else None

            self._load_settings()

            # Update session client for panels if settings changed
            if (
                old_url != self._settings.api_url
                or old_username != self._settings.username
            ):
                self._ui.history_panel.set_client(self._session_client)
                self._ui.documents_panel.set_client(self._session_client)

                # Reset connection if settings changed
                if self._connected:
                    self._connected = False
                    self._session_id = None
                    self._ui.set_connected_state(False)
                    self._ui.session_label.setText(
                        "Session: Settings changed - reconnect"
                    )
                    logger.info("Settings changed, connection reset")

            # Refresh history panel only if connected
            if self._connected:
                self._ui.history_panel.refresh_sessions()

        except Exception as e:
            logger.error(f"Error in showEvent: {e}")

    # --- Connection ---

    @Slot()
    def _on_connect_clicked(self):
        """Handle connect button click."""
        try:
            logger.info("Connect clicked")
            self._start_new_session()
        except Exception as e:
            logger.error(f"Error on connect: {e}")

    @Slot()
    def _on_reconnect_clicked(self):
        """Handle reconnect button click."""
        try:
            logger.info("Reconnect clicked")
            self._ui.reconnect_btn.hide()
            self._start_new_session()
        except Exception as e:
            logger.error(f"Error on reconnect: {e}")

    def _start_new_session(self):
        """Start a new chat or agent session asynchronously."""
        try:
            self._ui.session_label.setText("Session: Connecting...")
            self._ui.input_widget.set_enabled(False)
            self._ui.new_session_btn.setEnabled(False)
            self._ui.connect_btn.setEnabled(False)

            self._ui.chat_display.clear_messages()
            self._ui.documents_panel.collapse()
            self._ui.documents_panel.set_session(None)

            worker = self._workers.create_session_worker(
                client=self._client,
                agent_mode=self._agent_mode,
                agent_client=self._agent_client,
            )

            worker.session_started.connect(self._on_session_started)
            worker.error_occurred.connect(self._on_session_error)
            worker.finished.connect(self._workers.cleanup_session_worker)
            worker.start()

            logger.debug("Session worker started")

        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            self._ui.session_label.setText("Session: Error")
            self._ui.new_session_btn.setEnabled(True)
            self._ui.connect_btn.setEnabled(True)

    @Slot(str)
    def _on_session_started(self, session_id: str):
        """Handle successful session creation."""
        try:
            self._session_id = session_id
            self._connected = True

            short_id = session_id[:8] if len(session_id) > 8 else session_id
            self._ui.session_label.setText(f"Session: {short_id}...")
            self._ui.set_connected_state(True)
            self._ui.new_session_btn.setEnabled(True)
            self._ui.connect_btn.hide()

            self._ui.documents_panel.set_session(session_id)
            self._ui.history_panel.refresh_sessions()

            logger.info(f"Session started: {session_id}")

        except Exception as e:
            logger.error(f"Error handling session start: {e}")

    @Slot(str)
    def _on_session_error(self, error: str):
        """Handle session creation error."""
        try:
            self._connected = False
            self._ui.session_label.setText("Session: Connection failed")
            self._ui.new_session_btn.setEnabled(True)
            self._ui.connect_btn.setEnabled(True)
            self._ui.reconnect_btn.show()

            logger.error(f"Session error: {error}")
            QMessageBox.warning(
                self, "Connection Error", f"Failed to connect:\n{error}"
            )

        except Exception as e:
            logger.error(f"Error handling session error: {e}")

    # --- New Session ---

    @Slot()
    def _on_new_session_clicked(self):
        """Handle new session button click."""
        try:
            reply = QMessageBox.question(
                self,
                "New Session",
                "Start a new session? Current conversation will be cleared.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                self._start_new_session()

        except Exception as e:
            logger.error(f"Error on new session: {e}")

    # --- Agent Mode ---

    @Slot(bool)
    def _on_agent_mode_toggled(self, checked: bool):
        """Handle agent mode toggle."""
        try:
            self._agent_mode = checked
            mode_name = "Agent" if checked else "Chat"
            logger.info(f"{mode_name} mode enabled")
        except Exception as e:
            logger.error(f"Error toggling agent mode: {e}")

    # --- Message Handling ---

    @Slot(str, list)
    def _on_message_submitted(self, message: str, file_paths: list):
        """Handle message submission from input widget."""
        try:
            self._ui.set_busy_state(True)

            if not self._session_id:
                self._ui.set_busy_state(False)
                QMessageBox.warning(self, "No Session", "Please connect first.")
                return

            if not message.strip():
                self._ui.set_busy_state(False)
                return

            self._ui.chat_display.add_user_message(message)

            if file_paths:
                file_names = [p.split("/")[-1].split("\\")[-1] for p in file_paths]
                self._ui.chat_display.add_system_message(
                    f"Attached files: {', '.join(file_names)}"
                )

            if self._agent_mode:
                self._execute_agent_task(message, file_paths)
            else:
                self._execute_chat_stream(message, file_paths)

            logger.debug(f"Message submitted: {message[:50]}...")

        except Exception as e:
            self._ui.set_busy_state(False)
            logger.error(f"Error submitting message: {e}")

    def _execute_chat_stream(self, message: str, file_paths: list):
        """Execute chat streaming request."""
        try:
            self._ui.chat_display.add_assistant_message("")

            worker = self._workers.create_stream_worker(
                client=self._client,
                session_id=self._session_id,
                message=message,
                file_paths=file_paths,
            )

            worker.chunk_received.connect(self._on_chunk_received)
            worker.stream_finished.connect(self._on_stream_finished)
            worker.error_occurred.connect(self._on_stream_error)
            worker.finished.connect(self._on_stream_worker_finished)
            worker.start()

        except Exception as e:
            self._ui.set_busy_state(False)
            logger.error(f"Error executing chat stream: {e}")

    def _execute_agent_task(self, task: str, file_paths: list):
        """Execute agent task request."""
        try:
            worker = self._workers.create_agent_worker(
                client=self._agent_client,
                session_id=self._session_id,
                task=task,
                file_paths=file_paths,
            )

            worker.task_completed.connect(self._on_agent_task_completed)
            worker.error_occurred.connect(self._on_agent_task_error)
            worker.finished.connect(self._on_agent_worker_finished)
            worker.start()

        except Exception as e:
            self._ui.set_busy_state(False)
            logger.error(f"Error executing agent task: {e}")

    @Slot(str)
    def _on_chunk_received(self, text: str):
        """Handle streaming chunk received."""
        try:
            self._ui.chat_display.append_to_last_message(text)
        except Exception as e:
            logger.error(f"Error handling chunk: {e}")

    @Slot(dict)
    def _on_stream_finished(self, usage: dict):
        """Handle stream completion."""
        try:
            self._ui.input_widget.clear_input()
            self._ui.chat_display.finalize_last_message()
            self._ui.history_panel.refresh_sessions()

            if self._ui.documents_panel.is_expanded():
                self._ui.documents_panel.refresh_documents()

            if usage:
                tokens = usage.get("total_tokens", "N/A")
                logger.debug(f"Stream finished, tokens: {tokens}")

        except Exception as e:
            logger.error(f"Error handling stream finish: {e}")

    @Slot(str)
    def _on_stream_error(self, error: str):
        """Handle streaming error."""
        try:
            self._ui.chat_display.add_system_message(f"Error: {error}")
            logger.error(f"Stream error: {error}")
        except Exception as e:
            logger.error(f"Error handling stream error: {e}")

    @Slot()
    def _on_stream_worker_finished(self):
        """Handle stream worker finished."""
        self._ui.set_busy_state(False)
        self._workers.cleanup_stream_worker()

    @Slot(object)
    def _on_agent_task_completed(self, result: AgentTaskResult):
        """Handle agent task completion."""
        try:
            self._ui.input_widget.clear_input()

            if result.success:
                self._ui.chat_display.add_assistant_message(
                    result.output or "Task completed."
                )
                self._ui.chat_display.finalize_last_message()
                logger.info("Agent task completed successfully")
            else:
                error_msg = result.error or "Task failed"
                self._ui.chat_display.add_system_message(f"Agent error: {error_msg}")
                logger.warning(f"Agent task failed: {error_msg}")

            self._ui.history_panel.refresh_sessions()

            if self._ui.documents_panel.is_expanded():
                self._ui.documents_panel.refresh_documents()

        except Exception as e:
            logger.error(f"Error handling agent completion: {e}")

    @Slot(str)
    def _on_agent_task_error(self, error: str):
        """Handle agent task error."""
        try:
            self._ui.chat_display.add_system_message(f"Agent error: {error}")
            logger.error(f"Agent task error: {error}")
        except Exception as e:
            logger.error(f"Error handling agent error: {e}")

    @Slot()
    def _on_agent_worker_finished(self):
        """Handle agent worker finished."""
        self._ui.set_busy_state(False)
        self._workers.cleanup_agent_worker()

    # --- Message Edit ---

    @Slot(int, str)
    def _on_message_edit_requested(self, index: int, new_content: str):
        """Handle message edit request from chat display."""
        try:
            if not self._session_id:
                QMessageBox.warning(self, "No Session", "No active session.")
                return

            self._ui.set_busy_state(True)
            self._ui.chat_display.truncate_from_index(index)
            self._ui.chat_display.add_user_message(new_content)

            if self._agent_mode:
                worker = self._workers.create_agent_edit_worker(
                    client=self._agent_client,
                    session_client=self._session_client,
                    session_id=self._session_id,
                    from_index=index,
                    updated_task=new_content,
                    file_paths=[],
                )
                worker.completed.connect(self._on_agent_task_completed)
                worker.error_occurred.connect(self._on_edit_resume_error)
            else:
                self._ui.chat_display.add_assistant_message("")
                worker = self._workers.create_chat_edit_worker(
                    client=self._client,
                    session_client=self._session_client,
                    session_id=self._session_id,
                    from_index=index,
                    updated_message=new_content,
                    file_paths=[],
                )
                worker.chunk_received.connect(self._on_chunk_received)
                worker.stream_finished.connect(self._on_stream_finished)
                worker.error_occurred.connect(self._on_edit_resume_error)

            worker.finished.connect(self._on_edit_worker_finished)
            worker.start()

            logger.debug(f"Edit requested at index {index}")

        except Exception as e:
            self._ui.set_busy_state(False)
            logger.error(f"Error handling edit request: {e}")

    @Slot(str)
    def _on_edit_resume_error(self, error: str):
        """Handle edit-resume error."""
        try:
            self._ui.chat_display.add_system_message(f"Edit failed: {error}")
            logger.error(f"Edit resume error: {error}")
        except Exception as e:
            logger.error(f"Error handling edit error: {e}")

    @Slot()
    def _on_edit_worker_finished(self):
        """Handle edit worker finished."""
        self._ui.set_busy_state(False)
        self._workers.cleanup_edit_resume_worker()

    # --- History Panel ---

    @Slot(object)
    def _on_session_selected(self, history: SessionHistory):
        """Handle session selected from history panel."""
        try:
            self._session_id = history.session_id
            self._connected = True

            short_id = history.session_id[:8]
            self._ui.session_label.setText(f"Session: {short_id}...")
            self._ui.set_connected_state(True)
            self._ui.connect_btn.hide()

            # Load messages into display
            self._ui.chat_display.clear_messages()
            for msg in history.messages:
                if msg.role == "user":
                    self._ui.chat_display.add_user_message(msg.content)
                elif msg.role == "assistant":
                    self._ui.chat_display.add_assistant_message(msg.content)
                    self._ui.chat_display.finalize_last_message()
                elif msg.role == "system":
                    self._ui.chat_display.add_system_message(msg.content)

            self._ui.documents_panel.set_session(history.session_id)

            logger.info(f"Session selected: {history.session_id}")

        except Exception as e:
            logger.error(f"Error selecting session: {e}")

    @Slot(str)
    def _on_session_deleted(self, session_id: str):
        """Handle session deleted from history panel."""
        try:
            if self._session_id == session_id:
                self._session_id = None
                self._connected = False
                self._ui.session_label.setText("Session: Deleted")
                self._ui.set_connected_state(False)
                self._ui.chat_display.clear_messages()
                self._ui.documents_panel.set_session(None)

            logger.info(f"Session deleted: {session_id}")

        except Exception as e:
            logger.error(f"Error handling session delete: {e}")

    # --- Documents Panel ---

    @Slot()
    def _on_documents_btn_clicked(self):
        """Handle documents button click."""
        try:
            self._ui.documents_panel.toggle_expanded()
        except Exception as e:
            logger.error(f"Error toggling documents: {e}")

    @Slot(str)
    def _on_document_deleted(self, document_id: str):
        """Handle document deleted from documents panel."""
        logger.debug(f"Document deleted: {document_id}")

    # --- Cleanup ---

    def shutdown(self):
        """Shutdown all workers and panels."""
        try:
            logger.info("ChatWindow shutting down")
            self._workers.shutdown()
            self._ui.shutdown()
            logger.info("ChatWindow shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
