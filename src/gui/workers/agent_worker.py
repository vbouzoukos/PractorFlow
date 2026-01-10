"""
Agent worker - Background thread for agent task execution.

Handles agent API requests in a separate thread to keep
the UI responsive during task execution.
"""

from typing import Optional

from PySide6.QtCore import QThread, Signal

from gui.api.agent_client import AgentClient, AgentTaskResult


class AgentTaskWorker(QThread):
    """
    Worker thread for executing agent tasks.
    
    Runs the agent task in a background thread and emits
    signal when complete.
    
    Signals:
        task_completed: Emitted with AgentTaskResult on completion.
        error_occurred: Emitted with error message on failure.
    """
    
    task_completed = Signal(object)
    error_occurred = Signal(str)
    
    def __init__(
        self,
        client: AgentClient,
        session_id: str,
        task: str,
        file_paths: Optional[list] = None,
        parent=None
    ):
        """
        Initialize the agent task worker.
        
        Args:
            client: AgentClient instance.
            session_id: Session ID for the task.
            task: Task description to execute.
            file_paths: Optional list of file paths to upload.
            parent: Parent QObject.
        """
        super().__init__(parent)
        
        self._client = client
        self._session_id = session_id
        self._task = task
        self._file_paths = file_paths or []
    
    def run(self):
        """Execute the agent task."""
        try:
            result = self._client.execute_task(
                session_id=self._session_id,
                task=self._task,
                file_paths=self._file_paths,
            )
            self.task_completed.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(5000)
        self.deleteLater()


class StartAgentSessionWorker(QThread):
    """
    Worker thread for starting an agent session.
    
    Signals:
        session_started: Emitted with session_id on success.
        error_occurred: Emitted with error message on failure.
    """
    
    session_started = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, client: AgentClient, parent=None):
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


class DeleteAgentSessionWorker(QThread):
    """
    Worker thread for deleting an agent session.
    
    Signals:
        session_deleted: Emitted on success.
        error_occurred: Emitted with error message on failure.
    """
    
    session_deleted = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, client: AgentClient, session_id: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._session_id = session_id
    
    def run(self):
        try:
            self._client.delete_session(self._session_id)
            self.session_deleted.emit()
        except Exception:
            self.session_deleted.emit()
    
    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()