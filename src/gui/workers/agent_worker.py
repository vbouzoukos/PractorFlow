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
        try:
            job_id = self._client.execute_task(
                session_id=self._session_id,
                task=self._task,
                file_paths=self._file_paths,
            )

            while True:
                job = self._client.get_job(job_id)

                if job.status == "completed":
                    self.task_completed.emit(
                        AgentTaskResult(
                            success=True,
                            output=job.result,
                            error=None,
                        )
                    )
                    return

                if job.status == "failed":
                    self.error_occurred.emit(job.error or "Agent job failed")
                    return

                time.sleep(1)

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

class AgentEditResumeWorker(QThread):
    """
    Worker for agent edit → truncate → resume flow.

    This is ADDITIVE functionality:
    - Does NOT modify AgentTaskWorker
    - Reuses AgentClient.truncate_messages
    - Reuses AgentTaskWorker for execution
    """

    completed = Signal(object)   # AgentTaskResult
    error_occurred = Signal(str)

    def __init__(
        self,
        client: AgentClient,
        session_id: str,
        from_index: int,
        updated_task: str,
        file_paths: Optional[list] = None,
        parent=None,
    ):
        super().__init__(parent)

        self._client = client
        self._session_id = session_id
        self._from_index = from_index
        self._updated_task = updated_task
        self._file_paths = file_paths or []

        self._task_worker: Optional[AgentTaskWorker] = None

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

            # 2. Resume agent execution using EXISTING worker
            self._task_worker = AgentTaskWorker(
                client=self._client,
                session_id=self._session_id,
                task=self._updated_task,
                file_paths=self._file_paths,
            )

            self._task_worker.task_completed.connect(self.completed)
            self._task_worker.error_occurred.connect(self.error_occurred)

            self._task_worker.start()
            self._task_worker.wait()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker and child worker."""
        if self._task_worker:
            self._task_worker.safe_delete()

        if self.isRunning():
            self.wait(2000)

        self.deleteLater()
