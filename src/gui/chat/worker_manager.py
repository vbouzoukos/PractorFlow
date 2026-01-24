"""
Worker lifecycle management for chat window.

Handles creation, cleanup, and shutdown of all async workers.
"""

from typing import Optional

from PySide6.QtCore import QObject

from gui.logger import get_logger

logger = get_logger("practorflow-client", level="INFO", log_file="logs/practorflow-client.log")

from gui.api.chat_client import ChatClient
from gui.api.session_client import SessionClient
from gui.api.agent_client import AgentClient
from gui.workers.stream_worker import StreamWorker, ChatEditResumeWorker
from gui.workers.session_worker import StartSessionWorker, DeleteSessionWorker
from gui.workers.agent_worker import (
    AgentTaskWorker,
    StartAgentSessionWorker,
    AgentEditResumeWorker,
)


class WorkerManager(QObject):
    """
    Manages async worker lifecycle.
    
    Centralizes worker creation, cleanup, and shutdown to keep
    ChatWindow focused on coordination logic.
    """
    
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        
        self._stream_worker: Optional[StreamWorker] = None
        self._agent_worker: Optional[AgentTaskWorker] = None
        self._session_worker: Optional[StartSessionWorker] = None
        self._delete_worker: Optional[DeleteSessionWorker] = None
        self._edit_resume_worker: Optional[ChatEditResumeWorker] = None
        
        logger.debug("WorkerManager initialized")
    
    # --- Stream Worker ---
    
    def create_stream_worker(
        self,
        client: ChatClient,
        session_id: str,
        message: str,
        file_paths: list,
    ) -> StreamWorker:
        """Create and return a stream worker."""
        try:
            self._cleanup_stream_worker()
            self._stream_worker = StreamWorker(
                client=client,
                session_id=session_id,
                message=message,
                file_paths=file_paths,
                parent=self,
            )
            logger.debug(f"Stream worker created for session {session_id}")
            return self._stream_worker
        except Exception as e:
            logger.error(f"Failed to create stream worker: {e}")
            raise
    
    def cleanup_stream_worker(self):
        """Public cleanup method for stream worker."""
        self._cleanup_stream_worker()
    
    def _cleanup_stream_worker(self):
        """Clean up stream worker."""
        try:
            if self._stream_worker:
                self._stream_worker.deleteLater()
                self._stream_worker = None
                logger.debug("Stream worker cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up stream worker: {e}")
            self._stream_worker = None
    
    # --- Agent Worker ---
    
    def create_agent_worker(
        self,
        client: AgentClient,
        session_id: str,
        task: str,
        file_paths: list,
    ) -> AgentTaskWorker:
        """Create and return an agent task worker."""
        try:
            self._cleanup_agent_worker()
            self._agent_worker = AgentTaskWorker(
                client=client,
                session_id=session_id,
                task=task,
                file_paths=file_paths,
                parent=self,
            )
            logger.debug(f"Agent worker created for session {session_id}")
            return self._agent_worker
        except Exception as e:
            logger.error(f"Failed to create agent worker: {e}")
            raise
    
    def cleanup_agent_worker(self):
        """Public cleanup method for agent worker."""
        self._cleanup_agent_worker()
    
    def _cleanup_agent_worker(self):
        """Clean up agent worker."""
        try:
            if self._agent_worker:
                self._agent_worker.deleteLater()
                self._agent_worker = None
                logger.debug("Agent worker cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up agent worker: {e}")
            self._agent_worker = None
    
    # --- Session Worker ---
    
    def create_session_worker(
        self,
        client: ChatClient,
        agent_mode: bool,
        agent_client: Optional[AgentClient] = None,
    ) -> StartSessionWorker:
        """Create and return a session start worker."""
        try:
            self._cleanup_session_worker()
            if agent_mode and agent_client:
                self._session_worker = StartAgentSessionWorker(agent_client, parent=self)
                logger.debug("Agent session worker created")
            else:
                self._session_worker = StartSessionWorker(client, parent=self)
                logger.debug("Chat session worker created")
            return self._session_worker
        except Exception as e:
            logger.error(f"Failed to create session worker: {e}")
            raise
    
    def cleanup_session_worker(self):
        """Public cleanup method for session worker."""
        self._cleanup_session_worker()
    
    def _cleanup_session_worker(self):
        """Clean up session worker."""
        try:
            if self._session_worker:
                self._session_worker.deleteLater()
                self._session_worker = None
                logger.debug("Session worker cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up session worker: {e}")
            self._session_worker = None
    
    # --- Delete Worker ---
    
    def create_delete_worker(
        self,
        client: ChatClient,
        session_id: str,
    ) -> DeleteSessionWorker:
        """Create and return a delete session worker."""
        try:
            self._cleanup_delete_worker()
            self._delete_worker = DeleteSessionWorker(
                client=client,
                session_id=session_id,
                parent=self,
            )
            logger.debug(f"Delete worker created for session {session_id}")
            return self._delete_worker
        except Exception as e:
            logger.error(f"Failed to create delete worker: {e}")
            raise
    
    def cleanup_delete_worker(self):
        """Public cleanup method for delete worker."""
        self._cleanup_delete_worker()
    
    def _cleanup_delete_worker(self):
        """Clean up delete worker."""
        try:
            if self._delete_worker:
                self._delete_worker.deleteLater()
                self._delete_worker = None
                logger.debug("Delete worker cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up delete worker: {e}")
            self._delete_worker = None
    
    # --- Edit Resume Worker ---
    
    def create_chat_edit_worker(
        self,
        client: ChatClient,
        session_client: SessionClient,
        session_id: str,
        from_index: int,
        updated_message: str,
        file_paths: list,
    ) -> ChatEditResumeWorker:
        """Create and return a chat edit resume worker."""
        try:
            self._cleanup_edit_resume_worker()
            self._edit_resume_worker = ChatEditResumeWorker(
                client=client,
                session_client=session_client,
                session_id=session_id,
                from_index=from_index,
                updated_message=updated_message,
                file_paths=file_paths,
                parent=self,
            )
            logger.debug(f"Chat edit worker created for session {session_id}")
            return self._edit_resume_worker
        except Exception as e:
            logger.error(f"Failed to create chat edit worker: {e}")
            raise
    
    def create_agent_edit_worker(
        self,
        client: AgentClient,
        session_client: SessionClient,
        session_id: str,
        from_index: int,
        updated_task: str,
        file_paths: list,
    ) -> AgentEditResumeWorker:
        """Create and return an agent edit resume worker."""
        try:
            self._cleanup_edit_resume_worker()
            self._edit_resume_worker = AgentEditResumeWorker(
                client=client,
                session_client=session_client,
                session_id=session_id,
                from_index=from_index,
                updated_task=updated_task,
                file_paths=file_paths,
                parent=self,
            )
            logger.debug(f"Agent edit worker created for session {session_id}")
            return self._edit_resume_worker
        except Exception as e:
            logger.error(f"Failed to create agent edit worker: {e}")
            raise
    
    def cleanup_edit_resume_worker(self):
        """Public cleanup method for edit resume worker."""
        self._cleanup_edit_resume_worker()
    
    def _cleanup_edit_resume_worker(self):
        """Clean up edit resume worker."""
        try:
            if self._edit_resume_worker:
                self._edit_resume_worker.deleteLater()
                self._edit_resume_worker = None
                logger.debug("Edit resume worker cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up edit resume worker: {e}")
            self._edit_resume_worker = None
    
    # --- Shutdown ---
    
    def shutdown(self):
        """Stop and clean up all workers."""
        logger.info("WorkerManager shutting down all workers")
        
        try:
            if self._stream_worker and self._stream_worker.isRunning():
                self._stream_worker.stop()
                self._stream_worker.wait(2000)
            self._cleanup_stream_worker()
        except Exception as e:
            logger.error(f"Error stopping stream worker: {e}")
        
        try:
            if self._agent_worker and self._agent_worker.isRunning():
                self._agent_worker.wait(5000)
            self._cleanup_agent_worker()
        except Exception as e:
            logger.error(f"Error stopping agent worker: {e}")
        
        try:
            if self._session_worker and self._session_worker.isRunning():
                self._session_worker.wait(2000)
            self._cleanup_session_worker()
        except Exception as e:
            logger.error(f"Error stopping session worker: {e}")
        
        try:
            if self._delete_worker and self._delete_worker.isRunning():
                self._delete_worker.wait(2000)
            self._cleanup_delete_worker()
        except Exception as e:
            logger.error(f"Error stopping delete worker: {e}")
        
        try:
            if self._edit_resume_worker and self._edit_resume_worker.isRunning():
                self._edit_resume_worker.wait(2000)
            self._cleanup_edit_resume_worker()
        except Exception as e:
            logger.error(f"Error stopping edit resume worker: {e}")
        
        logger.info("WorkerManager shutdown complete")