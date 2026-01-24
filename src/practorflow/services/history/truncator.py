from typing import Optional

from practorflow.llm.base.session_store import SessionStore
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.logger.logger import get_logger

from practorflow.settings.app_settings import appConfiguration

logger = get_logger(
    "chat_management", level=appConfiguration.LoggerConfiguration.AgentLevel
)


class DeleteSessionService:
    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        session_store: SessionStore,
    ):
        """
        Initialize chat delete service.

        Args:
            knowledge_store: Knowledge store for document storage and search.
            session_store: Session store for persisting chat sessions.
        """

        self._knowledge_store = knowledge_store
        self._session_store = session_store

        logger.info("[DeleteSessionService] Initialized")

    async def truncate_messages(
        self,
        session_id: str,
        from_index: int,
        request_user: str,
    ) -> Optional[int]:
        """
        Truncate messages from a given index onwards.

        Used for edit-and-regenerate functionality where the user
        edits a message and all subsequent messages are removed.

        Args:
            session_id: Session ID to truncate messages from.
            from_index: Index from which to truncate (inclusive).
                        Messages at and after this index are removed.

        Returns:
            Number of messages removed if successful.
            None if session was not found.

        Raises:
            ValueError: If from_index is negative.
        """
        if not self._session_store.exists(session_id):
            return None

        session = self._session_store.get(session_id)
        if session.user != request_user:
            raise PermissionError("Forbidden")
        removed_count = session.truncate_messages(from_index)

        # Save updated session
        self._session_store.save(session)

        return removed_count

    async def delete_chat(
        self,
        session_id: str,
        request_user: str,
    ) -> bool:
        """
        Delete a chat session and its associated documents.

        Removes the session from storage and deletes all documents
        that were uploaded during this session from the knowledge store.

        Args:
            session_id: Session ID to delete.

        Returns:
            True if session was deleted, False if not found.
        """
        if not self._session_store.exists(session_id):
            return False

        session = self._session_store.get(session_id)
        if session.user != request_user:
            raise PermissionError("Forbidden")
        # Delete all session documents from knowledge store
        for doc in session.documents:
            doc_id = doc.get("id")
            if doc_id:
                try:
                    self._knowledge_store.delete_document(doc_id)
                except Exception as e:
                    logger.warning(
                        f"[DeleteSessionService] Failed to delete document {doc_id}: {e}"
                    )

        # Delete session
        self._session_store.delete(session_id)

        logger.info(f"[DeleteSessionService] Deleted chat session: {session_id}")

        return True

    async def delete_session_document(
        self,
        session_id: str,
        document_id: str,
        request_user: str,
    ) -> Optional[bool]:
        """
        Delete a single document from a session.

        Removes the document from both the session and the knowledge store.

        Args:
            session_id: Session ID containing the document.
            document_id: Document ID to delete.

        Returns:
            True if document was deleted successfully.
            False if document was not found in session.
            None if session was not found.
        """
        if not self._session_store.exists(session_id):
            logger.warning(f"[DeleteSessionService] Session not found: {session_id}")
            return None

        session = self._session_store.get(session_id)
        if session.user != request_user:
            raise PermissionError("Forbidden")
        # Remove document from session
        removed = session.remove_document(document_id)

        if not removed:
            logger.warning(
                f"[DeleteSessionService] Document not found in session: {document_id}"
            )
            return False

        # Delete from knowledge store
        try:
            self._knowledge_store.delete_document(document_id)
            logger.debug(
                f"[DeleteSessionService] Deleted document from knowledge store: {document_id}"
            )
        except Exception as e:
            logger.warning(
                f"[DeleteSessionService] Failed to delete document from knowledge store {document_id}: {e}"
            )

        # Save updated session
        self._session_store.save(session)

        logger.info(
            f"[DeleteSessionService] Deleted document {document_id} from session {session_id}"
        )

        return True
