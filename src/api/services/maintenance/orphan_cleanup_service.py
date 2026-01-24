"""
Orphan document cleanup service.

Removes documents from the knowledge store that are not referenced
by any existing session. This handles cleanup of documents that
may have been orphaned due to incomplete session deletions or
application crashes.
"""

from typing import Set

from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.session_store.session_history import SessionHistory
from practorflow.logger.logger import get_logger

logger = get_logger("orphan-cleanup", level="INFO")


class OrphanCleanupService:
    """
    Service for cleaning up orphaned documents from the knowledge store.
    
    A document is considered orphaned if it exists in the knowledge store
    but is not referenced by any session's documents list.
    """
    
    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        session_history: SessionHistory,
    ):
        """
        Initialize the orphan cleanup service.
        
        Args:
            knowledge_store: Knowledge store instance.
            session_history: Session history instance.
        """
        self._knowledge_store = knowledge_store
        self._session_history = session_history
    
    def get_session_document_ids(self) -> Set[str]:
        """
        Get all document IDs referenced by any session.
        
        Returns:
            Set of document IDs from all sessions.
        """
        document_ids: Set[str] = set()
        
        sessions = self._session_history.list_sessions(user=None)
        
        for session in sessions:
            for doc in session.documents:
                doc_id = doc.get("id")
                if doc_id:
                    document_ids.add(doc_id)
        
        return document_ids
    
    def get_knowledge_store_document_ids(self) -> Set[str]:
        """
        Get all document IDs in the knowledge store.
        
        Returns:
            Set of document IDs from the knowledge store.
        """
        documents = self._knowledge_store.list_documents()
        return {doc["id"] for doc in documents}
    
    def find_orphan_document_ids(self) -> Set[str]:
        """
        Find document IDs that are orphaned.
        
        A document is orphaned if it exists in the knowledge store
        but is not referenced by any session.
        
        Returns:
            Set of orphaned document IDs.
        """
        session_doc_ids = self.get_session_document_ids()
        knowledge_store_doc_ids = self.get_knowledge_store_document_ids()
        
        orphan_ids = knowledge_store_doc_ids - session_doc_ids
        
        return orphan_ids
    
    def cleanup(self) -> int:
        """
        Remove orphaned documents from the knowledge store.
        
        Returns:
            Number of documents deleted.
        """
        orphan_ids = self.find_orphan_document_ids()
        
        if not orphan_ids:
            logger.debug("[OrphanCleanupService] No orphaned documents found")
            return 0
        
        logger.info(
            f"[OrphanCleanupService] Found {len(orphan_ids)} orphaned documents"
        )
        
        deleted_count = 0
        
        for doc_id in orphan_ids:
            try:
                deleted = self._knowledge_store.delete_document(doc_id)
                if deleted:
                    deleted_count += 1
                    logger.debug(
                        f"[OrphanCleanupService] Deleted orphaned document: {doc_id}"
                    )
                else:
                    logger.warning(
                        f"[OrphanCleanupService] Document not found for deletion: {doc_id}"
                    )
            except Exception as e:
                logger.error(
                    f"[OrphanCleanupService] Failed to delete document {doc_id}: {e}"
                )
        
        logger.info(
            f"[OrphanCleanupService] Cleanup complete. Deleted {deleted_count} documents"
        )
        
        return deleted_count