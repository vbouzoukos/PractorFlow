"""
Execution context for multi-agent pipeline.

Provides a unified context object that encapsulates all session-related
state needed during task execution, including message history conversion.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set

from pydantic_ai.messages import ModelMessage

from practorflow.llm.base.session import Session
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration
from practorflow.services.history.builder import build_message_history

logger = get_logger("context", level=appConfiguration.LoggerConfiguration.AgentLevel)


@dataclass
class ExecutionContext:
    """
    Unified context for agent pipeline execution.

    Bundles all session-related state needed by runners:
    - Session reference for persistence
    - Pre-built message history in pydantic_ai format
    - Document scope for knowledge search filtering
    - Document context string for planner prompts
    """

    session: Session
    message_history: List[ModelMessage] = field(default_factory=list)
    document_scope: Optional[Set[str]] = None
    document_context: Optional[str] = None

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self.session.session_id

    @property
    def has_history(self) -> bool:
        """Check if conversation history exists."""
        return len(self.message_history) > 0

    @property
    def has_documents(self) -> bool:
        """Check if documents are available."""
        return self.document_scope is not None and len(self.document_scope) > 0

    @property
    def history_length(self) -> int:
        """Get number of messages in history."""
        return len(self.message_history)


def get_document_scope(session: Session) -> Optional[Set[str]]:
    """
    Extract document IDs from session for scoped search.

    Args:
        session: Session to extract document IDs from.

    Returns:
        Set of document IDs or None if no documents.
    """
    if not session.documents:
        return None

    doc_ids = {doc["id"] for doc in session.documents if "id" in doc}
    return doc_ids if doc_ids else None


def get_document_context(session: Session) -> Optional[str]:
    """
    Build document context string for planner prompts.

    Creates a human-readable description of available documents
    that helps the planner understand what resources are available.

    Args:
        session: Session containing documents.

    Returns:
        Document context string or None if no documents.
    """
    if not session.documents:
        return None

    doc_names = [doc.get("filename", "unknown") for doc in session.documents]
    if not doc_names:
        return None

    return "Available documents:\n" + "\n".join(f"- {name}" for name in doc_names)


def build_execution_context(
    session: Session,
    document_ids: Optional[Set[str]] = None,
) -> ExecutionContext:
    """
    Build complete execution context from session.

    Creates an ExecutionContext with all derived state pre-computed:
    - Full message history converted to pydantic_ai format
    - Document scope extracted for knowledge search (merged with provided document_ids)
    - Document context string for planner

    Args:
        session: Session to build context from.
        document_ids: Optional additional document IDs to include in scope.

    Returns:
        ExecutionContext ready for pipeline execution.
    """
    message_history = build_message_history(session)

    document_scope = get_document_scope(session)
    if document_ids:
        if document_scope:
            document_scope = document_scope | document_ids
        else:
            document_scope = document_ids

    logger.info(
        f"[Context] Built context with {len(message_history)} history messages"
        + (f", {len(document_scope)} documents in scope" if document_scope else "")
    )

    return ExecutionContext(
        session=session,
        message_history=message_history,
        document_scope=document_scope,
        document_context=get_document_context(session),
    )