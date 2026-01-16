"""
Session API endpoints.

Provides unified endpoints for session management:
- Listing all sessions
- Retrieving session history
- Deleting sessions
- Truncating session messages
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_delete_session_service, get_session_history
from api.auth import get_current_user, UserContext
from api.schemas import (
    DeleteResponse,
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    MessageResponse,
    SessionHistoryResponse,
    SessionSummary,
    TruncateRequest,
    TruncateResponse,
)
from practorflow.services.history.truncator import DeleteSessionService
from practorflow.session_store.session_history import SessionHistory
from practorflow.logger.logger import get_logger

logger = get_logger("session-api", level="INFO")

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get(
    "",
    response_model=List[SessionSummary],
    summary="List all sessions",
    description="Returns a list of all sessions, optionally filtered by user and type.",
)
async def list_sessions(
    current_user: UserContext = Depends(get_current_user),
    session_history: SessionHistory = Depends(get_session_history),
) -> List[SessionSummary]:
    """
    List all sessions.

    Args:
        current_user: Authenticated user context.
        session_history: Session history instance.

    Returns:
        List of SessionSummary objects sorted by updated_at descending.
    """
    logger.info(f"[Session API] Listing sessions for user: {current_user.user_id} ")

    sessions = session_history.list_sessions(user=current_user.user_id)

    summaries = []
    for session in sessions:
        summaries.append(
            SessionSummary(
                session_id=session.session_id,
                user=session.user,
                title=session.title,
                message_count=len(session.messages),
                document_count=len(session.documents),
                created_at=session.created_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
            )
        )

    logger.info(f"[Session API] Found {len(summaries)} sessions")

    return summaries


@router.get(
    "/search",
    response_model=List[SessionSummary],
    summary="Search in sessions",
    description="Returns a list of sessions matching the search term, optionally filtered by user and type.",
)
async def search(
    term: str = Query(default=None, description="Search term"),
    current_user: UserContext = Depends(get_current_user),
    session_history: SessionHistory = Depends(get_session_history),
) -> List[SessionSummary]:
    """
    List all sessions.

    Args:
        current_user: Authenticated user context.
        session_history: Session history instance.

    Returns:
        List of SessionSummary objects sorted by relevence and updated_at descending.
    """
    logger.info(f"[Session API] Listing sessions for user: {current_user.user_id} ")

    sessions = session_history.sessions_by_title(title=term, user=current_user.user_id)

    summaries = []
    for session in sessions:
        summaries.append(
            SessionSummary(
                session_id=session.session_id,
                user=session.user,
                title=session.title,
                message_count=len(session.messages),
                document_count=len(session.documents),
                created_at=session.created_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
            )
        )

    logger.info(f"[Session API] Found {len(summaries)} sessions")

    return summaries


@router.get(
    "/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get session history",
    description="Returns full session with complete message history.",
)
async def get_history(
    session_id: str,
    current_user: UserContext = Depends(get_current_user),
    session_history: SessionHistory = Depends(get_session_history),
) -> SessionHistoryResponse:
    """
    Get full session with message history.

    Args:
        session_id: Session identifier.
        current_user: Authenticated user context.
        session_history: Session history instance.

    Returns:
        SessionHistoryResponse with full message history.

    Raises:
        HTTPException: If session not found.
    """
    logger.info(
        f"[Session API] Getting history for session: {session_id} "
        f"by user: {current_user.user_id}"
    )

    session = session_history.get_history(session_id)

    if session is None:
        logger.warning(f"[Session API] Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    if session.user != current_user.user_id:
        raise HTTPException(status_code=403, detail=f"Forbidden")
    messages = [
        MessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.get_text_content(),
            timestamp=msg.timestamp.isoformat(),
        )
        for msg in session.messages
    ]

    logger.info(
        f"[Session API] Returning {len(messages)} messages for session: {session_id}"
    )

    return SessionHistoryResponse(
        session_id=session.session_id,
        user=session.user,
        instructions=session.instructions,
        messages=messages,
        document_count=len(session.documents),
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )


@router.delete(
    "/{session_id}",
    response_model=DeleteResponse,
    summary="Delete a session",
    description="Deletes a session and all associated documents.",
)
async def delete_session(
    session_id: str,
    current_user: UserContext = Depends(get_current_user),
    delete_session_service: DeleteSessionService = Depends(get_delete_session_service),
) -> DeleteResponse:
    """
    Delete a session.

    Args:
        session_id: Session identifier to delete.
        current_user: Authenticated user context.
        delete_session_service: Delete Session Service instance

    Returns:
        DeleteResponse with deletion status.

    Raises:
        HTTPException: If session not found.
    """
    logger.info(
        f"[Session API] Deleting session: {session_id} by user: {current_user.user_id}"
    )

    deleted = await delete_session_service.delete_chat(session_id, current_user.user_id)

    if not deleted:
        logger.warning(f"[Session API] Session not found for deletion: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    logger.info(f"[Session API] Session deleted: {session_id}")

    return DeleteResponse(
        session_id=session_id,
        deleted=True,
        message="Session deleted successfully",
    )


@router.put(
    "/{session_id}/truncate",
    response_model=TruncateResponse,
    summary="Truncate session messages",
    description="Removes all messages from the specified index onwards. Used for edit-and-regenerate functionality.",
)
async def truncate_session_messages(
    session_id: str,
    request: TruncateRequest,
    current_user: UserContext = Depends(get_current_user),
    delete_session_service: DeleteSessionService = Depends(get_delete_session_service),
) -> TruncateResponse:
    """
    Truncate messages from a given index onwards.

    Removes all messages starting from the specified index (inclusive).
    Used for edit-and-regenerate functionality where the user edits
    a message and all subsequent messages are removed.

    Args:
        session_id: Session identifier.
        request: TruncateRequest with from_index.
        current_user: Authenticated user context.
        delete_session_service: Delete Session Service instance

    Returns:
        TruncateResponse with truncation results.

    Raises:
        HTTPException: If session not found or invalid index.
    """
    logger.info(
        f"[Session API] Truncating messages for session: {session_id} "
        f"from index {request.from_index} by user: {current_user.user_id}"
    )

    try:
        truncated_count = await delete_session_service.truncate_messages(
            session_id, request.from_index, current_user.user_id
        )
    except ValueError as e:
        logger.warning(f"[Session API] Invalid truncate request: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    if truncated_count is None:
        logger.warning(f"[Session API] Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    logger.info(
        f"[Session API] Truncated {truncated_count} messages from session: {session_id}, "
    )

    return TruncateResponse(
        session_id=session_id,
        truncated_count=truncated_count,
        message="Messages truncated successfully",
    )


@router.get(
    "/{session_id}/documents",
    response_model=DocumentListResponse,
    summary="List session documents",
    description="Returns a list of all documents in the session.",
)
async def list_session_documents(
    session_id: str,
    current_user: UserContext = Depends(get_current_user),
    session_history: SessionHistory = Depends(get_session_history),
) -> DocumentListResponse:
    """
    List all documents in a session.

    Args:
        session_id: Session identifier.
        current_user: Authenticated user context.
        session_history: Session history instance.

    Returns:
        DocumentListResponse with list of documents.

    Raises:
        HTTPException: If session not found.
    """
    logger.info(
        f"[Session API] Listing documents for session: {session_id} "
        f"by user: {current_user.user_id}"
    )

    session = session_history.get_history(session_id)

    if session is None:
        logger.warning(f"[Session API] Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    documents = [
        DocumentInfo(
            id=doc["id"],
            filename=doc["filename"],
            file_type=doc.get("file_type", "unknown"),
        )
        for doc in session.documents
    ]

    logger.info(
        f"[Session API] Found {len(documents)} documents for session: {session_id}"
    )

    return DocumentListResponse(
        session_id=session_id,
        documents=documents,
        count=len(documents),
    )


@router.delete(
    "/{session_id}/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a session document",
    description="Deletes a document from the session and knowledge store.",
)
async def delete_session_document(
    session_id: str,
    document_id: str,
    current_user: UserContext = Depends(get_current_user),
    delete_session_service: DeleteSessionService = Depends(get_delete_session_service),
) -> DocumentDeleteResponse:
    """
    Delete a document from a session.

    Removes the document from both the session and the knowledge store.

    Args:
        session_id: Session identifier.
        document_id: Document identifier to delete.
        current_user: Authenticated user context.
        delete_session_service: Delete Session Service instance

    Returns:
        DocumentDeleteResponse with deletion status.

    Raises:
        HTTPException: If session or document not found.
    """
    logger.info(
        f"[Session API] Deleting document: {document_id} from session: {session_id} "
        f"by user: {current_user.user_id}"
    )

    deleted = await delete_session_service.delete_session_document(
        session_id,
        document_id,
        request_user=current_user.user_id,
    )

    if deleted is None:
        logger.warning(f"[Session API] Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    if not deleted:
        logger.warning(f"[Session API] Document not found: {document_id}")
        raise HTTPException(
            status_code=404, detail=f"Document not found: {document_id}"
        )

    logger.info(
        f"[Session API] Document deleted: {document_id} from session: {session_id}"
    )

    return DocumentDeleteResponse(
        session_id=session_id,
        document_id=document_id,
        deleted=True,
        message="Document deleted successfully",
    )
