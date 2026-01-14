"""
Chat API endpoints.

Provides endpoints for:
- Starting chat sessions
- Sending messages with optional file uploads (SSE streaming response)
- Deleting chat sessions
- Listing all sessions
- Retrieving session history
- Listing session documents
- Deleting session documents
"""

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from api.dependencies import get_chat_service, get_session_history
from api.auth import get_current_user, UserContext
from api.schemas import (
    DeleteResponse,
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    MessageResponse,
    SessionHistoryResponse,
    SessionResponse,
    SessionSummary,
    StreamChunkData,
    TruncateRequest,
    TruncateResponse,
)
from practorflow.services.chat import ChatService
from practorflow.services.history.truncator import truncate_messages
from practorflow.session_store.session_history import SessionHistory
from practorflow.logger.logger import get_logger

logger = get_logger("agent-api", level="INFO")

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get(
    "",
    response_model=SessionResponse,
    summary="Start a new chat session",
    description="Creates a new chat session and returns the session ID.",
)
async def start_session(
    current_user: UserContext = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> SessionResponse:
    """
    Start a new chat session.

    Args:
        current_user: Authenticated user context.
        chat_service: Chat service instance.

    Returns:
        SessionResponse with the new session_id.
    """
    logger.info(f"[Chat API] Starting new session for user: {current_user.user_id}")

    session_id = await chat_service.start_chat()

    logger.info(f"[Chat API] Session created: {session_id}")

    return SessionResponse(
        session_id=session_id,
        message="Session created successfully",
    )


@router.post(
    "/{session_id}",
    summary="Send a message and stream response",
    description="Send a message with optional file uploads. Returns SSE streaming response.",
)
async def chat_message(
    session_id: str,
    message: str = Form(..., description="User message text"),
    files: Optional[List[UploadFile]] = File(
        default=None, description="Optional files to upload"
    ),
    current_user: UserContext = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Send a message and receive streaming response.

    Args:
        session_id: Session identifier.
        message: User message text.
        files: Optional list of files to upload and index.
        current_user: Authenticated user context.
        chat_service: Chat service instance.

    Returns:
        SSE streaming response with chat chunks.

    Raises:
        HTTPException: If session not found or other errors occur.
    """
    logger.info(f"[Chat API] Message received for session: {session_id} from user: {current_user.user_id}")

    if files:
        filenames = [f.filename for f in files]
        logger.info(f"[Chat API] Files uploaded: {filenames}")

    async def generate_sse():
        """Generate SSE events from chat stream."""
        try:
            async for chunk in chat_service.chat_stream(
                session_id=session_id,
                message=message,
                user=current_user.user_id,
                files=files,
            ):
                chunk_data = StreamChunkData(
                    text=chunk.text,
                    finished=chunk.finished,
                    finish_reason=chunk.finish_reason,
                    usage=chunk.usage,
                )

                data = json.dumps(chunk_data.model_dump())
                yield f"data: {data}\n\n"

                if chunk.finished:
                    yield "data: [DONE]\n\n"

        except ValueError as e:
            logger.error(f"[Chat API] ValueError: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"

        except Exception as e:
            logger.error(f"[Chat API] Error during streaming: {e}")
            error_data = json.dumps(
                {"error": "Internal server error", "detail": str(e)}
            )
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete(
    "/{session_id}",
    response_model=DeleteResponse,
    summary="Delete a chat session",
    description="Deletes a chat session and all associated documents.",
)
async def delete_session(
    session_id: str,
    current_user: UserContext = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> DeleteResponse:
    """
    Delete a chat session.

    Args:
        session_id: Session identifier to delete.
        current_user: Authenticated user context.
        chat_service: Chat service instance.

    Returns:
        DeleteResponse with deletion status.

    Raises:
        HTTPException: If session not found.
    """
    logger.info(f"[Chat API] Deleting session: {session_id} by user: {current_user.user_id}")

    deleted = await chat_service.delete_chat(session_id)

    if not deleted:
        logger.warning(f"[Chat API] Session not found for deletion: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    logger.info(f"[Chat API] Session deleted: {session_id}")

    return DeleteResponse(
        session_id=session_id,
        deleted=True,
        message="Session deleted successfully",
    )


@router.get(
    "/sessions",
    response_model=List[SessionSummary],
    summary="List all chat sessions",
    description="Returns a list of all chat sessions, optionally filtered by user.",
)
async def list_sessions(
    user: Optional[str] = Query(default=None, description="Filter sessions by user"),
    current_user: UserContext = Depends(get_current_user),
    session_history: SessionHistory = Depends(get_session_history),
) -> List[SessionSummary]:
    """
    List all chat sessions.

    Args:
        user: Optional user identifier to filter sessions.
        current_user: Authenticated user context.
        session_history: Session history instance.

    Returns:
        List of SessionSummary objects sorted by updated_at descending.
    """
    filter_user = user if user is not None else current_user.user_id

    logger.info(f"[Chat API] Listing sessions for user: {filter_user} (requested by: {current_user.user_id})")

    sessions = session_history.list_sessions(user=filter_user)

    summaries = []
    for session in sessions:
        summaries.append(
            SessionSummary(
                session_id=session.session_id,
                user=session.user,
                message_count=len(session.messages),
                document_count=len(session.documents),
                created_at=session.created_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
            )
        )

    logger.info(f"[Chat API] Found {len(summaries)} sessions")

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
    logger.info(f"[Chat API] Getting history for session: {session_id} by user: {current_user.user_id}")

    session = session_history.get_history(session_id)

    if session is None:
        logger.warning(f"[Chat API] Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

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
        f"[Chat API] Returning {len(messages)} messages for session: {session_id}"
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
    logger.info(f"[Chat API] Listing documents for session: {session_id} by user: {current_user.user_id}")

    session = session_history.get_history(session_id)

    if session is None:
        logger.warning(f"[Chat API] Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    documents = [
        DocumentInfo(
            id=doc["id"],
            filename=doc["filename"],
            file_type=doc.get("file_type", "unknown"),
        )
        for doc in session.documents
    ]

    logger.info(f"[Chat API] Found {len(documents)} documents for session: {session_id}")

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
    chat_service: ChatService = Depends(get_chat_service),
) -> DocumentDeleteResponse:
    """
    Delete a document from a session.

    Removes the document from both the session and the knowledge store.

    Args:
        session_id: Session identifier.
        document_id: Document identifier to delete.
        current_user: Authenticated user context.
        chat_service: Chat service instance.

    Returns:
        DocumentDeleteResponse with deletion status.

    Raises:
        HTTPException: If session or document not found.
    """
    logger.info(
        f"[Chat API] Deleting document: {document_id} from session: {session_id} by user: {current_user.user_id}"
    )

    deleted = await chat_service.delete_session_document(session_id, document_id)

    if deleted is None:
        logger.warning(f"[Chat API] Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    if not deleted:
        logger.warning(f"[Chat API] Document not found: {document_id}")
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    logger.info(f"[Chat API] Document deleted: {document_id} from session: {session_id}")

    return DocumentDeleteResponse(
        session_id=session_id,
        document_id=document_id,
        deleted=True,
        message="Document deleted successfully",
    )

# Add this endpoint to api/routers/chat.py
#
# Update imports to include TruncateRequest and TruncateResponse:
#
# from api.schemas import (
#     DeleteResponse,
#     DocumentDeleteResponse,
#     DocumentInfo,
#     DocumentListResponse,
#     MessageResponse,
#     SessionHistoryResponse,
#     SessionResponse,
#     SessionSummary,
#     StreamChunkData,
#     TruncateRequest,      # <-- add
#     TruncateResponse,     # <-- add
# )


@router.put(
    "/{session_id}/truncate",
    response_model=TruncateResponse,
    summary="Truncate session messages",
    description="Removes all messages from the specified index onwards. Used for edit-and-regenerate functionality.",
)
async def truncate_message_request(
    session_id: str,
    request: TruncateRequest,
    current_user: UserContext = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
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
        chat_service: Chat service instance.

    Returns:
        TruncateResponse with truncation results.

    Raises:
        HTTPException: If session not found or invalid index.
    """
    logger.info(
        f"[Chat API] Truncating messages for session: {session_id} "
        f"from index {request.from_index} by user: {current_user.user_id}"
    )

    try:
        truncated_count = await truncate_messages(
            session_id, request.from_index, chat_service._session_store
        )
    except ValueError as e:
        logger.warning(f"[Chat API] Invalid truncate request: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    if truncated_count is None:
        logger.warning(f"[Chat API] Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    # Get remaining count
    session = chat_service.get_session(session_id)
    remaining_count = len(session.messages) if session else 0

    logger.info(
        f"[Chat API] Truncated {truncated_count} messages from session: {session_id}, "
        f"{remaining_count} remaining"
    )

    return TruncateResponse(
        session_id=session_id,
        truncated_count=truncated_count,
        remaining_count=remaining_count,
        message="Messages truncated successfully",
    )