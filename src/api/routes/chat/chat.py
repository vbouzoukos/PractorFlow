"""
Chat API endpoints.

Provides endpoints for:
- Starting chat sessions
- Sending messages with optional file uploads (SSE streaming response)
- Deleting chat sessions
- Listing all sessions
- Retrieving session history
"""

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from api.dependencies import get_chat_service, get_session_history
from api.schemas import (
    DeleteResponse,
    SessionResponse,
    SessionSummary,
    SessionHistoryResponse,
    MessageResponse,
    StreamChunkData,
)
from practorflow.services.chat import ChatService
from session_store.session_history import SessionHistory
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
    chat_service: ChatService = Depends(get_chat_service),
) -> SessionResponse:
    """
    Start a new chat session.

    Returns:
        SessionResponse with the new session_id.
    """
    logger.info("[Chat API] Starting new session")

    session = await chat_service.start_chat()

    logger.info(f"[Chat API] Session created: {session}")

    return SessionResponse(
        session_id=session,
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
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Send a message and receive streaming response.

    Args:
        session_id: Session identifier.
        message: User message text.
        files: Optional list of files to upload and index.

    Returns:
        SSE streaming response with chat chunks.

    Raises:
        HTTPException: If session not found or other errors occur.
    """
    logger.info(f"[Chat API] Message received for session: {session_id}")

    # Log file info if present
    if files:
        filenames = [f.filename for f in files]
        logger.info(f"[Chat API] Files uploaded: {filenames}")

    async def generate_sse():
        """Generate SSE events from chat stream."""
        try:
            async for chunk in chat_service.chat_stream(
                session_id=session_id,
                message=message,
                files=files,
            ):
                chunk_data = StreamChunkData(
                    text=chunk.text,
                    finished=chunk.finished,
                    finish_reason=chunk.finish_reason,
                    usage=chunk.usage,
                )

                # Format as SSE event
                data = json.dumps(chunk_data.model_dump())
                yield f"data: {data}\n\n"

                if chunk.finished:
                    # Send done event
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
    chat_service: ChatService = Depends(get_chat_service),
) -> DeleteResponse:
    """
    Delete a chat session.

    Args:
        session_id: Session identifier to delete.

    Returns:
        DeleteResponse with deletion status.

    Raises:
        HTTPException: If session not found.
    """
    logger.info(f"[Chat API] Deleting session: {session_id}")

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
    session_history: SessionHistory = Depends(get_session_history),
) -> List[SessionSummary]:
    """
    List all chat sessions.

    Args:
        user: Optional user identifier to filter sessions.

    Returns:
        List of SessionSummary objects sorted by updated_at descending.
    """
    logger.info(f"[Chat API] Listing sessions (user={user})")

    sessions = session_history.list_sessions(user=user)

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
    session_history: SessionHistory = Depends(get_session_history),
) -> SessionHistoryResponse:
    """
    Get full session with message history.

    Args:
        session_id: Session identifier.

    Returns:
        SessionHistoryResponse with full message history.

    Raises:
        HTTPException: If session not found.
    """
    logger.info(f"[Chat API] Getting history for session: {session_id}")

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
