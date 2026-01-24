"""
Chat API endpoints.

Provides endpoints for:
- Starting chat sessions
- Sending messages with optional file uploads (SSE streaming response)
"""

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.dependencies import get_chat_service
from api.auth import get_current_user, UserContext
from api.schemas import (
    SessionResponse,
    StreamChunkData,
)
from practorflow.services.chat import ChatService
from practorflow.logger.logger import get_logger

logger = get_logger("chat-api", level="INFO")

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