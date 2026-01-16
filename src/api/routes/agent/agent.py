"""
Agent API endpoints.

Provides endpoints for:
- Starting agent task sessions
- Executing tasks with optional file uploads
- Deleting agent sessions
- Listing all agent sessions
- Retrieving session history
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from api.dependencies import get_agent_service, get_session_history
from api.auth import get_current_user, UserContext
from api.routes.agent.agent_task import get_job, start_agent_job
from api.schemas import (
    AgentStartResponse,
    DeleteResponse,
    SessionResponse,
    SessionSummary,
    SessionHistoryResponse,
    MessageResponse,
    TruncateRequest,
    TruncateResponse,
)
from practorflow.services.agent import AgentService
from practorflow.services.history.truncator import truncate_messages
from practorflow.session_store.session_history import SessionHistory
from practorflow.logger.logger import get_logger

logger = get_logger("agent-api", level="INFO")

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get(
    "",
    response_model=SessionResponse,
    summary="Start a new agent task session",
    description="Creates a new agent task session and returns the session ID.",
)
async def start_session(
    current_user: UserContext = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service),
) -> SessionResponse:
    """
    Start a new agent task session.

    Args:
        current_user: Authenticated user context.
        agent_service: Agent service instance.

    Returns:
        SessionResponse with the new session_id.
    """
    logger.info(f"[Agent API] Starting new session for user: {current_user.user_id}")

    session_id = await agent_service.start_task()

    logger.info(f"[Agent API] Session created: {session_id}")

    return SessionResponse(
        session_id=session_id,
        message="Agent session created successfully",
    )


@router.post(
    "/{session_id}/execute",
    response_model=AgentStartResponse,
    summary="Execute an agent task",
    description="Execute a task using the multi-agent pipeline with optional file uploads.",
)
async def execute_task(
    session_id: str,
    task: str = Form(..., description="Task description to execute"),
    files: Optional[List[UploadFile]] = File(
        default=None, description="Optional files to upload"
    ),
    current_user: UserContext = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentStartResponse:
    """
    Execute a task using the multi-agent pipeline.

    Args:
        session_id: Session identifier.
        task: Task description to execute.
        files: Optional list of files to upload and index.
        current_user: Authenticated user context.
        agent_service: Agent service instance.

    Returns:
        AgentTaskResult with task outcome and artifacts.

    Raises:
        HTTPException: If execution fails.
    """
    logger.info(
        f"[Agent API] Scheduling async task for session: {session_id} "
        f"from user: {current_user.user_id}"
    )

    if files:
        filenames = [f.filename for f in files]
        logger.info(f"[Agent API] Files uploaded: {filenames}")

    try:
        job_id = await start_agent_job(
            agent_service=agent_service,
            session_id=session_id,
            task=task,
            user=current_user.user_id,
            files=files,
        )

        logger.info(f"[Agent API] Job {job_id} scheduled for session: {session_id}")

        return AgentStartResponse(job_id=job_id, status="scheduled")

    except Exception as e:
        logger.error(f"[Agent API] Failed to schedule job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to schedule agent task: {str(e)}",
        )


@router.get(
    "/jobs/{job_id}",
    summary="Get agent job status",
    description="Returns status and result of an agent execution job.",
)
async def get_agent_job(
    job_id: str,
    current_user: UserContext = Depends(get_current_user),
):
    """
    Return the current execution status of an agent job and, if available,
    its result or error information.
    """
    try:
        job = get_job(job_id)

        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job["user"] != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return {
            "job_id": job_id,
            "status": job["status"],
            "result": job["result"],
            "error": job["error"],
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch job status: {str(e)}",
        )


@router.delete(
    "/{session_id}",
    response_model=DeleteResponse,
    summary="Delete an agent session",
    description="Deletes an agent session and all associated documents.",
)
async def delete_session(
    session_id: str,
    current_user: UserContext = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service),
) -> DeleteResponse:
    """
    Delete an agent session.

    Args:
        session_id: Session identifier to delete.
        current_user: Authenticated user context.
        agent_service: Agent service instance.

    Returns:
        DeleteResponse with deletion status.

    Raises:
        HTTPException: If session not found.
    """
    logger.info(
        f"[Agent API] Deleting session: {session_id} by user: {current_user.user_id}"
    )

    deleted = await agent_service.delete_task(session_id)

    if not deleted:
        logger.warning(f"[Agent API] Session not found for deletion: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    logger.info(f"[Agent API] Session deleted: {session_id}")

    return DeleteResponse(
        session_id=session_id,
        deleted=True,
        message="Agent session deleted successfully",
    )


@router.get(
    "/sessions",
    response_model=List[SessionSummary],
    summary="List all agent sessions",
    description="Returns a list of all agent sessions, optionally filtered by user.",
)
async def list_sessions(
    user: Optional[str] = Query(default=None, description="Filter sessions by user"),
    current_user: UserContext = Depends(get_current_user),
    session_history: SessionHistory = Depends(get_session_history),
) -> List[SessionSummary]:
    """
    List all agent sessions.

    Args:
        user: Optional user identifier to filter sessions.
        current_user: Authenticated user context.
        session_history: Session history instance.

    Returns:
        List of SessionSummary objects sorted by updated_at descending.
    """
    filter_user = user if user is not None else current_user.user_id

    logger.info(
        f"[Agent API] Listing sessions for user: {filter_user} (requested by: {current_user.user_id})"
    )

    sessions = session_history.list_sessions(user=filter_user)

    summaries = []
    for session in sessions:
        if session.metadata.get("type") == "agent":
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

    logger.info(f"[Agent API] Found {len(summaries)} agent sessions")

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
        f"[Agent API] Getting history for session: {session_id} by user: {current_user.user_id}"
    )

    session = session_history.get_history(session_id)

    if session is None:
        logger.warning(f"[Agent API] Session not found: {session_id}")
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
        f"[Agent API] Returning {len(messages)} messages for session: {session_id}"
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


@router.put(
    "/{session_id}/truncate",
    response_model=TruncateResponse,
    summary="Truncate agent session messages",
    description="Removes all messages from the specified index onwards. Used for edit-and-regenerate functionality.",
)
async def truncate_message_request(
    session_id: str,
    request: TruncateRequest,
    current_user: UserContext = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service),
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
        agent_service: Agent service instance.

    Returns:
        TruncateResponse with truncation results.

    Raises:
        HTTPException: If session not found or invalid index.
    """
    logger.info(
        f"[Agent API] Truncating messages for session: {session_id} "
        f"from index {request.from_index} by user: {current_user.user_id}"
    )

    try:
        truncated_count = await truncate_messages(
            session_id, request.from_index, agent_service._session_store
        )
    except ValueError as e:
        logger.warning(f"[Agent API] Invalid truncate request: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    if truncated_count is None:
        logger.warning(f"[Agent API] Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    # Get remaining count
    session = agent_service.get_session(session_id)
    remaining_count = len(session.messages) if session else 0

    logger.info(
        f"[Agent API] Truncated {truncated_count} messages from session: {session_id}, "
        f"{remaining_count} remaining"
    )

    return TruncateResponse(
        session_id=session_id,
        truncated_count=truncated_count,
        remaining_count=remaining_count,
        message="Messages truncated successfully",
    )