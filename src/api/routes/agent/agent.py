"""
Agent API endpoints.

Provides endpoints for:
- Starting agent task sessions
- Executing tasks with optional file uploads
- Getting job status
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import get_agent_service
from api.auth import get_current_user, UserContext
from api.routes.agent.agent_task import get_job, start_agent_job
from api.schemas import (
    AgentStartResponse,
    SessionResponse,
)
from practorflow.services.agent import AgentService
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