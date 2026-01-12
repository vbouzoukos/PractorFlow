import asyncio
import uuid
from typing import Dict, Any, Optional

from practorflow.services.agent import AgentService
from practorflow.logger.logger import get_logger

logger = get_logger("agent-api", level="INFO")

# In-memory job registry (process-local)
JOBS: Dict[str, Dict[str, Any]] = {}


def create_job(user: str) -> str:
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "user": user,
        "status": "pending",
        "result": None,
        "error": None,
    }
    return job_id

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return JOBS.get(job_id)


async def _run_job(
    job_id: str,
    *,
    agent_service: AgentService,
    session_id: str,
    task: str,
    user: str,
    files,
) -> None:
    logger.info(f"[Agent Job] Starting job {job_id} for session {session_id}")
    JOBS[job_id]["status"] = "running"

    try:
        result = await agent_service.execute_task(
            session_id=session_id,
            task=task,
            user=user,
            files=files,
        )

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result
        logger.info(f"[Agent Job] Job {job_id} completed successfully")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        logger.exception(f"[Agent Job] Job {job_id} failed")


def start_agent_job(
    *,
    agent_service: AgentService,
    session_id: str,
    task: str,
    user: str,
    files,
) -> str:
    job_id = create_job()

    asyncio.create_task(
        _run_job(
            job_id,
            agent_service=agent_service,
            session_id=session_id,
            task=task,
            user=user,
            files=files,
        )
    )

    logger.info(f"[Agent Job] Job {job_id} scheduled")
    return job_id
