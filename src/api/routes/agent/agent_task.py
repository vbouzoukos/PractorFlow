import asyncio
import io
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, BinaryIO

from practorflow.services.agent import AgentService
from practorflow.logger.logger import get_logger

logger = get_logger("agent-api", level="INFO")

# In-memory job registry (process-local)
JOBS: Dict[str, Dict[str, Any]] = {}


@dataclass
class JobFile:
    """
    In-memory file storage for background job processing.
    
    Implements the ChatFile protocol to be compatible with AgentService.
    """
    
    file: BinaryIO
    filename: str
    content_type: Optional[str]


def create_job(user: str, files: Optional[List[JobFile]] = None) -> str:
    """
    Create a new job entry with optional file data.
    
    Args:
        user: User identifier.
        files: Optional list of JobFile objects with file data.
    
    Returns:
        Generated job ID.
    """
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "user": user,
        "status": "pending",
        "result": None,
        "error": None,
        "files": files,
    }
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return JOBS.get(job_id)


async def _read_upload_files(files) -> List[JobFile]:
    """
    Read UploadFile contents into memory before request completes.
    
    Args:
        files: List of FastAPI UploadFile objects.
    
    Returns:
        List of JobFile objects with file data in memory.
    """
    job_files = []
    
    for upload_file in files:
        content = await upload_file.read()
        job_file = JobFile(
            file=io.BytesIO(content),
            filename=upload_file.filename,
            content_type=upload_file.content_type,
        )
        job_files.append(job_file)
    
    return job_files


async def _run_job(
    job_id: str,
    *,
    agent_service: AgentService,
    session_id: str,
    task: str,
    user: str,
) -> None:
    logger.info(f"[Agent Job] Starting job {job_id} for session {session_id}")
    JOBS[job_id]["status"] = "running"

    try:
        # Retrieve stored files from job
        files = JOBS[job_id].get("files")
        
        result = await agent_service.execute_task(
            session_id=session_id,
            task=task,
            user=user,
            files=files,
        )

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result
        
        # Clear file data after processing
        JOBS[job_id]["files"] = None
        
        logger.info(f"[Agent Job] Job {job_id} completed successfully")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["files"] = None
        logger.exception(f"[Agent Job] Job {job_id} failed")


async def start_agent_job(
    *,
    agent_service: AgentService,
    session_id: str,
    task: str,
    user: str,
    files,
) -> str:
    """
    Start an agent job with file data stored in memory.
    
    Reads file contents before scheduling the background task to avoid
    closed file handle errors when FastAPI cleans up the request.
    
    Args:
        agent_service: Agent service instance.
        session_id: Session identifier.
        task: Task description.
        user: User identifier.
        files: Optional list of FastAPI UploadFile objects.
    
    Returns:
        Generated job ID.
    """
    # Read files into memory before request completes
    job_files = None
    if files:
        job_files = await _read_upload_files(files)
    
    job_id = create_job(user, job_files)

    asyncio.create_task(
        _run_job(
            job_id,
            agent_service=agent_service,
            session_id=session_id,
            task=task,
            user=user,
        )
    )

    logger.info(f"[Agent Job] Job {job_id} scheduled")
    return job_id