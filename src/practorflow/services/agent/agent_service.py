"""
Multi-agent task execution service.

Implements a deterministic plan → execute → verify pipeline:
- Planner agent: Decomposes tasks into structured plans
- Executor agent: Executes steps using tools with LLM reasoning
- Verifier agent: Validates results against success criteria

Integrates with existing infrastructure:
- ModelPool for LLM access
- KnowledgeStore for document search
- SessionStore for persistence
- ToolRegistry for tool management
"""

import uuid
from typing import Any, Dict, List, Optional, Set

from practorflow.llm.pool.model_pool import ModelPool
from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.base.session import Session, Message
from practorflow.llm.base.session_store import SessionStore
from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.services.dto.chat_file import ChatFile
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.agent.schemas import (
    Plan,
    ExecutionResult,
    VerificationResult,
    VerificationStatus,
    AgentTaskResult,
)
from practorflow.services.agent.context import build_execution_context
from practorflow.services.agent.session_utils import (
    persist_to_session,
    extract_final_output,
    build_failure_message,
)
from practorflow.services.agent.runners import (
    run_planner,
    run_executor,
    run_synthesizer,
    run_verifier,
    adapt_plan,
)
from practorflow.services.tools.registration import (
    register_default_tools,
    load_api_tools_for_user,
)

logger = get_logger(
    "agent_service", level=appConfiguration.LoggerConfiguration.AgentLevel
)


class AgentService:
    """
    Multi-agent task execution service.

    Orchestrates a three-phase pipeline:
    1. Planning: Decompose task into steps with tools and success criteria
    2. Execution: Execute steps in order using LLM reasoning and tools
    3. Verification: Validate results against criteria

    Supports retry logic and persists state to sessions.
    """

    def __init__(
        self,
        model_pool: ModelPool,
        model_config: LLMConfig,
        knowledge_store: KnowledgeStore,
        session_store: SessionStore,
        tool_registry: Optional[ToolRegistry] = None,
        user_instructions: Optional[str] = None,
        mcp_server_store=None,
    ):
        """
        Initialize agent service.

        Args:
            model_pool: Pool for acquiring LLM handles.
            model_config: Configuration for the LLM model.
            knowledge_store: Store for document search.
            session_store: Store for session persistence.
            tool_registry: Optional pre-configured tool registry.
            user_instructions: Optional user instructions for response synthesis.
        """
        self._model_pool = model_pool
        self._model_config = model_config
        self._knowledge_store = knowledge_store
        self._session_store = session_store
        self._tool_registry = tool_registry or ToolRegistry()
        self._user_instructions = user_instructions

        if mcp_server_store is not None:
            self._tool_registry.set_mcp_server_store(mcp_server_store)

        register_default_tools(self._tool_registry, self._knowledge_store)

        logger.info("[AgentService] Initialized")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{uuid.uuid4().hex}"

    async def _index_file(self, file: ChatFile) -> Dict[str, Any]:
        """
        Index a file into the knowledge store.

        Args:
            file: File to index.

        Returns:
            Document info dict with id and filename.
        """
        doc_info = self._knowledge_store.add_document_from_stream(
            file_stream=file.file,
            filename=file.filename,
            mime_type=file.content_type,
        )
        return doc_info

    def _get_or_create_session(self, session_id: str, user: str) -> Session:
        """
        Get existing session or create a new one.

        Args:
            session_id: Session identifier.
            user: User identifier for new sessions.

        Returns:
            Session object.
        """
        if self._session_store.exists(session_id):
            return self._session_store.get(session_id)

        session = Session(
            session_id=session_id,
            instructions="Multi-agent task execution",
            user=user,
        )
        logger.info(f"[AgentService] Created new session: {session_id}")
        return session

    async def start_task(self) -> str:
        """
        Start a new agent task session.

        Returns:
            Generated session ID.
        """
        session_id = self._generate_session_id()
        logger.info(f"[AgentService] Generated session ID: {session_id}")
        return session_id

    async def execute_task(
        self,
        session_id: str,
        task: str,
        user: str,
        files: Optional[List[ChatFile]] = None,
        document_ids: Optional[Set[str]] = None,
    ) -> AgentTaskResult:
        """
        Execute a task using the multi-agent pipeline.

        Args:
            session_id: Session identifier.
            task: User task description.
            user: User identifier.
            files: Optional files to upload and index.
            document_ids: Optional pre-existing document IDs.

        Returns:
            AgentTaskResult with outcome and artifacts.
        """
        logger.info(f"[AgentService] Executing task for session: {session_id}")

        session = self._get_or_create_session(session_id, user)

        # Load API tools for user
        load_api_tools_for_user(self._tool_registry, user)

        if files:
            for file in files:
                doc_info = await self._index_file(file)
                session.add_document(doc_info)
                logger.info(f"[AgentService] Indexed file: {doc_info['filename']}")

        session.messages.append(Message(role="user", content=task))

        ctx = build_execution_context(session, document_ids)
        if ctx.has_history:
            logger.debug(
                f"[AgentService] Context built with {ctx.history_length} history messages"
            )

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        plan: Optional[Plan] = None
        execution_result: Optional[ExecutionResult] = None
        verification_result: Optional[VerificationResult] = None

        try:
            plan = await run_planner(
                task=task,
                ctx=ctx,
                model_pool=self._model_pool,
                model_config=self._model_config,
                knowledge_store=self._knowledge_store,
                tool_registry=self._tool_registry,
            )
        except ValueError as e:
            error_msg = f"Planning failed: {e}"
            logger.error(f"[AgentService] {error_msg}")
            return AgentTaskResult(
                success=False,
                output=None,
                plan=None,
                execution_result=None,
                verification_result=None,
                error=error_msg,
                usage=total_usage,
            )

        retries = 0
        max_retries = plan.retry_policy.max_retries

        while True:
            execution_result = await run_executor(
                plan=plan,
                ctx=ctx,
                model_pool=self._model_pool,
                model_config=self._model_config,
                knowledge_store=self._knowledge_store,
                tool_registry=self._tool_registry,
            )

            synthesized_output = await run_synthesizer(
                plan=plan,
                execution_result=execution_result,
                ctx=ctx,
                model_pool=self._model_pool,
                model_config=self._model_config,
                knowledge_store=self._knowledge_store,
                tool_registry=self._tool_registry,
                user_instructions=self._user_instructions,
            )
            execution_result.synthesized_output = synthesized_output

            verification_result = await run_verifier(
                plan=plan,
                execution_result=execution_result,
                model_pool=self._model_pool,
                model_config=self._model_config,
                knowledge_store=self._knowledge_store,
            )

            if verification_result.verification_status in (
                VerificationStatus.PASSED,
                VerificationStatus.PARTIAL,
            ):
                logger.info(
                    f"[AgentService] Verification {verification_result.verification_status}"
                )
                break

            if verification_result.retry_recommended and retries < max_retries:
                retries += 1
                logger.info(
                    f"[AgentService] Retrying execution (attempt {retries}/{max_retries})"
                )

                try:
                    new_plan = await adapt_plan(
                        task=task,
                        failed_plan=plan,
                        execution_result=execution_result,
                        verification_result=verification_result,
                        ctx=ctx,
                        model_pool=self._model_pool,
                        model_config=self._model_config,
                        knowledge_store=self._knowledge_store,
                        tool_registry=self._tool_registry,
                        attempt_number=retries,
                    )
                    plan = new_plan
                    logger.info(
                        f"[AgentService] Adapted plan: {plan.plan_id} with {len(plan.steps)} steps"
                    )
                except Exception as e:
                    logger.warning(
                        f"[AgentService] adapt_plan failed: {e}, continuing with original plan"
                    )

                continue

            logger.info(
                f"[AgentService] Verification {verification_result.verification_status}, "
                f"no more retries"
            )
            break

        if verification_result.verification_status in (
            VerificationStatus.PASSED,
            VerificationStatus.PARTIAL,
        ):
            output = extract_final_output(plan, execution_result)
            session.messages.append(Message(role="assistant", content=output))
            await persist_to_session(
                session=session,
                session_store=self._session_store,
                plan=plan,
                execution_result=execution_result,
                verification_result=verification_result,
                model_pool=self._model_pool,
                model_config=self._model_config,
            )

            return AgentTaskResult(
                success=True,
                output=output,
                plan=plan,
                execution_result=execution_result,
                verification_result=verification_result,
                usage=total_usage,
            )

        error_msg = build_failure_message(verification_result)

        return AgentTaskResult(
            success=False,
            output=None,
            plan=plan,
            execution_result=execution_result,
            verification_result=verification_result,
            error=error_msg,
            usage=total_usage,
        )

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session if found, None otherwise.
        """
        if not self._session_store.exists(session_id):
            return None
        return self._session_store.get(session_id)