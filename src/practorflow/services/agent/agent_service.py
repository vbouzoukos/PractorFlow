"""
Multi-agent task execution service.

Implements a streamlined plan → execute → synthesize pipeline:
- Planner agent: Decomposes tasks into structured plans
- Executor agent: Executes steps using tools with LLM reasoning
- Synthesizer agent: Combines tool outputs into final answer
- Verifier agent: Validates execution (not content quality)

Integrates with existing infrastructure:
- ModelPool for LLM access
- KnowledgeStore for document search
- SessionStore for persistence
- ToolRegistry for tool management
"""

import uuid
from typing import Any, Dict, List, Optional, Set

from pydantic import ValidationError
from pydantic_ai import Agent

from practorflow.llm.pool.model_pool import ModelPool
from practorflow.llm.factory import create_runner
from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.base.session import Session, Message
from practorflow.llm.base.session_store import SessionStore
from practorflow.llm.pyai.model import LocalLLMModel
from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.services.dto.chat_file import ChatFile
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.agent.schemas import (
    Plan,
    ExecutionResult,
    StepStatus,
    VerificationResult,
    VerificationStatus,
    AgentTaskResult,
)
from practorflow.services.agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    build_planner_prompt,
    build_executor_prompt,
    build_verifier_prompt,
    build_synthesis_prompt,
)
from practorflow.services.agent.deps import AgentDeps
from practorflow.services.agent.tools import register_default_tools, register_executor_tools
from practorflow.services.agent.parser import parse_json_from_response, parse_executor_results
from practorflow.services.agent.verification import heuristic_verification
from practorflow.services.agent.session_utils import (
    get_document_context,
    get_document_scope,
    persist_to_session,
    build_execution_log,
    extract_final_output,
    build_failure_message,
)

logger = get_logger("agent_service", level=appConfiguration.LoggerConfiguration.AgentLevel)


class AgentService:
    """
    Multi-agent task execution service.

    Orchestrates a pipeline:
    1. Planning: Decompose task into steps with tools
    2. Execution: Execute steps in order using LLM reasoning and tools
    3. Synthesis: Combine outputs into final answer
    4. Verification: Validate execution happened (not content quality)

    Always returns synthesized output to user, verification is for logging only.
    """

    def __init__(
        self,
        model_pool: ModelPool,
        model_config: LLMConfig,
        knowledge_store: KnowledgeStore,
        session_store: SessionStore,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """
        Initialize agent service.

        Args:
            model_pool: Pool for acquiring LLM handles.
            model_config: Configuration for the LLM model.
            knowledge_store: Store for document search.
            session_store: Store for session persistence.
            tool_registry: Optional pre-configured tool registry.
        """
        self._model_pool = model_pool
        self._model_config = model_config
        self._knowledge_store = knowledge_store
        self._session_store = session_store
        self._tool_registry = tool_registry or ToolRegistry()

        register_default_tools(self._tool_registry, self._knowledge_store)

        logger.info("[AgentService] Initialized")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID for agent tasks."""
        return f"agent_{uuid.uuid4().hex}"

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

        Always returns a graceful response to the user, even if verification
        fails. Verification status is logged for debugging but never blocks
        the synthesized output from being returned.

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

        if self._session_store.exists(session_id):
            session = self._session_store.get(session_id)
        else:
            session = Session(
                session_id=session_id,
                instructions="Multi-agent task execution",
                user=user,
                metadata={"type": "agent"},
            )
            logger.info(f"[AgentService] Created new session: {session_id}")

        if files:
            for file in files:
                doc_info = await self._index_file(file)
                session.add_document(doc_info)
                logger.info(f"[AgentService] Indexed file: {doc_info['filename']}")

        document_scope = get_document_scope(session, document_ids)
        document_context = get_document_context(session, document_ids)

        session.messages.append(Message(role="user", content=task))

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        plan: Optional[Plan] = None
        execution_result: Optional[ExecutionResult] = None
        verification_result: Optional[VerificationResult] = None

        try:
            plan = await self._run_planner(task, document_context)
        except ValueError as e:
            error_msg = f"Planning failed: {e}"
            logger.error(f"[AgentService] {error_msg}")
            session.messages.append(Message(role="assistant", content=error_msg))
            persist_to_session(session, self._session_store, None, None, None)
            return AgentTaskResult(
                success=False,
                error=error_msg,
                usage=total_usage,
            )

        persist_to_session(session, self._session_store, plan, None, None)

        deps = AgentDeps(
            knowledge_store=self._knowledge_store,
            tool_registry=self._tool_registry,
            document_scope=document_scope,
        )

        retries = 0
        max_retries = plan.retry_policy.max_retries

        while True:
            execution_result = await self._run_executor(plan, deps)

            # Synthesize final answer from tool outputs
            synthesized_output = await self._run_synthesizer(plan, execution_result)
            execution_result.synthesized_output = synthesized_output

            persist_to_session(session, self._session_store, plan, execution_result, None)

            verification_result = await self._run_verifier(plan, execution_result)

            persist_to_session(session, self._session_store, plan, execution_result, verification_result)

            # PASSED or PARTIAL are both acceptable outcomes
            if verification_result.verification_status in (
                VerificationStatus.PASSED,
                VerificationStatus.PARTIAL,
            ):
                logger.info(
                    f"[AgentService] Verification {verification_result.verification_status}"
                )
                break

            # Only FAILED status triggers retry
            if verification_result.retry_recommended and retries < max_retries:
                retries += 1
                logger.info(f"[AgentService] Retrying execution (attempt {retries}/{max_retries})")
                continue

            logger.info(
                f"[AgentService] Verification {verification_result.verification_status}, "
                f"no more retries"
            )
            break

        # Always return synthesized output gracefully - never block user with verification errors
        output = extract_final_output(plan, execution_result)

        # Log verification issues for debugging without blocking response
        if verification_result.verification_status == VerificationStatus.FAILED:
            failure_details = build_failure_message(verification_result)
            logger.warning(
                f"[AgentService] Verification failed but returning synthesized output. "
                f"Details:\n{failure_details}"
            )
        else:
            logger.info(
                f"[AgentService] Verification {verification_result.verification_status}"
            )

        session.messages.append(Message(role="assistant", content=output))
        persist_to_session(session, self._session_store, plan, execution_result, verification_result)

        return AgentTaskResult(
            success=True,
            output=output,
            plan=plan,
            execution_result=execution_result,
            verification_result=verification_result,
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

    async def delete_task(self, session_id: str) -> bool:
        """
        Delete a task session and associated documents.

        Args:
            session_id: Session identifier.

        Returns:
            True if deleted, False if not found.
        """
        if not self._session_store.exists(session_id):
            logger.warning(f"[AgentService] Session not found: {session_id}")
            return False

        session = self._session_store.get(session_id)

        for doc in session.documents:
            doc_id = doc.get("id")
            if doc_id:
                try:
                    self._knowledge_store.delete_document(doc_id)
                    logger.debug(f"[AgentService] Deleted document: {doc_id}")
                except Exception as e:
                    logger.warning(f"[AgentService] Failed to delete document {doc_id}: {e}")

        self._session_store.delete(session_id)
        logger.info(f"[AgentService] Deleted session: {session_id}")

        return True

    async def _run_planner(
        self,
        task: str,
        document_context: Optional[str],
    ) -> Plan:
        """
        Run the Planner agent to create an execution plan.

        Args:
            task: User task to plan.
            document_context: Description of available documents.

        Returns:
            Validated Plan object.

        Raises:
            ValueError: If planning fails or output is invalid.
        """
        tools_metadata = self._tool_registry.get_schemas()

        # Filter out knowledge_search if no documents available
        if document_context is None:
            tools_metadata = [
                t for t in tools_metadata
                if t.get("function", {}).get("name") != "knowledge_search"
            ]
            logger.debug("[AgentService] No documents - excluded knowledge_search from tools")

        prompt = build_planner_prompt(task, tools_metadata, document_context)

        logger.debug(f"[AgentService] Running planner for task: {task[:100]}...")

        async with self._model_pool.acquire_context(self._model_config) as handle:
            runner = create_runner(handle, knowledge_store=self._knowledge_store)
            model = LocalLLMModel(runner, system_prompt=PLANNER_SYSTEM_PROMPT)

            agent = Agent(
                model=model,
                deps_type=AgentDeps,
                system_prompt=PLANNER_SYSTEM_PROMPT,
            )

            result = await agent.run(prompt)
            response_text = result.output if isinstance(result.output, str) else str(result.output)

        parsed = parse_json_from_response(response_text)
        if not parsed:
            logger.error(f"[AgentService] Failed to parse planner response: {response_text[:500]}")
            raise ValueError("Planner did not return valid JSON")

        try:
            plan = Plan.model_validate(parsed)
        except ValidationError as e:
            logger.error(f"[AgentService] Invalid plan structure: {e}")
            raise ValueError(f"Invalid plan structure: {e}")

        logger.info(f"[AgentService] Plan created: {plan.plan_id} with {len(plan.steps)} steps")
        return plan

    async def _run_executor(
        self,
        plan: Plan,
        deps: AgentDeps,
    ) -> ExecutionResult:
        """
        Run the Executor agent to execute the plan using LLM reasoning.

        The executor uses the LLM to reason through each step, calling tools
        as needed and synthesizing results.

        Args:
            plan: Plan to execute.
            deps: Dependencies including tools and document scope.

        Returns:
            ExecutionResult with step outcomes.
        """
        logger.debug(f"[AgentService] Executing plan: {plan.plan_id}")

        prompt = build_executor_prompt(plan)

        async with self._model_pool.acquire_context(self._model_config) as handle:
            runner = create_runner(handle, knowledge_store=self._knowledge_store)
            model = LocalLLMModel(runner, system_prompt=EXECUTOR_SYSTEM_PROMPT)

            agent = Agent(
                model=model,
                deps_type=AgentDeps,
                system_prompt=EXECUTOR_SYSTEM_PROMPT,
            )

            register_executor_tools(agent, deps)

            async with agent.iter(prompt, deps=deps) as agent_run:
                async for node in agent_run:
                    logger.debug(
                        f"[AgentService][Executor] node={getattr(node, 'name', type(node).__name__)}"
                    )

                response_text = ""
                if agent_run.result:
                    response_text = (
                        agent_run.result.output
                        if isinstance(agent_run.result.output, str)
                        else str(agent_run.result.output)
                    )

        step_results = parse_executor_results(plan, response_text, deps)
        execution_log = build_execution_log(step_results, plan)

        execution_result = ExecutionResult(
            plan_id=plan.plan_id,
            step_results=step_results,
            execution_log=execution_log,
        )

        success_count = sum(1 for r in step_results if r.status == StepStatus.SUCCESS)
        logger.info(
            f"[AgentService] Execution complete: {success_count}/{len(step_results)} steps succeeded"
        )

        return execution_result

    async def _run_synthesizer(
        self,
        plan: Plan,
        execution_result: ExecutionResult,
    ) -> str:
        """
        Run the Synthesizer agent to create final answer from tool outputs.

        Args:
            plan: The original plan with user's task.
            execution_result: Results from execution with tool outputs.

        Returns:
            Synthesized final answer string.
        """
        prompt = build_synthesis_prompt(plan.task, execution_result)

        logger.debug(f"[AgentService] Running synthesizer for plan: {plan.plan_id}")

        async with self._model_pool.acquire_context(self._model_config) as handle:
            runner = create_runner(handle, knowledge_store=self._knowledge_store)
            model = LocalLLMModel(runner, system_prompt=SYNTHESIZER_SYSTEM_PROMPT)

            agent = Agent(
                model=model,
                deps_type=AgentDeps,
                system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            )

            result = await agent.run(prompt)
            synthesized = result.output if isinstance(result.output, str) else str(result.output)

        logger.info(f"[AgentService] Synthesis complete: {len(synthesized)} chars")
        return synthesized

    async def _run_verifier(
        self,
        plan: Plan,
        execution_result: ExecutionResult,
    ) -> VerificationResult:
        """
        Run the Verifier agent to validate execution results.

        Args:
            plan: The original plan.
            execution_result: Results from execution.

        Returns:
            VerificationResult with validation outcome.
        """
        prompt = build_verifier_prompt(plan, execution_result)

        logger.debug(f"[AgentService] Running verifier for plan: {plan.plan_id}")

        async with self._model_pool.acquire_context(self._model_config) as handle:
            runner = create_runner(handle, knowledge_store=self._knowledge_store)
            model = LocalLLMModel(runner, system_prompt=VERIFIER_SYSTEM_PROMPT)

            agent = Agent(
                model=model,
                deps_type=AgentDeps,
                system_prompt=VERIFIER_SYSTEM_PROMPT,
            )

            result = await agent.run(prompt)
            response_text = result.output if isinstance(result.output, str) else str(result.output)

        parsed = parse_json_from_response(response_text)
        if not parsed:
            logger.warning("[AgentService] Verifier did not return valid JSON, using heuristic")
            return heuristic_verification(plan, execution_result)

        try:
            verification = VerificationResult.model_validate(parsed)
        except ValidationError as e:
            logger.warning(f"[AgentService] Invalid verification structure: {e}, using heuristic")
            return heuristic_verification(plan, execution_result)

        logger.info(f"[AgentService] Verification complete: {verification.verification_status}")
        return verification