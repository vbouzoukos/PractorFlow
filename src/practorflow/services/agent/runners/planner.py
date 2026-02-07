"""
Planner and AdaptPlan agent runners.

- run_planner: Decomposes tasks into structured execution plans
- adapt_plan: Analyzes failures and creates corrected plans
"""

from pydantic import ValidationError
from pydantic_ai import Agent

from practorflow.llm.pool.model_pool import ModelPool
from practorflow.llm.factory import create_runner
from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.pyai.model import LocalLLMModel
from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.agent.schemas import (
    Plan,
    ExecutionResult,
    VerificationResult,
)
from practorflow.services.agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    ADAPT_PLAN_SYSTEM_PROMPT,
    build_planner_prompt,
    build_adapt_plan_prompt,
)
from practorflow.services.agent.context import ExecutionContext
from practorflow.services.agent.deps import AgentDeps
from practorflow.services.agent.parser import parse_json_from_response
from practorflow.services.agent.json_helpers import repair_plan_json
from practorflow.services.agent.runners.agent_context import prepare_agent_history

logger = get_logger(
    "agent_planner", level=appConfiguration.LoggerConfiguration.AgentLevel
)


async def run_planner(
    task: str,
    ctx: ExecutionContext,
    model_pool: ModelPool,
    model_config: LLMConfig,
    knowledge_store: KnowledgeStore,
    tool_registry: ToolRegistry,
) -> Plan:
    """
    Run the Planner agent to create an execution plan.

    Args:
        task: User task to plan.
        ctx: Execution context with session state and message history.
        model_pool: Pool for acquiring LLM handles.
        model_config: Configuration for the LLM model.
        knowledge_store: Store for document search.
        tool_registry: Registry containing available tools.

    Returns:
        Validated Plan object.

    Raises:
        ValueError: If planning fails or output is invalid.
    """
    tools_metadata = tool_registry.get_schemas()

    if not ctx.has_documents:
        tools_metadata = [
            t
            for t in tools_metadata
            if t.get("function", {}).get("name") != "knowledge_search"
        ]
        logger.debug("[Planner] No documents - excluded knowledge_search from tools")

    prompt = build_planner_prompt(task, tools_metadata, ctx.document_context)

    prepared_history = await prepare_agent_history(
        task=task,
        ctx=ctx,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        task_prompt=prompt,
        model_pool=model_pool,
        model_config=model_config,
    )

    logger.debug(f"[Planner] Running planner for task: {task[:100]}...")
    if prepared_history:
        logger.debug(f"[Planner] Using {len(prepared_history)} history messages")

    async with model_pool.acquire_context(model_config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        model = LocalLLMModel(runner)

        agent = Agent(
            model=model,
            deps_type=AgentDeps,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

        result = await agent.run(prompt, message_history=prepared_history)
        response_text = (
            result.output if isinstance(result.output, str) else str(result.output)
        )

    parsed = parse_json_from_response(response_text)
    if not parsed:
        logger.warning("[Planner] Standard parsing failed, attempting JSON repair")
        parsed = repair_plan_json(response_text)

    if not parsed:
        logger.error(
            f"[Planner] Failed to parse planner response: {response_text[:500]}"
        )
        raise ValueError("Planner did not return valid JSON")

    try:
        plan = Plan.model_validate(parsed)
    except ValidationError as e:
        logger.error(f"[Planner] Invalid plan structure: {e}")
        raise ValueError(f"Invalid plan structure: {e}")

    logger.info(f"[Planner] Plan created: {plan.plan_id} with {len(plan.steps)} steps")
    return plan


async def adapt_plan(
    task: str,
    failed_plan: Plan,
    execution_result: ExecutionResult,
    verification_result: VerificationResult,
    ctx: ExecutionContext,
    model_pool: ModelPool,
    model_config: LLMConfig,
    knowledge_store: KnowledgeStore,
    tool_registry: ToolRegistry,
    attempt_number: int,
) -> Plan:
    """
    Run the Replanner agent to create a corrected plan after verification failure.

    Analyzes the failure context and generates a new plan that addresses
    the specific issues found during verification.

    Args:
        task: Original user task.
        failed_plan: The plan that failed verification.
        execution_result: Results from the failed execution.
        verification_result: Verification result with failure details.
        ctx: Execution context with session state and message history.
        model_pool: Pool for acquiring LLM handles.
        model_config: Configuration for the LLM model.
        knowledge_store: Store for document search.
        tool_registry: Registry containing available tools.
        attempt_number: Current replan attempt number.

    Returns:
        Validated Plan object with corrected steps.

    Raises:
        ValueError: If replanning fails or output is invalid.
    """
    tools_metadata = tool_registry.get_schemas()

    if not ctx.has_documents:
        tools_metadata = [
            t
            for t in tools_metadata
            if t.get("function", {}).get("name") != "knowledge_search"
        ]
        logger.debug("[AdaptPlan] No documents - excluded knowledge_search from tools")

    prompt = build_adapt_plan_prompt(
        task=task,
        failed_plan=failed_plan,
        execution_result=execution_result,
        verification_result=verification_result,
        tools_metadata=tools_metadata,
        attempt_number=attempt_number,
        document_context=ctx.document_context,
    )

    prepared_history = await prepare_agent_history(
        task=task,
        ctx=ctx,
        system_prompt=ADAPT_PLAN_SYSTEM_PROMPT,
        task_prompt=prompt,
        model_pool=model_pool,
        model_config=model_config,
    )

    logger.debug(f"[AdaptPlan] Running replanner for task: {task[:100]}...")
    if prepared_history:
        logger.debug(f"[AdaptPlan] Using {len(prepared_history)} history messages")

    async with model_pool.acquire_context(model_config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        model = LocalLLMModel(runner)

        agent = Agent(
            model=model,
            deps_type=AgentDeps,
            system_prompt=ADAPT_PLAN_SYSTEM_PROMPT,
        )

        result = await agent.run(prompt, message_history=prepared_history)
        response_text = (
            result.output if isinstance(result.output, str) else str(result.output)
        )

    parsed = parse_json_from_response(response_text)
    if not parsed:
        logger.warning("[AdaptPlan] Standard parsing failed, attempting JSON repair")
        parsed = repair_plan_json(response_text)

    if not parsed:
        logger.error(
            f"[AdaptPlan] Failed to parse replanner response: {response_text[:500]}"
        )
        raise ValueError("Replanner did not return valid JSON")

    try:
        plan = Plan.model_validate(parsed)
    except ValidationError as e:
        logger.error(f"[AdaptPlan] Invalid plan structure: {e}")
        raise ValueError(f"Invalid plan structure: {e}")

    logger.info(f"[AdaptPlan] Corrected plan created: {plan.plan_id} with {len(plan.steps)} steps")
    return plan