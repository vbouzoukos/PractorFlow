"""
Executor agent runner.

- run_executor: Executes plan steps using tools with LLM reasoning
"""

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
    StepStatus,
)
from practorflow.services.agent.prompts import (
    EXECUTOR_SYSTEM_PROMPT,
    build_executor_prompt,
)
from practorflow.services.agent.context import ExecutionContext
from practorflow.services.agent.deps import AgentDeps
from practorflow.services.agent.parser import parse_executor_results
from practorflow.services.agent.session_utils import build_execution_log

from practorflow.services.tools.registration import build_tools_for_registry
from practorflow.services.agent.runners.agent_context import prepare_agent_history

logger = get_logger(
    "agent_executor", level=appConfiguration.LoggerConfiguration.AgentLevel
)


async def run_executor(
    plan: Plan,
    ctx: ExecutionContext,
    model_pool: ModelPool,
    model_config: LLMConfig,
    knowledge_store: KnowledgeStore,
    tool_registry: ToolRegistry,
) -> ExecutionResult:
    """
    Run the Executor agent to execute the plan using LLM reasoning.

    The executor uses the LLM to reason through each step, calling tools
    as needed and synthesizing results.

    Args:
        plan: Plan to execute.
        ctx: Execution context with session state and message history.
        model_pool: Pool for acquiring LLM handles.
        model_config: Configuration for the LLM model.
        knowledge_store: Store for document search.
        tool_registry: Registry containing available tools.

    Returns:
        ExecutionResult with step outcomes.
    """
    logger.debug(f"[Executor] Executing plan: {plan.plan_id}")

    prompt = build_executor_prompt(plan)

    prepared_history = await prepare_agent_history(
        task=plan.task,
        ctx=ctx,
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        task_prompt=prompt,
        model_pool=model_pool,
        model_config=model_config,
    )

    if prepared_history:
        logger.debug(f"[Executor] Using {len(prepared_history)} history messages")

    deps = AgentDeps(
        knowledge_store=knowledge_store,
        tool_registry=tool_registry,
        document_scope=ctx.document_scope,
    )

    tools = build_tools_for_registry(tool_registry)
    mcp_toolsets = tool_registry.get_mcp_toolsets()

    async with model_pool.acquire_context(model_config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        model = LocalLLMModel(runner)

        agent = Agent(
            model=model,
            deps_type=AgentDeps,
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
            tools=tools,
            toolsets=mcp_toolsets,
        )

        async with agent.iter(
            prompt,
            deps=deps,
            message_history=prepared_history,
        ) as agent_run:
            async for node in agent_run:
                logger.debug(
                    f"[Executor] node={getattr(node, 'name', type(node).__name__)}"
                )

            response_text = ""
            if agent_run.result:
                response_text = (
                    agent_run.result.output
                    if isinstance(agent_run.result.output, str)
                    else str(agent_run.result.output)
                )

    step_results = await parse_executor_results(plan, deps)
    execution_log = build_execution_log(step_results, plan)

    execution_result = ExecutionResult(
        plan_id=plan.plan_id,
        step_results=step_results,
        execution_log=execution_log,
    )

    success_count = sum(1 for r in step_results if r.status == StepStatus.SUCCESS)
    logger.info(
        f"[Executor] Execution complete: {success_count}/{len(step_results)} steps succeeded"
    )

    return execution_result