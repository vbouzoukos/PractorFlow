"""
Synthesizer agent runner.

- run_synthesizer: Creates final answer from tool outputs (with persona extraction)
"""

from typing import Optional

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
)
from practorflow.services.agent.prompts import (
    SYNTHESIZER_SYSTEM_PROMPT,
    build_synthesis_prompt,
)
from practorflow.services.agent.context import ExecutionContext
from practorflow.services.agent.deps import AgentDeps
from practorflow.services.agent.runners.agent_context import (
    prepare_agent_history,
    extract_context,
    build_context_enhanced_prompt,
)

logger = get_logger(
    "agent_synthesizer", level=appConfiguration.LoggerConfiguration.AgentLevel
)


async def run_synthesizer(
    plan: Plan,
    execution_result: ExecutionResult,
    ctx: ExecutionContext,
    model_pool: ModelPool,
    model_config: LLMConfig,
    knowledge_store: KnowledgeStore,
    tool_registry: ToolRegistry,
    user_instructions: Optional[str] = None,
) -> str:
    """
    Run the Synthesizer agent to create final answer from tool outputs.

    Uses two-step agentic approach:
    1. Extract persona from conversation history
    2. Generate synthesized response with persona context

    Args:
        plan: The original plan with user's task.
        execution_result: Results from execution with tool outputs.
        ctx: Execution context with session state and message history.
        model_pool: Pool for acquiring LLM handles.
        model_config: Configuration for the LLM model.
        knowledge_store: Store for document search.
        tool_registry: Tool registry for agent dependencies.
        user_instructions: Optional user instructions for response synthesis.

    Returns:
        Synthesized final answer string.
    """
    prompt = build_synthesis_prompt(plan.task, execution_result)

    if user_instructions:
        system_prompt = f"{SYNTHESIZER_SYSTEM_PROMPT}\n\n<user_instructions>\n{user_instructions}\n</user_instructions>"
    else:
        system_prompt = SYNTHESIZER_SYSTEM_PROMPT

    prepared_history = await prepare_agent_history(
        task=plan.task,
        ctx=ctx,
        system_prompt=system_prompt,
        task_prompt=prompt,
        model_pool=model_pool,
        model_config=model_config,
    )

    logger.debug(f"[Synthesizer] Running synthesizer for plan: {plan.plan_id}")
    if prepared_history:
        logger.debug(f"[Synthesizer] Using {len(prepared_history)} history messages")

    async with model_pool.acquire_context(model_config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        model = LocalLLMModel(runner)

        # Step 1: Extract persona and instructions from history
        persona, instructions = await extract_context(prepared_history, model, knowledge_store, tool_registry)

        # Step 2: Enhance prompt with persona/instructions if detected
        if persona or instructions:
            prompt = build_context_enhanced_prompt(persona, instructions, prompt)
            logger.debug(f"[Synthesizer] Applied context - persona: {persona}, instructions: {instructions}")

        # Step 3: Generate synthesized response with full context
        agent = Agent(
            model=model,
            deps_type=AgentDeps,
            system_prompt=system_prompt,
        )

        async with agent.iter(
            prompt,
            deps=AgentDeps(knowledge_store=knowledge_store, tool_registry=tool_registry),
            message_history=prepared_history,
        ) as agent_run:
            async for node in agent_run:
                logger.debug(
                    f"[Synthesizer] node={getattr(node, 'name', type(node).__name__)}"
                ) # pragma: no cover

            synthesized = ""
            if agent_run.result:
                synthesized = (
                    agent_run.result.output
                    if isinstance(agent_run.result.output, str)
                    else str(agent_run.result.output)
                )

    logger.info(f"[Synthesizer] Synthesis complete: {len(synthesized)} chars")
    return synthesized