"""
Agent runner functions for multi-agent pipeline.

Contains the individual agent runners for each pipeline phase:
- Planner: Decomposes tasks into structured plans
- Executor: Executes steps using tools with LLM reasoning
- Synthesizer: Creates final answer from tool outputs (with persona extraction)
- Verifier: Validates results against success criteria

Uses shared history library for context window management.
"""

from typing import List, Optional

from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

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
    VerificationResult,
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
from practorflow.services.agent.context import ExecutionContext
from practorflow.services.agent.deps import AgentDeps
from practorflow.services.agent.parser import (
    parse_json_from_response,
    parse_executor_results,
)
from practorflow.services.agent.json_helpers import repair_plan_json
from practorflow.services.agent.verification import heuristic_verification
from practorflow.services.agent.session_utils import build_execution_log

from practorflow.services.tools.registration import build_tools_for_registry
from practorflow.services.history.types import HistoryConfig
from practorflow.services.history.preparer import prepare_history

logger = get_logger(
    "agent_runners", level=appConfiguration.LoggerConfiguration.AgentLevel
)


# Prompt for persona and instructions extraction from conversation history
_CONTEXT_EXTRACTION_PROMPT = """Analyze the conversation history and extract TWO things:

1. PERSONA: Any requested persona, character, tone, or role for the assistant.
   Look for patterns like:
   - "act as...", "be a...", "pretend you are...", "respond like..."
   - "you are a...", "play the role of..."
   - Character requests: pirate, teacher, expert, conspiracist, etc.

2. INSTRUCTIONS: Any custom instructions the user has explicitly given.
   Extract the EXACT user instructions as they were written - do not summarize or paraphrase.
   Look for messages where the user gives explicit directives about how to respond.

Respond in EXACTLY this format:
PERSONA: <persona description or NONE>
INSTRUCTIONS_START
<exact user instructions copied verbatim, or NONE if no instructions found>
INSTRUCTIONS_END

Examples:

Example 1 - User said "act as a pirate":
PERSONA: pirate
INSTRUCTIONS_START
NONE
INSTRUCTIONS_END

Example 2 - User said "You are a formal assistant. Always respond in bullet points and include references.":
PERSONA: formal assistant
INSTRUCTIONS_START
Always respond in bullet points and include references.
INSTRUCTIONS_END

Example 3 - User said "Remember these rules: 1. Be concise 2. Use examples 3. Avoid jargon":
PERSONA: NONE
INSTRUCTIONS_START
1. Be concise 2. Use examples 3. Avoid jargon
INSTRUCTIONS_END

Example 4 - No persona or instructions found:
PERSONA: NONE
INSTRUCTIONS_START
NONE
INSTRUCTIONS_END

Extract and respond now."""


async def _prepare_history(
    task: str,
    ctx: ExecutionContext,
    system_prompt: str,
    task_prompt: str,
    model_pool: ModelPool,
    model_config: LLMConfig,
) -> List[ModelMessage]:
    """
    Prepare message history for LLM call.

    Delegates to shared history library for context window management.

    Args:
        task: Current task.
        ctx: Execution context with message history.
        system_prompt: System prompt.
        task_prompt: Task prompt.
        model_pool: Pool for LLM access.
        model_config: LLM configuration with n_ctx.

    Returns:
        List of ModelMessage to use as history.
    """
    if not ctx.has_history:
        return []

    config = HistoryConfig(n_ctx=model_config.n_ctx)

    prepared = await prepare_history(
        task=task,
        messages=ctx.message_history,
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        model_pool=model_pool,
        model_config=model_config,
        config=config,
    )

    if prepared.was_truncated:
        logger.info(
            f"[History] Reduced from {prepared.original_count} to "
            f"{prepared.included_count} messages ({prepared.estimated_tokens} tokens)"
        )

    return prepared.messages


async def _extract_context(
    message_history: List[ModelMessage],
    model: LocalLLMModel,
    knowledge_store: KnowledgeStore,
    tool_registry: ToolRegistry,
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract persona and instructions from conversation history using the agent.

    Args:
        message_history: Conversation history in pydantic_ai format.
        model: LocalLLMModel instance.
        knowledge_store: Knowledge store for AgentDeps.
        tool_registry: Tool registry for AgentDeps.

    Returns:
        Tuple of (persona, instructions) - either can be None if not detected.
    """
    if not message_history:
        return None, None

    tools = build_tools_for_registry(tool_registry)

    agent = Agent(
        model=model,
        deps_type=AgentDeps,
        system_prompt="You are a context extraction assistant. Analyze conversation history and extract any requested persona and instructions.",
        tools=tools,
    )

    try:
        async with agent.iter(
            _CONTEXT_EXTRACTION_PROMPT,
            deps=AgentDeps(knowledge_store=knowledge_store, tool_registry=tool_registry),
            message_history=message_history,
        ) as agent_run:
            async for node in agent_run:
                logger.debug(
                    f"[ContextExtract] node={getattr(node, 'name', type(node).__name__)}"
                )

            if agent_run.result:
                result = agent_run.result.output or ""
                result = result.strip()

                persona = None
                instructions = None

                # Parse persona
                for line in result.split("\n"):
                    line = line.strip()
                    if line.upper().startswith("PERSONA:"):
                        value = line[8:].strip()
                        if value.upper() != "NONE" and value:
                            persona = value
                        break

                # Parse instructions between markers
                if "INSTRUCTIONS_START" in result and "INSTRUCTIONS_END" in result:
                    start_idx = result.find("INSTRUCTIONS_START") + len("INSTRUCTIONS_START")
                    end_idx = result.find("INSTRUCTIONS_END")
                    if start_idx < end_idx:
                        instructions_text = result[start_idx:end_idx].strip()
                        if instructions_text.upper() != "NONE" and instructions_text:
                            instructions = instructions_text

                if persona:
                    logger.info(f"[ContextExtract] Extracted persona: {persona}")
                if instructions:
                    logger.info(f"[ContextExtract] Extracted instructions: {instructions[:100]}...")

                return persona, instructions

    except Exception as e:
        logger.warning(f"[ContextExtract] Context extraction failed: {e}")
        return None, None

    return None, None


def _build_context_enhanced_prompt(
    persona: Optional[str],
    instructions: Optional[str],
    prompt: str,
) -> str:
    """
    Build prompt with persona/instructions reminder injected.

    Args:
        persona: Extracted persona description (or None).
        instructions: Extracted instructions (or None).
        prompt: Original synthesis prompt.

    Returns:
        Prompt with persona/instructions reminder prepended.
    """
    parts = []
    
    if persona:
        parts.append(f'PERSONA ACTIVE: You are acting as "{persona}". Stay fully in character for your entire response. Do not break character with disclaimers or objective commentary.')
    
    if instructions:
        parts.append(f'USER INSTRUCTIONS: {instructions}')
    
    if parts:
        context = "[" + " | ".join(parts) + "]"
        return f"{context}\n\n{prompt}"
    
    return prompt


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

    prepared_history = await _prepare_history(
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

    prepared_history = await _prepare_history(
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

    async with model_pool.acquire_context(model_config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        model = LocalLLMModel(runner)

        agent = Agent(
            model=model,
            deps_type=AgentDeps,
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
            tools=tools,
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

    prepared_history = await _prepare_history(
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
        persona, instructions = await _extract_context(prepared_history, model, knowledge_store, tool_registry)

        # Step 2: Enhance prompt with persona/instructions if detected
        if persona or instructions:
            prompt = _build_context_enhanced_prompt(persona, instructions, prompt)
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


async def run_verifier(
    plan: Plan,
    execution_result: ExecutionResult,
    model_pool: ModelPool,
    model_config: LLMConfig,
    knowledge_store: KnowledgeStore,
) -> VerificationResult:
    """
    Run the Verifier agent to validate execution results.

    Args:
        plan: The original plan.
        execution_result: Results from execution.
        model_pool: Pool for acquiring LLM handles.
        model_config: Configuration for the LLM model.
        knowledge_store: Store for document search.

    Returns:
        VerificationResult with validation outcome.
    """
    prompt = build_verifier_prompt(plan, execution_result)

    logger.debug(f"[Verifier] Running verifier for plan: {plan.plan_id}")

    async with model_pool.acquire_context(model_config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        model = LocalLLMModel(runner)

        agent = Agent(
            model=model,
            deps_type=AgentDeps,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
        )

        result = await agent.run(prompt)
        response_text = (
            result.output if isinstance(result.output, str) else str(result.output)
        )

    parsed = parse_json_from_response(response_text)
    if not parsed:
        logger.warning("[Verifier] Verifier did not return valid JSON, using heuristic")
        return heuristic_verification(plan, execution_result)

    try:
        verification = VerificationResult.model_validate(parsed)
    except ValidationError as e:
        logger.warning(
            f"[Verifier] Invalid verification structure: {e}, using heuristic"
        )
        return heuristic_verification(plan, execution_result)

    logger.info(f"[Verifier] Verification complete: {verification.verification_status}")
    return verification