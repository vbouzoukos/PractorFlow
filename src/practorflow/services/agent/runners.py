"""
Agent runner functions for multi-agent pipeline.

Contains the individual agent runners for each pipeline phase:
- Planner: Decomposes tasks into structured plans
- Executor: Executes steps using tools with LLM reasoning
- Synthesizer: Creates final answer from tool outputs
- Verifier: Validates results against success criteria

Includes task-aware memory extraction when context exceeds n_ctx.
"""

from typing import List

from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    UserPromptPart,
)

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
from practorflow.services.agent.tools import register_executor_tools
from practorflow.services.agent.parser import parse_json_from_response, parse_executor_results
from practorflow.services.agent.json_helpers import repair_plan_json
from practorflow.services.agent.verification import heuristic_verification
from practorflow.services.agent.session_utils import build_execution_log

logger = get_logger("agent_runners", level=appConfiguration.LoggerConfiguration.AgentLevel)

CHARS_PER_TOKEN = 4

MEMORY_EXTRACTION_PROMPT = """You are a memory extraction assistant. Your task is to extract relevant information from conversation history that is needed to answer the current task.

CURRENT TASK:
{task}

CONVERSATION HISTORY:
{history}

Extract ONLY information from the history that is directly relevant to completing the current task. Include:
- Specific requirements or constraints mentioned
- Decisions already made
- Important names, numbers, dates, or technical details
- Context that affects how the task should be completed

If nothing in the history is relevant to the current task, respond with: "No relevant prior context."

Provide a concise summary of relevant context. Do not include irrelevant information."""


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text."""
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN


def _estimate_messages_tokens(messages: List[ModelMessage]) -> int:
    """Estimate total tokens in message history."""
    total = 0
    for msg in messages:
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if hasattr(part, 'content'):
                    total += _estimate_tokens(part.content)
    return total


def _messages_to_text(messages: List[ModelMessage]) -> str:
    """Convert messages to plain text for memory extraction."""
    parts = []
    for msg in messages:
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if hasattr(part, 'content'):
                    role = "User" if isinstance(msg, ModelRequest) else "Assistant"
                    parts.append(f"{role}: {part.content}")
    return "\n\n".join(parts)


def _estimate_total_context(
    system_prompt: str,
    task_prompt: str,
    messages: List[ModelMessage],
) -> int:
    """Estimate total tokens for the full context."""
    total = _estimate_tokens(system_prompt)
    total += _estimate_tokens(task_prompt)
    total += _estimate_messages_tokens(messages)
    return total


async def _extract_relevant_memory(
    task: str,
    messages: List[ModelMessage],
    model_pool: ModelPool,
    model_config: LLMConfig,
    target_tokens: int,
) -> str:
    """
    Use LLM to extract task-relevant memory from conversation history.
    
    Args:
        task: Current task to complete.
        messages: Full conversation history.
        model_pool: Pool for acquiring LLM handles.
        model_config: LLM configuration.
        target_tokens: Target token size for extracted memory.
    
    Returns:
        Extracted relevant memory.
    """
    history_text = _messages_to_text(messages)
    
    prompt = MEMORY_EXTRACTION_PROMPT.format(
        task=task,
        history=history_text,
    )
    
    logger.debug("[Memory] Extracting relevant memory for task")
    
    async with model_pool.acquire_context(model_config) as handle:
        runner = create_runner(handle, knowledge_store=None)
        
        result = await runner.generate(
            prompt=prompt,
            instructions=f"Extract relevant context. Keep response under {target_tokens * CHARS_PER_TOKEN} characters.",
        )
        
        memory = result.get("reply", "").strip()
    
    if "no relevant prior context" in memory.lower():
        logger.debug("[Memory] No relevant prior context found")
        return ""
    
    logger.info(f"[Memory] Extracted {_estimate_tokens(memory)} tokens of relevant context")
    return memory


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
    
    If total context fits within n_ctx, returns full history.
    If total context exceeds n_ctx, uses LLM to extract relevant memory.
    
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
    
    history = ctx.message_history
    n_ctx = model_config.n_ctx
    
    total_tokens = _estimate_total_context(system_prompt, task_prompt, history)
    
    logger.debug(f"[History] Total context: {total_tokens} tokens, n_ctx: {n_ctx}")
    
    if total_tokens <= n_ctx:
        logger.debug("[History] Full context fits within n_ctx, using all messages")
        return history
    
    logger.info(
        f"[History] Context ({total_tokens} tokens) exceeds n_ctx ({n_ctx}), "
        "extracting relevant memory"
    )
    
    base_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(task_prompt)
    available_for_history = n_ctx - base_tokens
    
    memory = await _extract_relevant_memory(
        task=task,
        messages=history,
        model_pool=model_pool,
        model_config=model_config,
        target_tokens=available_for_history,
    )
    
    if not memory:
        recent_messages = []
        tokens_used = 0
        
        for msg in reversed(history):
            msg_tokens = 0
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'content'):
                        msg_tokens += _estimate_tokens(part.content)
            
            if tokens_used + msg_tokens > available_for_history:
                break
            
            recent_messages.insert(0, msg)
            tokens_used += msg_tokens
        
        logger.info(f"[History] Using {len(recent_messages)} recent messages")
        return recent_messages
    
    memory_tokens = _estimate_tokens(memory)
    remaining_for_recent = available_for_history - memory_tokens
    
    recent_messages = []
    tokens_used = 0
    
    for msg in reversed(history):
        msg_tokens = 0
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if hasattr(part, 'content'):
                    msg_tokens += _estimate_tokens(part.content)
        
        if tokens_used + msg_tokens > remaining_for_recent:
            break
        
        recent_messages.insert(0, msg)
        tokens_used += msg_tokens
    
    result = []
    
    memory_message = ModelRequest(
        parts=[UserPromptPart(
            content=f"[Relevant context from earlier conversation: {memory}]"
        )]
    )
    result.append(memory_message)
    result.extend(recent_messages)
    
    logger.info(
        f"[History] Prepared context: memory ({memory_tokens} tokens) + "
        f"{len(recent_messages)} recent messages ({tokens_used} tokens)"
    )
    
    return result


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
            t for t in tools_metadata
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
        model = LocalLLMModel(runner, system_prompt=PLANNER_SYSTEM_PROMPT)

        agent = Agent(
            model=model,
            deps_type=AgentDeps,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

        result = await agent.run(prompt, message_history=prepared_history)
        response_text = result.output if isinstance(result.output, str) else str(result.output)

    parsed = parse_json_from_response(response_text)
    if not parsed:
        logger.warning("[Planner] Standard parsing failed, attempting JSON repair")
        parsed = repair_plan_json(response_text)

    if not parsed:
        logger.error(f"[Planner] Failed to parse planner response: {response_text[:500]}")
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
        logger.debug(f"[Executor] Using {len(prepared_history)} history messages") # NOT COVERED

    deps = AgentDeps(
        knowledge_store=knowledge_store,
        tool_registry=tool_registry,
        document_scope=ctx.document_scope,
    )

    async with model_pool.acquire_context(model_config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        model = LocalLLMModel(runner, system_prompt=EXECUTOR_SYSTEM_PROMPT)

        agent = Agent(
            model=model,
            deps_type=AgentDeps,
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
        )

        register_executor_tools(agent, deps)

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

    step_results = parse_executor_results(plan, response_text, deps)
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
) -> str:
    """
    Run the Synthesizer agent to create final answer from tool outputs.

    Args:
        plan: The original plan with user's task.
        execution_result: Results from execution with tool outputs.
        ctx: Execution context with session state and message history.
        model_pool: Pool for acquiring LLM handles.
        model_config: Configuration for the LLM model.
        knowledge_store: Store for document search.

    Returns:
        Synthesized final answer string.
    """
    prompt = build_synthesis_prompt(plan.task, execution_result)

    prepared_history = await _prepare_history(
        task=plan.task,
        ctx=ctx,
        system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
        task_prompt=prompt,
        model_pool=model_pool,
        model_config=model_config,
    )

    logger.debug(f"[Synthesizer] Running synthesizer for plan: {plan.plan_id}")
    if prepared_history:
        logger.debug(f"[Synthesizer] Using {len(prepared_history)} history messages") # NOT COVERED

    async with model_pool.acquire_context(model_config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        model = LocalLLMModel(runner, system_prompt=SYNTHESIZER_SYSTEM_PROMPT)

        agent = Agent(
            model=model,
            deps_type=AgentDeps,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
        )

        result = await agent.run(prompt, message_history=prepared_history)
        synthesized = result.output if isinstance(result.output, str) else str(result.output)

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
        logger.warning("[Verifier] Verifier did not return valid JSON, using heuristic")
        return heuristic_verification(plan, execution_result)

    try:
        verification = VerificationResult.model_validate(parsed)
    except ValidationError as e:
        logger.warning(f"[Verifier] Invalid verification structure: {e}, using heuristic")
        return heuristic_verification(plan, execution_result)

    logger.info(f"[Verifier] Verification complete: {verification.verification_status}")
    return verification