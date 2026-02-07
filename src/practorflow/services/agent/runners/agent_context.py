"""
Agent context helpers for multi-agent pipeline.

Provides shared utilities used across agent runners:
- History preparation for LLM calls
- Persona and instructions extraction from conversation history
- Context-enhanced prompt building
"""

from typing import List, Optional

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from practorflow.llm.pool.model_pool import ModelPool
from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.pyai.model import LocalLLMModel
from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.agent.context import ExecutionContext
from practorflow.services.agent.deps import AgentDeps

from practorflow.services.tools.registration import build_tools_for_registry
from practorflow.services.history.types import HistoryConfig
from practorflow.services.history.preparer import prepare_history

logger = get_logger(
    "agent_context", level=appConfiguration.LoggerConfiguration.AgentLevel
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


async def prepare_agent_history(
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


async def extract_context(
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


def build_context_enhanced_prompt(
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