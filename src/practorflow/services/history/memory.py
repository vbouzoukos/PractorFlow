"""
Memory extraction using LLM.

Extracts task-relevant context from conversation history
when full history exceeds context window limits.
"""

from typing import List, Optional

from pydantic_ai.messages import ModelMessage

from practorflow.llm.pool.model_pool import ModelPool
from practorflow.llm.factory import create_runner
from practorflow.llm.llm_config import LLMConfig
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.history.types import CHARS_PER_TOKEN
from practorflow.services.history.builder import messages_to_text
from practorflow.services.history.estimator import estimate_tokens

logger = get_logger("history_memory", level=appConfiguration.LoggerConfiguration.AgentLevel)


MEMORY_EXTRACTION_PROMPT = """You are a memory extraction assistant. Your task is to extract relevant information from conversation history that is needed to complete the current task.

CURRENT TASK:
{task}

CONVERSATION HISTORY:
{history}

Extract ONLY information from the history that is directly relevant to completing the current task. Include:
- Specific requirements or constraints mentioned
- Decisions already made
- Important names, numbers, dates, or technical details
- Context that affects how the task should be completed

Provide a concise summary of relevant context. Do not include irrelevant information. If nothing is relevant, return an empty response."""


async def extract_relevant_memory(
    task: str,
    messages: List[ModelMessage],
    model_pool: ModelPool,
    model_config: LLMConfig,
    target_tokens: int,
    chars_per_token: int = CHARS_PER_TOKEN,
) -> Optional[str]:
    """
    Use LLM to extract task-relevant memory from conversation history.

    Sends the full history to an LLM with instructions to extract
    only information relevant to the current task.

    Args:
        task: Current task to complete.
        messages: Full conversation history.
        model_pool: Pool for acquiring LLM handles.
        model_config: LLM configuration.
        target_tokens: Target token size for extracted memory.
        chars_per_token: Character-to-token ratio.

    Returns:
        Extracted relevant memory or None if no relevant context.
    """
    if not messages:
        return None

    history_text = messages_to_text(messages)

    prompt = MEMORY_EXTRACTION_PROMPT.format(
        task=task,
        history=history_text,
    )

    logger.debug("[Memory] Extracting relevant memory for task")

    try:
        async with model_pool.acquire_context(model_config) as handle:
            runner = create_runner(handle, knowledge_store=None)

            target_chars = target_tokens * chars_per_token
            result = await runner.generate(
                prompt=prompt,
                instructions=f"Extract relevant context. Keep response under {target_chars} characters.",
            )

            memory = result.get("reply", "").strip()

        if not memory:
            logger.debug("[Memory] No memory extracted")
            return None

        memory_tokens = estimate_tokens(memory, chars_per_token)
        logger.info(f"[Memory] Extracted {memory_tokens} tokens of relevant context")

        return memory

    except Exception as e:
        logger.warning(f"[Memory] Failed to extract memory: {e}")
        return None