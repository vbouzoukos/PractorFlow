"""
History preparer.

Orchestrates history preparation for LLM calls, handling context
window limits through truncation or memory extraction.
"""

from typing import List, Optional

from pydantic_ai.messages import ModelMessage

from practorflow.llm.pool.model_pool import ModelPool
from practorflow.llm.llm_config import LLMConfig
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.history.types import (
    HistoryConfig,
    PreparedHistory,
)
from practorflow.services.history.estimator import (
    estimate_tokens,
    estimate_message_tokens,
    estimate_context,
)
from practorflow.services.history.builder import create_memory_message
from practorflow.services.history.memory import extract_relevant_memory

logger = get_logger("history_preparer", level=appConfiguration.LoggerConfiguration.AgentLevel)


def _select_recent_messages(
    messages: List[ModelMessage],
    available_tokens: int,
    chars_per_token: int,
    min_messages: int = 0,
) -> List[ModelMessage]:
    """
    Select most recent messages that fit within token budget.

    Args:
        messages: Full message history.
        available_tokens: Token budget for messages.
        chars_per_token: Character-to-token ratio.
        min_messages: Minimum messages to include regardless of budget.

    Returns:
        List of recent messages within budget.
    """
    if not messages:
        return []

    recent_messages: List[ModelMessage] = []
    tokens_used = 0

    for msg in reversed(messages):
        msg_tokens = estimate_message_tokens(msg, chars_per_token)

        if tokens_used + msg_tokens > available_tokens:
            if len(recent_messages) >= min_messages:
                break

        recent_messages.insert(0, msg)
        tokens_used += msg_tokens

    return recent_messages


async def prepare_history(
    task: str,
    messages: List[ModelMessage],
    system_prompt: str,
    task_prompt: str,
    model_pool: ModelPool,
    model_config: LLMConfig,
    config: Optional[HistoryConfig] = None,
) -> PreparedHistory:
    """
    Prepare message history for LLM call.

    If total context fits within n_ctx, returns full history.
    If total context exceeds n_ctx, uses LLM to extract relevant
    memory and combines with recent messages.

    Args:
        task: Current task description.
        messages: Full message history.
        system_prompt: System-level instructions.
        task_prompt: Current task prompt.
        model_pool: Pool for LLM access.
        model_config: LLM configuration with n_ctx.
        config: Optional history configuration.

    Returns:
        PreparedHistory with processed messages and metadata.
    """
    if config is None:
        config = HistoryConfig(n_ctx=model_config.n_ctx)

    original_count = len(messages)

    if not messages:
        return PreparedHistory(
            messages=[],
            original_count=0,
            included_count=0,
            estimated_tokens=0,
            was_truncated=False,
        )

    estimate = estimate_context(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        messages=messages,
        chars_per_token=config.chars_per_token,
    )

    logger.debug(
        f"[Preparer] Context estimate: {estimate.total_tokens} tokens, "
        f"n_ctx: {config.n_ctx}"
    )

    if estimate.total_tokens <= config.available_context:
        logger.debug("[Preparer] Full context fits, using all messages")
        return PreparedHistory(
            messages=messages,
            original_count=original_count,
            included_count=original_count,
            estimated_tokens=estimate.history_tokens,
            was_truncated=False,
        )

    logger.info(
        f"[Preparer] Context ({estimate.total_tokens} tokens) exceeds limit "
        f"({config.available_context}), preparing reduced history"
    )

    available_for_history = config.available_context - estimate.non_history_tokens

    memory = await extract_relevant_memory(
        task=task,
        messages=messages,
        model_pool=model_pool,
        model_config=model_config,
        target_tokens=available_for_history // 2,
        chars_per_token=config.chars_per_token,
    )

    if not memory:
        recent = _select_recent_messages(
            messages=messages,
            available_tokens=available_for_history,
            chars_per_token=config.chars_per_token,
            min_messages=config.min_recent_messages,
        )

        recent_tokens = sum(
            estimate_message_tokens(m, config.chars_per_token) for m in recent
        )

        logger.info(f"[Preparer] Using {len(recent)} recent messages ({recent_tokens} tokens)")

        return PreparedHistory(
            messages=recent,
            original_count=original_count,
            included_count=len(recent),
            estimated_tokens=recent_tokens,
            was_truncated=True,
        )

    memory_tokens = estimate_tokens(memory, config.chars_per_token)
    remaining_for_recent = available_for_history - memory_tokens

    recent = _select_recent_messages(
        messages=messages,
        available_tokens=remaining_for_recent,
        chars_per_token=config.chars_per_token,
        min_messages=config.min_recent_messages,
    )

    result_messages: List[ModelMessage] = []
    memory_message = create_memory_message(memory)
    result_messages.append(memory_message)
    result_messages.extend(recent)

    recent_tokens = sum(
        estimate_message_tokens(m, config.chars_per_token) for m in recent
    )
    total_tokens = memory_tokens + recent_tokens

    logger.info(
        f"[Preparer] Prepared: memory ({memory_tokens} tokens) + "
        f"{len(recent)} recent messages ({recent_tokens} tokens)"
    )

    return PreparedHistory(
        messages=result_messages,
        extracted_memory=memory,
        original_count=original_count,
        included_count=len(recent) + 1,
        estimated_tokens=total_tokens,
        was_truncated=True,
    )