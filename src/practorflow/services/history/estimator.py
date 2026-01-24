"""
Token estimation utilities.

Provides functions for estimating token counts from text and messages.
"""

from typing import List

from pydantic_ai.messages import ModelMessage

from practorflow.services.history.types import (
    CHARS_PER_TOKEN,
    TokenEstimate,
)


def estimate_tokens(text: str, chars_per_token: int = CHARS_PER_TOKEN) -> int:
    """
    Estimate token count from text.

    Uses character-based estimation as an approximation.
    Actual token count varies by tokenizer.

    Args:
        text: Text to estimate tokens for.
        chars_per_token: Character-to-token ratio.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return len(text) // chars_per_token


def estimate_message_tokens(
    message: ModelMessage,
    chars_per_token: int = CHARS_PER_TOKEN,
) -> int:
    """
    Estimate token count for a single message.

    Extracts content from all parts and sums their token estimates.

    Args:
        message: ModelMessage to estimate.
        chars_per_token: Character-to-token ratio.

    Returns:
        Estimated token count.
    """
    total = 0
    if hasattr(message, "parts"):
        for part in message.parts:
            if hasattr(part, "content"):
                total += estimate_tokens(part.content, chars_per_token)
    return total


def estimate_messages_tokens(
    messages: List[ModelMessage],
    chars_per_token: int = CHARS_PER_TOKEN,
) -> int:
    """
    Estimate total tokens in message history.

    Args:
        messages: List of ModelMessage to estimate.
        chars_per_token: Character-to-token ratio.

    Returns:
        Total estimated token count.
    """
    total = 0
    for msg in messages:
        total += estimate_message_tokens(msg, chars_per_token)
    return total


def estimate_context(
    system_prompt: str,
    task_prompt: str,
    messages: List[ModelMessage],
    chars_per_token: int = CHARS_PER_TOKEN,
) -> TokenEstimate:
    """
    Estimate total tokens for full context.

    Provides breakdown of token usage across system prompt,
    task prompt, and message history.

    Args:
        system_prompt: System-level instructions.
        task_prompt: Current task or user prompt.
        messages: Message history.
        chars_per_token: Character-to-token ratio.

    Returns:
        TokenEstimate with breakdown.
    """
    system_tokens = estimate_tokens(system_prompt, chars_per_token)
    prompt_tokens = estimate_tokens(task_prompt, chars_per_token)
    history_tokens = estimate_messages_tokens(messages, chars_per_token)

    return TokenEstimate(
        system_tokens=system_tokens,
        prompt_tokens=prompt_tokens,
        history_tokens=history_tokens,
        total_tokens=system_tokens + prompt_tokens + history_tokens,
    )