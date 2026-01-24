"""
Message history builder.

Converts Session messages to pydantic_ai ModelMessage format.
"""

from typing import List

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)

from practorflow.llm.base.session import Session
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("history_builder", level=appConfiguration.LoggerConfiguration.AgentLevel)


def build_message_history(
    session: Session,
    exclude_last: bool = True,
) -> List[ModelMessage]:
    """
    Build message history from session.

    Converts session messages to pydantic_ai ModelMessage format.
    By default excludes the last message (assumed to be current user input).

    Args:
        session: Session containing message history.
        exclude_last: If True, excludes the last message from history.

    Returns:
        List of ModelMessage objects (ModelRequest/ModelResponse).
    """
    if not session.messages:
        return []

    messages = session.messages[:-1] if exclude_last else session.messages

    if not messages:
        return []

    history: List[ModelMessage] = []

    for msg in messages:
        content = msg.get_text_content()

        if msg.role == "user":
            history.append(
                ModelRequest(parts=[UserPromptPart(content=content)])
            )
        elif msg.role == "assistant":
            history.append(
                ModelResponse(parts=[TextPart(content=content)])
            )

    logger.debug(f"[HistoryBuilder] Built history with {len(history)} messages")

    return history


def messages_to_text(messages: List[ModelMessage]) -> str:
    """
    Convert messages to plain text format.

    Useful for memory extraction or summarization tasks.

    Args:
        messages: List of ModelMessage to convert.

    Returns:
        Plain text representation with role prefixes.
    """
    parts = []

    for msg in messages:
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if hasattr(part, "content"):
                    role = "User" if isinstance(msg, ModelRequest) else "Assistant"
                    parts.append(f"{role}: {part.content}")

    return "\n\n".join(parts)


def create_memory_message(memory: str) -> ModelRequest:
    """
    Create a ModelRequest containing extracted memory context.

    Used to inject summarized prior context into message history.

    Args:
        memory: Extracted memory text.

    Returns:
        ModelRequest with memory as context.
    """
    return ModelRequest(
        parts=[UserPromptPart(
            content=f"[Relevant context from earlier conversation: {memory}]"
        )]
    )