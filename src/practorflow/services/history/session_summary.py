"""
Session title summary generator.

Provides helper class for generating concise conversation titles
from message history using LLM inference.
"""

from typing import List, Optional

from practorflow.llm.base.session import Message, Session
from practorflow.llm.pool.model_pool import ModelPool
from practorflow.llm.factory import create_runner
from practorflow.llm.llm_config import LLMConfig
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger(
    "session_summary", level=appConfiguration.LoggerConfiguration.AgentLevel
)

TITLE_SYSTEM_PROMPT = """You are a helpful assistant that generates concise conversation titles.
Generate a short, descriptive title (5-10 words maximum) that captures the main topic or purpose of the conversation.
Respond with ONLY the title text, no quotes, no explanations, no punctuation at the end."""

TITLE_USER_PROMPT_TEMPLATE = """Based on the following conversation, generate a concise title (5-10 words):

{conversation}

Title:"""

MIN_MESSAGES_FOR_TITLE = 2
"""Minimum number of messages required to generate a title (1 user + 1 assistant)."""


def _has_sufficient_messages(messages: List[Message]) -> bool:
    """
    Check if there are sufficient messages to generate a meaningful title.

    Requires at least one user message and one assistant message.

    Args:
        messages: List of Message objects.

    Returns:
        True if sufficient messages exist, False otherwise.
    """
    if len(messages) < MIN_MESSAGES_FOR_TITLE:
        return False

    has_user = any(msg.role == "user" for msg in messages)
    has_assistant = any(msg.role == "assistant" for msg in messages)

    return has_user and has_assistant


async def generate_session_title(
    messages: List[Message],
    model_pool: ModelPool,
    model_config: LLMConfig,
) -> Optional[str]:
    """
    Generate a concise title for a conversation.

    Uses the LLM to create a short, descriptive title based on
    the conversation history. Returns None if insufficient messages
    exist or if generation fails.

    Args:
        messages: List of Message objects from the session.
        model_pool: Pool for acquiring LLM handles.
        model_config: Configuration for the LLM model.

    Returns:
        Generated title string, or None if generation fails or
        insufficient data exists.
    """
    if not _has_sufficient_messages(messages):
        logger.debug("[SessionSummary] Insufficient messages for title generation")
        return None

    parts = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        content = msg.get_text_content()
        parts.append(f"{role}: {content}")

    conversation_text = "\n".join(parts)
    prompt = TITLE_USER_PROMPT_TEMPLATE.format(conversation=conversation_text)

    try:
        async with model_pool.acquire_context(model_config) as handle:
            runner = create_runner(handle)

            response = await runner.generate(
                prompt=prompt,
                instructions=TITLE_SYSTEM_PROMPT,
                temperature=0.3,
            )

            title = response.get("reply", "").strip().strip('"').strip("'").strip()

            if title and len(title) > 0:
                logger.debug(f"[SessionSummary] Generated title: {title}")
                return title

            logger.debug("[SessionSummary] Empty title generated")
            return None

    except Exception as e:
        logger.warning(f"[SessionSummary] Title generation failed: {e}")
        return None


async def update_session_title(
    session: Session,
    model_pool: ModelPool,
    model_config: LLMConfig,
) -> bool:
    """
    Update session title if not already set.

    Generates a title for the session if one doesn't exist and
    sufficient message history is available. Does not persist
    the session - caller is responsible for saving.

    Args:
        session: Session to update.
        model_pool: Pool for acquiring LLM handles.
        model_config: Configuration for the LLM model.

    Returns:
        True if title was generated, False otherwise.
    """
    if session.title is not None:
        logger.debug(f"[SessionSummary] Session {session.session_id} already has title")
        return False

    title = await generate_session_title(
        messages=session.messages,
        model_pool=model_pool,
        model_config=model_config,
    )

    if title is None:
        return False

    session.title = title

    logger.info(
        f"[SessionSummary] Updated session {session.session_id} with title: {title}"
    )
    return True
