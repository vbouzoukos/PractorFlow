"""
History types and data structures.

Provides core types for history management across chat and agent services.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from pydantic_ai.messages import ModelMessage


CHARS_PER_TOKEN = 4
"""Default character-to-token ratio for estimation."""


@dataclass
class TokenEstimate:
    """
    Result of token estimation.

    Provides breakdown of token counts for context budgeting.
    """

    system_tokens: int = 0
    prompt_tokens: int = 0
    history_tokens: int = 0
    total_tokens: int = 0

    @property
    def non_history_tokens(self) -> int:
        """Tokens used by system and prompt."""
        return self.system_tokens + self.prompt_tokens


@dataclass
class HistoryConfig:
    """
    Configuration for history preparation.

    Controls how history is processed and truncated.
    """

    n_ctx: int = 4096
    """Maximum context window size in tokens."""

    chars_per_token: int = CHARS_PER_TOKEN
    """Character-to-token ratio for estimation."""

    reserve_for_response: int = 512
    """Tokens to reserve for model response."""

    min_recent_messages: int = 2
    """Minimum recent messages to keep even when truncating."""

    @property
    def available_context(self) -> int:
        """Context available after reserving for response."""
        return self.n_ctx - self.reserve_for_response


@dataclass
class PreparedHistory:
    """
    Result of history preparation.

    Contains the processed message history ready for LLM consumption,
    along with metadata about the preparation.
    """

    messages: List[ModelMessage] = field(default_factory=list)
    """Prepared message history."""

    extracted_memory: Optional[str] = None
    """Extracted memory summary if context was exceeded."""

    original_count: int = 0
    """Number of messages in original history."""

    included_count: int = 0
    """Number of messages included after preparation."""

    estimated_tokens: int = 0
    """Estimated token count of prepared history."""

    was_truncated: bool = False
    """Whether history was truncated or summarized."""

    @property
    def has_memory(self) -> bool:
        """Check if memory extraction was performed."""
        return self.extracted_memory is not None and len(self.extracted_memory) > 0

    @property
    def reduction_ratio(self) -> float:
        """Ratio of included to original messages (1.0 = no reduction)."""
        if self.original_count == 0:
            return 1.0
        return self.included_count / self.original_count