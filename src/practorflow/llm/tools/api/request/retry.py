"""
Retry handler for API Tool HTTP requests.

Provides configurable retry logic with backoff strategies.
"""

import asyncio

from practorflow.llm.tools.api.models import RetryBackoff


class RetryHandler:
    """
    Handles retry logic with configurable backoff strategies.
    
    Strategies:
    - none: No delay between retries
    - linear: delay = base_delay * attempt
    - exponential: delay = base_delay * (2 ^ attempt)
    """
    
    BASE_DELAY = 1.0
    MAX_DELAY = 60.0
    
    def __init__(
        self,
        max_attempts: int,
        backoff: RetryBackoff,
        retry_on_status: list[int],
    ):
        self.max_attempts = max_attempts
        self.backoff = backoff
        self.retry_on_status = set(retry_on_status)
    
    def is_retryable(self, status_code: int | None, is_network_error: bool) -> bool:
        """Check if error is retryable."""
        if is_network_error:
            return True
        return status_code is not None and status_code in self.retry_on_status
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.backoff == RetryBackoff.NONE:
            return 0.0
        if self.backoff == RetryBackoff.LINEAR:
            delay = self.BASE_DELAY * attempt
        else:  # EXPONENTIAL
            delay = self.BASE_DELAY * (2 ** (attempt - 1))
        return min(delay, self.MAX_DELAY)
    
    async def wait(self, attempt: int) -> None:
        """Wait before retry based on backoff strategy."""
        delay = self.get_delay(attempt)
        if delay > 0:
            await asyncio.sleep(delay)