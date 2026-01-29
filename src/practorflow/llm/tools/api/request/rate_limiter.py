"""
Rate limiter for API Tool HTTP requests.

Provides token bucket rate limiting for requests per minute.
"""

import asyncio
import time


class RateLimiter:
    """
    Token bucket rate limiter for RPM limiting.
    
    Uses token bucket algorithm to limit requests per minute.
    """
    
    def __init__(self, rpm: int = 0):
        """
        Initialize rate limiter.
        
        Args:
            rpm: Requests per minute limit (0 = unlimited).
        """
        self.rpm = rpm
        self._tokens = float(rpm) if rpm > 0 else float("inf")
        self._max_tokens = float(rpm) if rpm > 0 else float("inf")
        self._refill_rate = rpm / 60.0 if rpm > 0 else 0.0
        self._last_refill = time.monotonic()
    
    @property
    def is_enabled(self) -> bool:
        """Whether rate limiting is enabled."""
        return self.rpm > 0
    
    async def acquire(self) -> None:
        """Acquire permission to make a request, waiting if necessary."""
        if not self.is_enabled:
            return
        
        while True:
            self._refill()
            
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            
            wait_time = (1.0 - self._tokens) / self._refill_rate
            await asyncio.sleep(wait_time)
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)