"""Tests for RateLimiter."""

import time

import pytest
from unittest.mock import patch

from practorflow.llm.tools.api.request.rate_limiter import RateLimiter


def test_init_unlimited():
    limiter = RateLimiter(rpm=0)
    assert limiter.rpm == 0
    assert not limiter.is_enabled
    assert limiter._tokens == float("inf")
    assert limiter._max_tokens == float("inf")
    assert limiter._refill_rate == 0.0


def test_init_limited():
    limiter = RateLimiter(rpm=60)
    assert limiter.rpm == 60
    assert limiter.is_enabled
    assert limiter._tokens == 60.0
    assert limiter._max_tokens == 60.0
    assert limiter._refill_rate == 1.0  # 60 / 60


def test_is_enabled_false_when_zero():
    limiter = RateLimiter(rpm=0)
    assert limiter.is_enabled is False


def test_is_enabled_true_when_nonzero():
    limiter = RateLimiter(rpm=30)
    assert limiter.is_enabled is True


@pytest.mark.asyncio
async def test_acquire_unlimited_returns_immediately():
    limiter = RateLimiter(rpm=0)
    with patch("practorflow.llm.tools.api.request.rate_limiter.asyncio.sleep") as mock_sleep:
        await limiter.acquire()
        mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_acquire_with_tokens_available():
    limiter = RateLimiter(rpm=60)
    initial_tokens = limiter._tokens
    await limiter.acquire()
    assert limiter._tokens == initial_tokens - 1.0


@pytest.mark.asyncio
async def test_acquire_waits_when_no_tokens():
    limiter = RateLimiter(rpm=60)
    limiter._tokens = 0.0

    async def fake_sleep(t):
        limiter._tokens = 1.0  # Simulate token refill during sleep

    with patch("practorflow.llm.tools.api.request.rate_limiter.asyncio.sleep", side_effect=fake_sleep):
        await limiter.acquire()


def test_refill_adds_tokens_based_on_elapsed_time():
    limiter = RateLimiter(rpm=60)
    limiter._tokens = 0.0
    limiter._last_refill = time.monotonic() - 1.0  # Simulate 1 second passing
    limiter._refill()
    assert limiter._tokens > 0.0


def test_refill_caps_at_max_tokens():
    limiter = RateLimiter(rpm=60)
    limiter._tokens = 59.0
    limiter._last_refill = time.monotonic() - 100.0  # Lots of time passed
    limiter._refill()
    assert limiter._tokens == limiter._max_tokens
