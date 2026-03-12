"""Tests for RetryHandler."""

import pytest
from unittest.mock import patch

from practorflow.llm.tools.api.models.enums import RetryBackoff
from practorflow.llm.tools.api.request.retry import RetryHandler


def _make_handler(max_attempts=3, backoff=RetryBackoff.EXPONENTIAL, retry_on_status=None):
    return RetryHandler(
        max_attempts=max_attempts,
        backoff=backoff,
        retry_on_status=retry_on_status or [429, 500, 503],
    )


def test_init_sets_attributes():
    handler = _make_handler(max_attempts=5, backoff=RetryBackoff.LINEAR)
    assert handler.max_attempts == 5
    assert handler.backoff == RetryBackoff.LINEAR
    assert 429 in handler.retry_on_status
    assert 500 in handler.retry_on_status


def test_is_retryable_network_error_always_true():
    handler = _make_handler()
    assert handler.is_retryable(None, is_network_error=True) is True


def test_is_retryable_status_code_in_list():
    handler = _make_handler(retry_on_status=[429, 500])
    assert handler.is_retryable(429, is_network_error=False) is True
    assert handler.is_retryable(500, is_network_error=False) is True


def test_is_retryable_status_code_not_in_list():
    handler = _make_handler(retry_on_status=[429, 500])
    assert handler.is_retryable(404, is_network_error=False) is False


def test_is_retryable_none_status_not_network_error():
    handler = _make_handler()
    assert handler.is_retryable(None, is_network_error=False) is False


def test_get_delay_none_backoff_always_zero():
    handler = _make_handler(backoff=RetryBackoff.NONE)
    assert handler.get_delay(1) == 0.0
    assert handler.get_delay(3) == 0.0


def test_get_delay_linear_backoff():
    handler = _make_handler(backoff=RetryBackoff.LINEAR)
    assert handler.get_delay(1) == 1.0
    assert handler.get_delay(2) == 2.0
    assert handler.get_delay(3) == 3.0


def test_get_delay_exponential_backoff():
    handler = _make_handler(backoff=RetryBackoff.EXPONENTIAL)
    assert handler.get_delay(1) == 1.0   # 1.0 * 2^0
    assert handler.get_delay(2) == 2.0   # 1.0 * 2^1
    assert handler.get_delay(3) == 4.0   # 1.0 * 2^2
    assert handler.get_delay(4) == 8.0   # 1.0 * 2^3


def test_get_delay_capped_at_max():
    handler = _make_handler(backoff=RetryBackoff.LINEAR)
    assert handler.get_delay(100) == RetryHandler.MAX_DELAY


@pytest.mark.asyncio
async def test_wait_calls_sleep_for_positive_delay():
    handler = _make_handler(backoff=RetryBackoff.LINEAR)
    with patch("practorflow.llm.tools.api.request.retry.asyncio.sleep") as mock_sleep:
        mock_sleep.return_value = None
        await handler.wait(1)
        mock_sleep.assert_called_once_with(1.0)


@pytest.mark.asyncio
async def test_wait_no_sleep_for_none_backoff():
    handler = _make_handler(backoff=RetryBackoff.NONE)
    with patch("practorflow.llm.tools.api.request.retry.asyncio.sleep") as mock_sleep:
        await handler.wait(1)
        mock_sleep.assert_not_called()
