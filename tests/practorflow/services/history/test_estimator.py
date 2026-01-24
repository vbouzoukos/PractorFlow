from unittest.mock import MagicMock

from practorflow.services.history.estimator import (
    estimate_tokens,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_context,
)
from practorflow.services.history.types import CHARS_PER_TOKEN, TokenEstimate


def test_estimate_tokens_empty_string_returns_zero():
    assert estimate_tokens("") == 0


def test_estimate_tokens_none_returns_zero():
    assert estimate_tokens(None) == 0


def test_estimate_tokens_non_empty_text():
    text = "abcd" * 5  # 20 chars
    assert estimate_tokens(text, chars_per_token=4) == 5


def test_estimate_message_tokens_no_parts():
    message = MagicMock()
    delattr(message, "parts")

    assert estimate_message_tokens(message) == 0


def test_estimate_message_tokens_with_parts_and_content():
    part1 = MagicMock(content="abcd" * 2)  # 8 chars → 2 tokens
    part2 = MagicMock(content="abcd" * 3)  # 12 chars → 3 tokens

    message = MagicMock()
    message.parts = [part1, part2]

    assert estimate_message_tokens(message, chars_per_token=4) == 5


def test_estimate_message_tokens_ignores_parts_without_content():
    part_with_content = MagicMock(content="abcd" * 2)
    part_without_content = MagicMock()
    delattr(part_without_content, "content")

    message = MagicMock()
    message.parts = [part_with_content, part_without_content]

    assert estimate_message_tokens(message, chars_per_token=4) == 2


def test_estimate_messages_tokens_multiple_messages():
    msg1 = MagicMock()
    msg1.parts = [MagicMock(content="abcd" * 2)]  # 2 tokens

    msg2 = MagicMock()
    msg2.parts = [MagicMock(content="abcd" * 3)]  # 3 tokens

    total = estimate_messages_tokens([msg1, msg2], chars_per_token=4)

    assert total == 5


def test_estimate_context_full_breakdown():
    system_prompt = "abcd" * 5   # 20 chars → 5 tokens
    task_prompt = "abcd" * 3     # 12 chars → 3 tokens

    msg = MagicMock()
    msg.parts = [MagicMock(content="abcd" * 2)]  # 8 chars → 2 tokens

    estimate = estimate_context(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        messages=[msg],
        chars_per_token=4,
    )

    assert isinstance(estimate, TokenEstimate)
    assert estimate.system_tokens == 5
    assert estimate.prompt_tokens == 3
    assert estimate.history_tokens == 2
    assert estimate.total_tokens == 10
