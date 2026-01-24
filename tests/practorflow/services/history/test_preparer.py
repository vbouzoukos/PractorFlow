import pytest
from unittest.mock import AsyncMock, MagicMock

from practorflow.services.history.preparer import (
    prepare_history,
    _select_recent_messages,
)
from practorflow.services.history.types import HistoryConfig, PreparedHistory


def _make_message(token_count: int):
    """
    Create a fake message whose token count is controlled
    via estimate_message_tokens mocking.
    """
    msg = MagicMock()
    msg._token_count = token_count
    return msg


@pytest.mark.asyncio
async def test_prepare_history_no_messages():
    result = await prepare_history(
        task="task",
        messages=[],
        system_prompt="sys",
        task_prompt="task",
        model_pool=MagicMock(),
        model_config=MagicMock(n_ctx=100),
        config=None,
    )

    assert isinstance(result, PreparedHistory)
    assert result.messages == []
    assert result.was_truncated is False
    assert result.original_count == 0


@pytest.mark.asyncio
async def test_prepare_history_full_context_fits(monkeypatch):
    messages = [_make_message(5), _make_message(5)]

    monkeypatch.setattr(
        "practorflow.services.history.preparer.estimate_context",
        lambda **kwargs: MagicMock(
            total_tokens=10,
            history_tokens=10,
            non_history_tokens=0,
        ),
    )

    result = await prepare_history(
        task="task",
        messages=messages,
        system_prompt="sys",
        task_prompt="task",
        model_pool=MagicMock(),
        model_config=MagicMock(n_ctx=50),
        config=HistoryConfig(
            n_ctx=50,
            reserve_for_response=0,  # FIX
        ),
    )

    assert result.messages == messages
    assert result.was_truncated is False
    assert result.included_count == 2


@pytest.mark.asyncio
async def test_prepare_history_truncation_without_memory(monkeypatch):
    messages = [_make_message(10), _make_message(10), _make_message(10)]

    monkeypatch.setattr(
        "practorflow.services.history.preparer.estimate_context",
        lambda **kwargs: MagicMock(
            total_tokens=100,
            history_tokens=30,
            non_history_tokens=10,
        ),
    )

    monkeypatch.setattr(
        "practorflow.services.history.preparer.extract_relevant_memory",
        AsyncMock(return_value=None),
    )

    monkeypatch.setattr(
        "practorflow.services.history.preparer.estimate_message_tokens",
        lambda msg, *_: msg._token_count,
    )

    result = await prepare_history(
        task="task",
        messages=messages,
        system_prompt="sys",
        task_prompt="task",
        model_pool=MagicMock(),
        model_config=MagicMock(n_ctx=20),
        config=HistoryConfig(n_ctx=20, min_recent_messages=1),
    )

    assert result.was_truncated is True
    assert result.included_count >= 1
    assert result.messages[-1] is messages[-1]


@pytest.mark.asyncio
async def test_prepare_history_truncation_with_memory(monkeypatch):
    messages = [_make_message(10), _make_message(10), _make_message(10)]

    monkeypatch.setattr(
        "practorflow.services.history.preparer.estimate_context",
        lambda **kwargs: MagicMock(
            total_tokens=100,
            history_tokens=30,
            non_history_tokens=10,
        ),
    )

    monkeypatch.setattr(
        "practorflow.services.history.preparer.extract_relevant_memory",
        AsyncMock(return_value="important memory"),
    )

    monkeypatch.setattr(
        "practorflow.services.history.preparer.estimate_tokens",
        lambda text, *_: 5,
    )

    monkeypatch.setattr(
        "practorflow.services.history.preparer.estimate_message_tokens",
        lambda msg, *_: msg._token_count,
    )

    memory_message = MagicMock()
    monkeypatch.setattr(
        "practorflow.services.history.preparer.create_memory_message",
        lambda memory: memory_message,
    )

    result = await prepare_history(
        task="task",
        messages=messages,
        system_prompt="sys",
        task_prompt="task",
        model_pool=MagicMock(),
        model_config=MagicMock(n_ctx=20),
        config=HistoryConfig(n_ctx=20, min_recent_messages=1),
    )

    assert result.was_truncated is True
    assert result.extracted_memory == "important memory"
    assert result.messages[0] is memory_message
    assert result.included_count >= 1


def test_select_recent_messages_empty():
    assert _select_recent_messages([], 10, 4) == []


def test_select_recent_messages_respects_budget(monkeypatch):
    msgs = [_make_message(5), _make_message(5), _make_message(5)]

    monkeypatch.setattr(
        "practorflow.services.history.preparer.estimate_message_tokens",
        lambda msg, *_: msg._token_count,
    )

    recent = _select_recent_messages(
        messages=msgs,
        available_tokens=10,
        chars_per_token=4,
        min_messages=0,
    )

    assert recent == msgs[-2:]


def test_select_recent_messages_min_messages(monkeypatch):
    msgs = [_make_message(5), _make_message(5), _make_message(5)]

    monkeypatch.setattr(
        "practorflow.services.history.preparer.estimate_message_tokens",
        lambda msg, *_: msg._token_count,
    )

    recent = _select_recent_messages(
        messages=msgs,
        available_tokens=0,
        chars_per_token=4,
        min_messages=2,
    )

    assert recent == msgs[-2:]
