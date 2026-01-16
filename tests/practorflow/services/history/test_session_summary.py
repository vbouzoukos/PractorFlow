# tests/practorflow/services/history/test_session_summary.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.practorflow.common.fixtures import mock_llm_config

from practorflow.llm.base.session import Message, Session
from practorflow.services.history.session_summary import (
    _has_sufficient_messages,
    update_session_title,
    MIN_MESSAGES_FOR_TITLE,
)


def _msg(role: str, text: str) -> Message:
    msg = MagicMock(spec=Message)
    msg.role = role
    msg.get_text_content.return_value = text
    return msg


# -------------------------
# helpers
# -------------------------

def test_has_sufficient_messages_variants():
    assert _has_sufficient_messages([_msg("user", "hi")]) is False
    assert _has_sufficient_messages([
        _msg("user", "hi"),
        _msg("user", "again"),
    ]) is False
    assert _has_sufficient_messages([
        _msg("user", "hi"),
        _msg("assistant", "hello"),
    ]) is True
    assert MIN_MESSAGES_FOR_TITLE == 2


# -------------------------
# update_session_title
# -------------------------
@pytest.mark.asyncio
async def test_update_session_title_already_set(mock_llm_config):
    session = MagicMock(spec=Session)
    session.title = "Existing"
    session.session_id = "s1"

    result = await update_session_title(
        session=session,
        model_pool=MagicMock(),
        model_config=mock_llm_config,
    )

    assert result is False


@pytest.mark.asyncio
async def test_update_session_title_generated(mock_llm_config):
    session = MagicMock(spec=Session)
    session.title = None
    session.session_id = "s1"
    session.messages = [
        _msg("user", "hi"),
        _msg("assistant", "hello"),
    ]

    mock_runner = MagicMock()
    mock_runner.generate = AsyncMock(return_value="New Title")

    mock_handle = MagicMock()
    mock_handle.backend = "llama_cpp"

    mock_pool = MagicMock()
    mock_pool.acquire_context.return_value.__aenter__ = AsyncMock(return_value=mock_handle)
    mock_pool.acquire_context.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "practorflow.services.history.session_summary.create_runner",
        return_value=mock_runner,
    ):
        result = await update_session_title(
            session=session,
            model_pool=mock_pool,
            model_config=mock_llm_config,
        )

    assert result is True
    assert session.title == "New Title"

@pytest.mark.asyncio
async def test_update_session_title_not_generated_due_to_insufficient_messages(mock_llm_config):
    session = MagicMock(spec=Session)
    session.title = None
    session.session_id = "s1"
    session.messages = [_msg("user", "hi")]

    result = await update_session_title(
        session=session,
        model_pool=MagicMock(),
        model_config=mock_llm_config,
    )

    assert result is False

@pytest.mark.asyncio
async def test_update_session_title_generation_exception(mock_llm_config):
    session = MagicMock(spec=Session)
    session.title = None
    session.session_id = "s1"
    session.messages = [
        _msg("user", "hi"),
        _msg("assistant", "hello"),
    ]

    mock_pool = MagicMock()
    mock_pool.acquire_context.side_effect = RuntimeError("boom")

    result = await update_session_title(
        session=session,
        model_pool=mock_pool,
        model_config=mock_llm_config,
    )

    assert result is False

@pytest.mark.asyncio
async def test_update_session_title_no_title_generated(mock_llm_config):
    session = MagicMock(spec=Session)
    session.title = None
    session.session_id = "s1"
    session.messages = [
        _msg("user", "hi"),
        _msg("assistant", "hello"),
    ]

    mock_runner = MagicMock()
    # Empty / whitespace-only title → generate_session_title returns None
    mock_runner.generate = AsyncMock(return_value="   ")

    mock_handle = MagicMock()
    mock_handle.backend = "llama_cpp"
    mock_handle.config = mock_llm_config

    mock_pool = MagicMock()
    mock_pool.acquire_context.return_value.__aenter__ = AsyncMock(return_value=mock_handle)
    mock_pool.acquire_context.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "practorflow.services.history.session_summary.create_runner",
        return_value=mock_runner,
    ):
        result = await update_session_title(
            session=session,
            model_pool=mock_pool,
            model_config=mock_llm_config,
        )

    assert result is False
    assert session.title is None
