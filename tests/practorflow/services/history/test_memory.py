import pytest
from unittest.mock import AsyncMock, MagicMock

from practorflow.services.history.memory import extract_relevant_memory


@pytest.mark.asyncio
async def test_extract_relevant_memory_no_messages_returns_none():
    result = await extract_relevant_memory(
        task="test task",
        messages=[],
        model_pool=MagicMock(),
        model_config=MagicMock(),
        target_tokens=100,
    )

    assert result is None


@pytest.mark.asyncio
async def test_extract_relevant_memory_success(monkeypatch):
    messages = [
        MagicMock(role="user", content="The API timeout is 30 seconds."),
        MagicMock(role="assistant", content="Acknowledged."),
    ]

    mock_runner = MagicMock()
    mock_runner.generate = AsyncMock(
        return_value={"reply": "API timeout is 30 seconds."}
    )

    mock_handle = MagicMock()
    mock_pool = MagicMock()
    mock_pool.acquire_context.return_value.__aenter__.return_value = mock_handle
    mock_pool.acquire_context.return_value.__aexit__.return_value = None

    monkeypatch.setattr(
        "practorflow.services.history.memory.create_runner",
        lambda handle, knowledge_store=None: mock_runner,
    )

    result = await extract_relevant_memory(
        task="What is the API timeout?",
        messages=messages,
        model_pool=mock_pool,
        model_config=MagicMock(),
        target_tokens=50,
    )

    assert result == "API timeout is 30 seconds."


@pytest.mark.asyncio
async def test_extract_relevant_memory_empty_reply_returns_none(monkeypatch):
    messages = [
        MagicMock(role="user", content="Hello"),
    ]

    mock_runner = MagicMock()
    mock_runner.generate = AsyncMock(return_value={"reply": "   "})

    mock_handle = MagicMock()
    mock_pool = MagicMock()
    mock_pool.acquire_context.return_value.__aenter__.return_value = mock_handle
    mock_pool.acquire_context.return_value.__aexit__.return_value = None

    monkeypatch.setattr(
        "practorflow.services.history.memory.create_runner",
        lambda handle, knowledge_store=None: mock_runner,
    )

    result = await extract_relevant_memory(
        task="test",
        messages=messages,
        model_pool=mock_pool,
        model_config=MagicMock(),
        target_tokens=50,
    )

    assert result is None


@pytest.mark.asyncio
async def test_extract_relevant_memory_exception_returns_none(monkeypatch):
    messages = [
        MagicMock(role="user", content="Important detail"),
    ]

    mock_runner = MagicMock()
    mock_runner.generate = AsyncMock(side_effect=RuntimeError("LLM failure"))

    mock_handle = MagicMock()
    mock_pool = MagicMock()
    mock_pool.acquire_context.return_value.__aenter__.return_value = mock_handle
    mock_pool.acquire_context.return_value.__aexit__.return_value = None

    monkeypatch.setattr(
        "practorflow.services.history.memory.create_runner",
        lambda handle, knowledge_store=None: mock_runner,
    )

    result = await extract_relevant_memory(
        task="test",
        messages=messages,
        model_pool=mock_pool,
        model_config=MagicMock(),
        target_tokens=50,
    )

    assert result is None
