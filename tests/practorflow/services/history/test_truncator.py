import pytest
from unittest.mock import MagicMock

from practorflow.services.history.truncator import DeleteSessionService


@pytest.mark.asyncio
async def test_truncate_messages_session_not_found_returns_none():
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = False
    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.truncate_messages(
        session_id="missing",
        from_index=1,
        session_store=session_store,
    )

    assert result is None
    session_store.get.assert_not_called()
    session_store.save.assert_not_called()


@pytest.mark.asyncio
async def test_truncate_messages_successful_truncation():
    session = MagicMock()

    session.truncate_messages.return_value = 3

    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = session
    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.truncate_messages(
        session_id="session_1",
        from_index=2,
        session_store=session_store,
    )

    assert result == 3
    session.truncate_messages.assert_called_once_with(2)
    session_store.save.assert_called_once_with(session)


@pytest.mark.asyncio
async def test_truncate_messages_propagates_value_error():
    session = MagicMock()
    session.truncate_messages.side_effect = ValueError("negative index")

    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = session

    with pytest.raises(ValueError):
        service = DeleteSessionService(knowledge_store, session_store)
        await service.truncate_messages(
            session_id="session_1",
            from_index=-1,
            session_store=session_store,
        )

    session_store.save.assert_not_called()
