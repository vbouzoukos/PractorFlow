import asyncio
import pytest
from unittest.mock import MagicMock, patch

from api.scheduler.cleanup_scheduler import CleanupScheduler


@pytest.fixture
def cleanup_service():
    service = MagicMock()
    service.cleanup = MagicMock(return_value=0)
    return service


def test_properties(cleanup_service):
    scheduler = CleanupScheduler(cleanup_service, interval_minutes=10)

    assert scheduler.is_running is False
    assert scheduler.interval_minutes == 10


@pytest.mark.asyncio
async def test_run_cleanup_deleted_items(cleanup_service):
    cleanup_service.cleanup.return_value = 2
    scheduler = CleanupScheduler(cleanup_service)

    with patch("api.scheduler.cleanup_scheduler.logger") as logger:
        await scheduler._run_cleanup()
        cleanup_service.cleanup.assert_called_once()
        logger.info.assert_called_once()


@pytest.mark.asyncio
async def test_run_cleanup_no_items(cleanup_service):
    cleanup_service.cleanup.return_value = 0
    scheduler = CleanupScheduler(cleanup_service)

    with patch("api.scheduler.cleanup_scheduler.logger") as logger:
        await scheduler._run_cleanup()
        cleanup_service.cleanup.assert_called_once()
        logger.debug.assert_called()


@pytest.mark.asyncio
async def test_run_cleanup_exception(cleanup_service):
    cleanup_service.cleanup.side_effect = Exception("fail")
    scheduler = CleanupScheduler(cleanup_service)

    with patch("api.scheduler.cleanup_scheduler.logger") as logger:
        await scheduler._run_cleanup()
        logger.error.assert_called_once()


def test_start_already_running(cleanup_service):
    scheduler = CleanupScheduler(cleanup_service)
    scheduler._running = True

    with patch("api.scheduler.cleanup_scheduler.logger") as logger:
        scheduler.start()
        logger.warning.assert_called_once()


def test_start_disabled_interval(cleanup_service):
    scheduler = CleanupScheduler(cleanup_service, interval_minutes=0)

    with patch("api.scheduler.cleanup_scheduler.logger") as logger:
        scheduler.start()
        assert scheduler.is_running is False
        logger.info.assert_called_once()


def test_start_creates_task(cleanup_service):
    scheduler = CleanupScheduler(cleanup_service, interval_minutes=1)

    def create_task(coro):
        # explicitly close the coroutine to avoid "never awaited"
        coro.close()
        return MagicMock()

    with (
        patch(
            "api.scheduler.cleanup_scheduler.asyncio.create_task",
            side_effect=create_task,
        ) as create_task_mock,
        patch("api.scheduler.cleanup_scheduler.logger"),
    ):
        scheduler.start()

        assert scheduler.is_running is True
        create_task_mock.assert_called_once()


@pytest.mark.asyncio
async def test_stop_not_running(cleanup_service):
    scheduler = CleanupScheduler(cleanup_service)

    with patch("api.scheduler.cleanup_scheduler.logger") as logger:
        await scheduler.stop()
        logger.debug.assert_called_once()


@pytest.mark.asyncio
async def test_stop_running_cancels_task(cleanup_service):
    scheduler = CleanupScheduler(cleanup_service)
    scheduler._running = True

    async def dummy_task():
        await asyncio.sleep(0)

    task = asyncio.create_task(dummy_task())
    scheduler._task = task

    with patch("api.scheduler.cleanup_scheduler.logger"):
        await scheduler.stop()

    assert task.cancelled() or task.done()
    assert scheduler.is_running is False
    assert scheduler._task is None


@pytest.mark.asyncio
async def test_scheduler_loop_runs_cleanup_once(cleanup_service):
    scheduler = CleanupScheduler(cleanup_service, interval_minutes=1)
    scheduler._running = True

    async def fast_sleep(_):
        return None

    async def run_cleanup_and_stop():
        scheduler._running = False

    with (
        patch("api.scheduler.cleanup_scheduler.asyncio.sleep", side_effect=fast_sleep),
        patch.object(scheduler, "_run_cleanup", side_effect=run_cleanup_and_stop) as run_cleanup,
        patch("api.scheduler.cleanup_scheduler.logger"),
    ):
        await scheduler._scheduler_loop()

    run_cleanup.assert_called_once()



@pytest.mark.asyncio
async def test_run_now_calls_cleanup_twice_and_returns_value(cleanup_service):
    cleanup_service.cleanup.return_value = 5
    scheduler = CleanupScheduler(cleanup_service)

    with patch("api.scheduler.cleanup_scheduler.logger"):
        result = await scheduler.run_now()

    assert cleanup_service.cleanup.call_count == 2
    assert result == 5

