"""
Background scheduler for periodic cleanup tasks.

Provides an asyncio-based scheduler that runs the orphan document
cleanup service at configurable intervals.
"""

import asyncio
from typing import Optional

from api.services.maintenance.orphan_cleanup_service import OrphanCleanupService
from practorflow.logger.logger import get_logger

logger = get_logger("cleanup-scheduler", level="INFO")


class CleanupScheduler:
    """
    Background scheduler for running cleanup tasks periodically.
    
    Uses asyncio to run cleanup tasks at configurable intervals
    without blocking the main application.
    """
    
    def __init__(
        self,
        cleanup_service: OrphanCleanupService,
        interval_minutes: int = 60,
    ):
        """
        Initialize the cleanup scheduler.
        
        Args:
            cleanup_service: Orphan cleanup service instance.
            interval_minutes: Interval between cleanup runs in minutes.
        """
        self._cleanup_service = cleanup_service
        self._interval_minutes = interval_minutes
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    @property
    def is_running(self) -> bool:
        """Check if the scheduler is currently running."""
        return self._running
    
    @property
    def interval_minutes(self) -> int:
        """Get the cleanup interval in minutes."""
        return self._interval_minutes
    
    async def _run_cleanup(self) -> None:
        """Run a single cleanup cycle."""
        try:
            logger.debug("[CleanupScheduler] Running orphan document cleanup...")
            deleted_count = self._cleanup_service.cleanup()
            if deleted_count > 0:
                logger.info(
                    f"[CleanupScheduler] Cleanup completed. Deleted {deleted_count} orphaned documents"
                )
            else:
                logger.debug("[CleanupScheduler] Cleanup completed. No orphaned documents found")
        except Exception as e:
            logger.error(f"[CleanupScheduler] Cleanup failed: {e}")
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop that runs cleanup at intervals."""
        logger.info(
            f"[CleanupScheduler] Started with interval: {self._interval_minutes} minutes"
        )
        
        interval_seconds = self._interval_minutes * 60
        
        while self._running:
            await asyncio.sleep(interval_seconds)
            
            if self._running:
                await self._run_cleanup()
    
    def start(self) -> None:
        """
        Start the background scheduler.
        
        Creates an asyncio task that runs the cleanup loop.
        Does nothing if scheduler is already running.
        """
        if self._running:
            logger.warning("[CleanupScheduler] Scheduler is already running")
            return
        
        if self._interval_minutes <= 0:
            logger.info("[CleanupScheduler] Scheduler disabled (interval <= 0)")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("[CleanupScheduler] Scheduler started")
    
    async def stop(self) -> None:
        """
        Stop the background scheduler.
        
        Cancels the running task and waits for it to complete.
        Does nothing if scheduler is not running.
        """
        if not self._running:
            logger.debug("[CleanupScheduler] Scheduler is not running")
            return
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        logger.info("[CleanupScheduler] Scheduler stopped")
    
    async def run_now(self) -> int:
        """
        Run cleanup immediately (on-demand).
        
        Returns:
            Number of documents deleted.
        """
        logger.info("[CleanupScheduler] Running cleanup on-demand...")
        await self._run_cleanup()
        return self._cleanup_service.cleanup()