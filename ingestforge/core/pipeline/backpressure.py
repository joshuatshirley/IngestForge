"""
Backpressure Buffer for Streaming Pipeline ().

Provides bounded buffer for managing flow control between pipeline execution
and SSE streaming to prevent memory exhaustion.

NASA JPL Power of Ten compliant.
"""

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_BUFFER_SIZE = 50
PAUSE_THRESHOLD_PERCENT = 80  # Pause pipeline at 80% capacity


class BackpressureBuffer:
    """
    Bounded async queue for backpressure management.

    Streaming Foundry API.
    Prevents memory exhaustion by pausing pipeline when client can't keep up.

    JPL Compliance:
    - Rule #2: Fixed MAX_BUFFER_SIZE = 50
    - Rule #4: All methods < 60 lines
    - Rule #5: Assert preconditions
    - Rule #9: 100% type hints
    """

    def __init__(self, max_size: int = MAX_BUFFER_SIZE) -> None:
        """
        Initialize bounded buffer.

        Args:
            max_size: Maximum items in buffer (default 50)

        Raises:
            AssertionError: If max_size invalid

        Rule #5: Assert preconditions.
        """
        assert max_size > 0, "max_size must be positive"
        assert (
            max_size <= MAX_BUFFER_SIZE
        ), f"max_size exceeds limit of {MAX_BUFFER_SIZE}"

        self._max_size: int = max_size
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_size)
        self._paused: bool = False
        self._total_put: int = 0
        self._total_get: int = 0

    async def put(self, item: Any) -> None:
        """
        Add item to buffer (blocks if full).

        Args:
            item: Item to add to buffer

        Rule #2: Bounded by max_size.
        Rule #4: < 60 lines.
        """
        assert item is not None, "Cannot put None into buffer"

        await self._queue.put(item)
        self._total_put += 1

        if self.should_pause() and not self._paused:
            self._paused = True
            logger.warning(f"Buffer at {self.fill_percent():.1f}% - pausing pipeline")

    async def get(self) -> Any:
        """
        Retrieve next item from buffer.

        Returns:
            Next item in queue

        Rule #4: < 60 lines.
        """
        item = await self._queue.get()
        self._total_get += 1

        # Resume if buffer drains below 50%
        if self._paused and self.fill_percent() < 50:
            self._paused = False
            logger.info("Buffer drained - resuming pipeline")

        return item

    def try_get_nowait(self) -> Optional[Any]:
        """
        Try to get item without blocking.

        Returns:
            Item if available, None otherwise

        Rule #4: < 60 lines.
        """
        try:
            item = self._queue.get_nowait()
            self._total_get += 1
            return item
        except asyncio.QueueEmpty:
            return None

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._queue.empty()

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self._queue.full()

    def should_pause(self) -> bool:
        """
        Check if pipeline should pause (80% threshold).

        Returns:
            True if buffer >= 80% full

        Rule #2: Fixed PAUSE_THRESHOLD_PERCENT = 80.
        """
        return self.fill_percent() >= PAUSE_THRESHOLD_PERCENT

    def size(self) -> int:
        """Get current buffer size."""
        return self._queue.qsize()

    def fill_percent(self) -> float:
        """
        Calculate buffer fill percentage.

        Returns:
            Percentage (0-100)

        Rule #4: < 60 lines.
        """
        if self._max_size == 0:
            return 0.0
        return (self.size() / self._max_size) * 100.0

    def get_stats(self) -> dict[str, Any]:
        """
        Get buffer statistics for monitoring.

        Returns:
            Dict with total_put, total_get, current_size, fill_percent

        Rule #4: < 60 lines.
        """
        return {
            "total_put": self._total_put,
            "total_get": self._total_get,
            "current_size": self.size(),
            "max_size": self._max_size,
            "fill_percent": self.fill_percent(),
            "is_paused": self._paused,
        }
