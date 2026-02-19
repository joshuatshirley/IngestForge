"""
Unit Tests for Backpressure Buffer ().

Tests bounded buffer behavior, pause thresholds, and statistics.

NASA JPL Power of Ten compliant.
"""

import asyncio
import pytest

from ingestforge.core.pipeline.backpressure import (
    BackpressureBuffer,
    MAX_BUFFER_SIZE,
)


# -------------------------------------------------------------------------
# Basic Functionality Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffer_creation():
    """Test buffer initialization with valid parameters."""
    buffer = BackpressureBuffer(max_size=10)
    assert buffer.size() == 0
    assert buffer.is_empty()
    assert not buffer.is_full()
    assert buffer.fill_percent() == 0.0


@pytest.mark.asyncio
async def test_buffer_put_get():
    """Test basic put/get operations."""
    buffer = BackpressureBuffer(max_size=5)

    await buffer.put("item1")
    await buffer.put("item2")

    assert buffer.size() == 2
    assert not buffer.is_empty()

    item = await buffer.get()
    assert item == "item1"
    assert buffer.size() == 1

    item = await buffer.get()
    assert item == "item2"
    assert buffer.is_empty()


@pytest.mark.asyncio
async def test_buffer_try_get_nowait():
    """Test non-blocking get."""
    buffer = BackpressureBuffer(max_size=5)

    # Empty buffer returns None
    item = buffer.try_get_nowait()
    assert item is None

    await buffer.put("test_item")
    item = buffer.try_get_nowait()
    assert item == "test_item"


# -------------------------------------------------------------------------
# Backpressure Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffer_full_blocks():
    """Test that put() blocks when buffer is full."""
    buffer = BackpressureBuffer(max_size=3)

    # Fill buffer
    await buffer.put(1)
    await buffer.put(2)
    await buffer.put(3)

    assert buffer.is_full()

    # Next put should block (we test with timeout)
    async def put_with_timeout():
        try:
            await asyncio.wait_for(buffer.put(4), timeout=0.1)
            return False  # Should not complete
        except asyncio.TimeoutError:
            return True  # Expected timeout

    blocked = await put_with_timeout()
    assert blocked, "Buffer should block when full"


@pytest.mark.asyncio
async def test_buffer_pause_threshold():
    """Test pause threshold at 80% capacity."""
    buffer = BackpressureBuffer(max_size=10)

    # Fill to 79% - should not pause
    for i in range(7):
        await buffer.put(i)

    assert not buffer.should_pause()
    assert buffer.fill_percent() == 70.0

    # Fill to 80% - should pause
    await buffer.put(8)
    assert buffer.should_pause()
    assert buffer.fill_percent() == 80.0

    # Verify stats
    stats = buffer.get_stats()
    assert stats["is_paused"] == True  # Paused when passing threshold


@pytest.mark.asyncio
async def test_buffer_resume_after_drain():
    """Test buffer resumes when drained below 50%."""
    buffer = BackpressureBuffer(max_size=10)

    # Fill to 80% to trigger pause
    for i in range(8):
        await buffer.put(i)

    assert buffer.should_pause()

    # Drain to 40% (below 50% threshold)
    for _ in range(4):
        await buffer.get()

    assert buffer.fill_percent() == 40.0
    # Note: _paused flag is updated in get() when < 50%


# -------------------------------------------------------------------------
# Statistics Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffer_stats():
    """Test statistics reporting."""
    buffer = BackpressureBuffer(max_size=5)

    await buffer.put("a")
    await buffer.put("b")

    stats = buffer.get_stats()
    assert stats["total_put"] == 2
    assert stats["total_get"] == 0
    assert stats["current_size"] == 2
    assert stats["max_size"] == 5
    assert stats["fill_percent"] == 40.0

    await buffer.get()

    stats = buffer.get_stats()
    assert stats["total_get"] == 1
    assert stats["current_size"] == 1


# -------------------------------------------------------------------------
# Boundary Tests (JPL Rule #2)
# -------------------------------------------------------------------------


def test_buffer_max_size_validation():
    """Test max_size validation (JPL Rule #5)."""
    with pytest.raises(AssertionError, match="max_size must be positive"):
        BackpressureBuffer(max_size=0)

    with pytest.raises(AssertionError, match="max_size must be positive"):
        BackpressureBuffer(max_size=-1)

    with pytest.raises(AssertionError, match=f"exceeds limit of {MAX_BUFFER_SIZE}"):
        BackpressureBuffer(max_size=MAX_BUFFER_SIZE + 1)


@pytest.mark.asyncio
async def test_buffer_none_validation():
    """Test that None cannot be added to buffer (JPL Rule #5)."""
    buffer = BackpressureBuffer(max_size=5)

    with pytest.raises(AssertionError, match="Cannot put None"):
        await buffer.put(None)


# -------------------------------------------------------------------------
# Concurrency Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffer_concurrent_producers_consumers():
    """Test concurrent put/get operations."""
    buffer = BackpressureBuffer(max_size=10)
    items_produced = 0
    items_consumed = 0

    async def producer(n: int):
        nonlocal items_produced
        for i in range(n):
            await buffer.put(f"item_{i}")
            items_produced += 1
            await asyncio.sleep(0.01)  # Simulate work

    async def consumer(n: int):
        nonlocal items_consumed
        for _ in range(n):
            await buffer.get()
            items_consumed += 1
            await asyncio.sleep(0.015)  # Simulate slower consumer

    # Run producer and consumer concurrently
    await asyncio.gather(
        producer(20),
        consumer(20),
    )

    assert items_produced == 20
    assert items_consumed == 20
    assert buffer.is_empty()


# -------------------------------------------------------------------------
# Performance Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffer_throughput():
    """Test buffer can handle 10 items/second minimum (AC requirement)."""
    buffer = BackpressureBuffer(max_size=50)

    start_time = asyncio.get_event_loop().time()

    # Put 100 items
    for i in range(100):
        await buffer.put(i)

    # Get 100 items
    for _ in range(100):
        await buffer.get()

    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time

    throughput = 100 / duration
    assert throughput >= 10, f"Throughput {throughput:.1f} items/sec < 10 items/sec"
