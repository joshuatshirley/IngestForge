"""Tests for WebSocket Job Status.

Async Job Status WebSocket
Tests real-time job status WebSocket functionality.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestforge.api.routes.ws_status import (
    ConnectionInfo,
    ConnectionManager,
    MAX_CONNECTIONS,
    manager,
    notify_job_update,
)


class TestConnectionInfo:
    """Tests for ConnectionInfo dataclass."""

    def test_create_connection_info(self) -> None:
        """Test creating ConnectionInfo."""
        ws = MagicMock()
        info = ConnectionInfo(websocket=ws, connected_at=1234567890.0)

        assert info.websocket == ws
        assert info.connected_at == 1234567890.0
        assert info.job_ids == set()

    def test_connection_info_default_job_ids(self) -> None:
        """Test default job_ids is empty set."""
        ws = MagicMock()
        info = ConnectionInfo(websocket=ws)

        assert isinstance(info.job_ids, set)
        assert len(info.job_ids) == 0


class TestConnectionManager:
    """Tests for ConnectionManager."""

    @pytest.fixture
    def manager(self) -> ConnectionManager:
        """Create fresh ConnectionManager."""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self) -> AsyncMock:
        """Create mock WebSocket."""
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_connect_success(
        self,
        manager: ConnectionManager,
        mock_websocket: AsyncMock,
    ) -> None:
        """Test successful connection."""
        result = await manager.connect(mock_websocket, "client-1")

        assert result is True
        assert manager.get_connection_count() == 1
        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_limit_reached(
        self,
        manager: ConnectionManager,
        mock_websocket: AsyncMock,
    ) -> None:
        """Test connection rejected when limit reached."""
        # Fill up connections
        for i in range(MAX_CONNECTIONS):
            ws = AsyncMock()
            ws.accept = AsyncMock()
            await manager.connect(ws, f"client-{i}")

        # Next connection should fail
        result = await manager.connect(mock_websocket, "overflow-client")

        assert result is False
        assert manager.get_connection_count() == MAX_CONNECTIONS

    @pytest.mark.asyncio
    async def test_disconnect(
        self,
        manager: ConnectionManager,
        mock_websocket: AsyncMock,
    ) -> None:
        """Test disconnecting a client."""
        await manager.connect(mock_websocket, "client-1")
        assert manager.get_connection_count() == 1

        await manager.disconnect("client-1")

        assert manager.get_connection_count() == 0

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(
        self,
        manager: ConnectionManager,
    ) -> None:
        """Test disconnecting nonexistent client."""
        await manager.disconnect("nonexistent")

        assert manager.get_connection_count() == 0

    @pytest.mark.asyncio
    async def test_subscribe_success(
        self,
        manager: ConnectionManager,
        mock_websocket: AsyncMock,
    ) -> None:
        """Test subscribing to a job."""
        await manager.connect(mock_websocket, "client-1")

        result = await manager.subscribe("client-1", "job-123")

        assert result is True
        assert "job-123" in manager.get_subscriptions("client-1")

    @pytest.mark.asyncio
    async def test_subscribe_nonexistent_client(
        self,
        manager: ConnectionManager,
    ) -> None:
        """Test subscribing with nonexistent client."""
        result = await manager.subscribe("nonexistent", "job-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_unsubscribe_success(
        self,
        manager: ConnectionManager,
        mock_websocket: AsyncMock,
    ) -> None:
        """Test unsubscribing from a job."""
        await manager.connect(mock_websocket, "client-1")
        await manager.subscribe("client-1", "job-123")

        result = await manager.unsubscribe("client-1", "job-123")

        assert result is True
        assert "job-123" not in manager.get_subscriptions("client-1")

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_client(
        self,
        manager: ConnectionManager,
    ) -> None:
        """Test unsubscribing with nonexistent client."""
        result = await manager.unsubscribe("nonexistent", "job-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_broadcast_to_subscribers(
        self,
        manager: ConnectionManager,
    ) -> None:
        """Test broadcasting job update to subscribers."""
        # Setup two clients, one subscribed
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        await manager.connect(ws1, "client-1")
        await manager.connect(ws2, "client-2")
        await manager.subscribe("client-1", "job-123")

        status = {"status": "RUNNING", "progress": 50}
        notified = await manager.broadcast_job_update("job-123", status)

        assert notified == 1
        ws1.send_json.assert_called_once()
        ws2.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed_connections(
        self,
        manager: ConnectionManager,
    ) -> None:
        """Test broadcast removes connections that fail to send."""
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock(side_effect=Exception("Connection lost"))

        await manager.connect(ws, "client-1")
        await manager.subscribe("client-1", "job-123")

        await manager.broadcast_job_update("job-123", {"status": "RUNNING"})

        # Client should be disconnected after failed send
        assert manager.get_connection_count() == 0

    @pytest.mark.asyncio
    async def test_get_subscriptions_nonexistent(
        self,
        manager: ConnectionManager,
    ) -> None:
        """Test getting subscriptions for nonexistent client."""
        subs = manager.get_subscriptions("nonexistent")

        assert subs == set()


class TestNotifyJobUpdate:
    """Tests for notify_job_update function."""

    @pytest.mark.asyncio
    async def test_notify_job_update(self) -> None:
        """Test notify_job_update delegates to manager."""
        with patch.object(
            manager, "broadcast_job_update", new_callable=AsyncMock
        ) as mock_broadcast:
            mock_broadcast.return_value = 5

            result = await notify_job_update("job-123", {"status": "COMPLETED"})

            assert result == 5
            mock_broadcast.assert_called_once_with("job-123", {"status": "COMPLETED"})


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_max_connections_defined(self) -> None:
        """Test MAX_CONNECTIONS bound is defined."""
        assert MAX_CONNECTIONS > 0
        assert MAX_CONNECTIONS <= 1000  # Reasonable upper limit

    def test_rule_5_assertions(self) -> None:
        """Test precondition assertions."""
        manager = ConnectionManager()

        with pytest.raises(AssertionError):
            asyncio.get_event_loop().run_until_complete(
                manager.connect(None, "client-1")  # type: ignore
            )

        with pytest.raises(AssertionError):
            asyncio.get_event_loop().run_until_complete(
                manager.connect(MagicMock(), "")  # Empty client_id
            )

    def test_rule_9_type_hints(self) -> None:
        """Test type hints exist on key methods."""
        import inspect

        manager = ConnectionManager()

        for method_name in [
            "connect",
            "disconnect",
            "subscribe",
            "broadcast_job_update",
        ]:
            method = getattr(manager, method_name)
            sig = inspect.signature(method)
            assert sig.return_annotation is not inspect.Parameter.empty
