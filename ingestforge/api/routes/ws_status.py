"""WebSocket Real-Time Job Status.

Async Job Status WebSocket
Epic: API Enhancement

Provides real-time forge job status via WebSocket connections.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_CONNECTIONS)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# JPL Rule #2: Fixed upper bounds
MAX_CONNECTIONS = 100
BROADCAST_INTERVAL_MS = 500
MAX_MESSAGE_SIZE = 65536

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/ws", tags=["websocket"])


@dataclass
class ConnectionInfo:
    """WebSocket connection metadata.

    Rule #9: Complete type hints.
    """

    websocket: WebSocket
    job_ids: Set[str] = field(default_factory=set)
    connected_at: float = 0.0


class ConnectionManager:
    """Manage WebSocket connections for job status updates.

    Rule #9: Complete type hints.
    """

    def __init__(self) -> None:
        """Initialize connection manager."""
        self._connections: Dict[str, ConnectionInfo] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
    ) -> bool:
        """Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection.
            client_id: Unique client identifier.

        Returns:
            True if connection accepted, False if limit reached.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.
        """
        assert websocket is not None, "websocket cannot be None"
        assert client_id, "client_id cannot be empty"

        async with self._lock:
            if len(self._connections) >= MAX_CONNECTIONS:
                logger.warning("Connection limit reached: %d", MAX_CONNECTIONS)
                return False

            await websocket.accept()
            import time

            self._connections[client_id] = ConnectionInfo(
                websocket=websocket,
                connected_at=time.time(),
            )
            logger.info("Client connected: %s", client_id)
            return True

    async def disconnect(self, client_id: str) -> None:
        """Remove a client connection.

        Args:
            client_id: The client to disconnect.

        Rule #4: Function < 60 lines.
        """
        async with self._lock:
            if client_id in self._connections:
                del self._connections[client_id]
                logger.info("Client disconnected: %s", client_id)

    async def subscribe(self, client_id: str, job_id: str) -> bool:
        """Subscribe a client to job updates.

        Args:
            client_id: The subscribing client.
            job_id: The job to subscribe to.

        Returns:
            True if subscribed successfully.

        Rule #4: Function < 60 lines.
        """
        assert job_id, "job_id cannot be empty"

        async with self._lock:
            if client_id not in self._connections:
                return False
            self._connections[client_id].job_ids.add(job_id)
            return True

    async def unsubscribe(self, client_id: str, job_id: str) -> bool:
        """Unsubscribe a client from job updates.

        Args:
            client_id: The client.
            job_id: The job to unsubscribe from.

        Returns:
            True if unsubscribed successfully.

        Rule #4: Function < 60 lines.
        """
        async with self._lock:
            if client_id not in self._connections:
                return False
            self._connections[client_id].job_ids.discard(job_id)
            return True

    async def broadcast_job_update(
        self,
        job_id: str,
        status: Dict[str, Any],
    ) -> int:
        """Broadcast job status to all subscribed clients.

        Args:
            job_id: The job that was updated.
            status: The job status payload.

        Returns:
            Number of clients notified.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.
        """
        assert status is not None, "status cannot be None"

        notified = 0
        disconnected: List[str] = []

        async with self._lock:
            for client_id, info in self._connections.items():
                if job_id in info.job_ids:
                    try:
                        await info.websocket.send_json(
                            {
                                "type": "job_update",
                                "job_id": job_id,
                                "data": status,
                            }
                        )
                        notified += 1
                    except Exception as e:
                        logger.warning("Failed to send to %s: %s", client_id, e)
                        disconnected.append(client_id)

        for client_id in disconnected:
            await self.disconnect(client_id)

        return notified

    def get_connection_count(self) -> int:
        """Get number of active connections.

        Returns:
            Number of active connections.
        """
        return len(self._connections)

    def get_subscriptions(self, client_id: str) -> Set[str]:
        """Get job subscriptions for a client.

        Args:
            client_id: The client ID.

        Returns:
            Set of subscribed job IDs.
        """
        if client_id in self._connections:
            return self._connections[client_id].job_ids.copy()
        return set()


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/status/{client_id}")
async def websocket_status(websocket: WebSocket, client_id: str) -> None:
    """WebSocket endpoint for job status updates.

    Args:
        websocket: The WebSocket connection.
        client_id: Unique client identifier.

    Rule #4: Function < 60 lines.
    """
    connected = await manager.connect(websocket, client_id)
    if not connected:
        await websocket.close(code=1013, reason="Max connections reached")
        return

    try:
        while True:
            data = await websocket.receive_json()
            await _handle_message(client_id, data)
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error("WebSocket error for %s: %s", client_id, e)
        await manager.disconnect(client_id)


async def _handle_message(client_id: str, data: Dict[str, Any]) -> None:
    """Handle incoming WebSocket message.

    Args:
        client_id: The client sending the message.
        data: The message payload.

    Rule #4: Function < 60 lines.
    """
    assert data is not None, "data cannot be None"

    action = data.get("action")
    job_id = data.get("job_id")

    if action == "subscribe" and job_id:
        await manager.subscribe(client_id, job_id)
    elif action == "unsubscribe" and job_id:
        await manager.unsubscribe(client_id, job_id)


async def notify_job_update(job_id: str, status: Dict[str, Any]) -> int:
    """Public function to notify job updates.

    Args:
        job_id: The job that was updated.
        status: The job status payload.

    Returns:
        Number of clients notified.

    Rule #4: Function < 60 lines.
    """
    return await manager.broadcast_job_update(job_id, status)
