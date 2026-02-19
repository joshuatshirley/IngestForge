"""Network lock - Air-gap verification for offline mode.

SEC-002.1: Global Network Interceptor to block non-whitelisted network calls.

Provides network isolation for sensitive operations by intercepting
socket connections and raising SecurityError on unauthorized access."""

from __future__ import annotations

import os
import socket
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Set, Tuple, Iterator, Any

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class NetworkSecurityError(Exception):
    """Raised when network access is blocked in offline mode."""

    pass


@dataclass
class NetworkConfig:
    """Configuration for network lock.

    Attributes:
        enabled: Whether offline mode is active
        whitelist: Set of allowed host:port tuples
        allow_localhost: Whether localhost is allowed
    """

    enabled: bool = False
    whitelist: Set[Tuple[str, int]] = field(default_factory=set)
    allow_localhost: bool = True

    def is_allowed(self, host: str, port: int) -> bool:
        """Check if connection is allowed.

        Args:
            host: Target hostname or IP
            port: Target port

        Returns:
            True if connection is allowed
        """
        if not self.enabled:
            return True

        # Check localhost
        if self.allow_localhost:
            if host in ("localhost", "127.0.0.1", "::1"):
                return True

        # Check whitelist
        if (host, port) in self.whitelist:
            return True

        return False


# Global configuration
_config = NetworkConfig()
_original_connect: Optional[Any] = None
_lock = threading.Lock()


def is_offline_mode() -> bool:
    """Check if offline mode is enabled.

    Returns:
        True if offline mode is active
    """
    # Check environment variable
    env_offline = os.environ.get("INGESTFORGE_OFFLINE_MODE", "").lower()
    if env_offline in ("1", "true", "yes"):
        return True

    # Check config
    return _config.enabled


def enable_offline_mode(
    whitelist: Optional[Set[Tuple[str, int]]] = None,
    allow_localhost: bool = True,
) -> None:
    """Enable offline mode with optional whitelist.

    Args:
        whitelist: Set of (host, port) tuples to allow
        allow_localhost: Whether to allow localhost connections
    """
    global _config, _original_connect

    with _lock:
        _config.enabled = True
        _config.allow_localhost = allow_localhost

        if whitelist:
            _config.whitelist = whitelist
        else:
            _config.whitelist = set()

        # Install socket interceptor if not already installed
        if _original_connect is None:
            _install_interceptor()

        # Set environment variable for child processes
        os.environ["INGESTFORGE_OFFLINE_MODE"] = "1"

        logger.info("Offline mode enabled")


def disable_offline_mode() -> None:
    """Disable offline mode and restore network access."""
    global _config

    with _lock:
        _config.enabled = False
        _config.whitelist = set()

        # Remove environment variable
        os.environ.pop("INGESTFORGE_OFFLINE_MODE", None)

        logger.info("Offline mode disabled")


def add_to_whitelist(host: str, port: int) -> None:
    """Add host:port to network whitelist.

    Args:
        host: Hostname or IP address
        port: Port number
    """
    with _lock:
        _config.whitelist.add((host, port))


def _intercepted_connect(self: socket.socket, address: Any) -> None:
    """Intercepted socket connect that checks whitelist.

    Args:
        self: Socket instance
        address: Connection address (host, port)

    Raises:
        NetworkSecurityError: If connection blocked
    """
    global _original_connect

    # Handle different address formats
    if isinstance(address, tuple) and len(address) >= 2:
        host, port = address[0], address[1]

        if not _config.is_allowed(host, port):
            raise NetworkSecurityError(
                f"Network access blocked in offline mode: {host}:{port}. "
                f"Add to whitelist or disable offline mode."
            )

    # Call original connect
    if _original_connect is not None:
        _original_connect(self, address)


def _install_interceptor() -> None:
    """Install socket connect interceptor."""
    global _original_connect

    if _original_connect is not None:
        return

    _original_connect = socket.socket.connect
    socket.socket.connect = _intercepted_connect  # type: ignore

    logger.debug("Network interceptor installed")


def _uninstall_interceptor() -> None:
    """Restore original socket connect."""
    global _original_connect

    if _original_connect is None:
        return

    socket.socket.connect = _original_connect  # type: ignore
    _original_connect = None

    logger.debug("Network interceptor uninstalled")


@contextmanager
def offline_context(
    whitelist: Optional[Set[Tuple[str, int]]] = None,
    allow_localhost: bool = True,
) -> Iterator[None]:
    """Context manager for temporary offline mode.

    Args:
        whitelist: Allowed connections
        allow_localhost: Allow localhost

    Yields:
        None
    """
    was_enabled = _config.enabled
    old_whitelist = _config.whitelist.copy()

    try:
        enable_offline_mode(whitelist, allow_localhost)
        yield
    finally:
        if not was_enabled:
            disable_offline_mode()
        else:
            _config.whitelist = old_whitelist


def get_network_status() -> dict:
    """Get current network status.

    Returns:
        Dictionary with network status
    """
    return {
        "offline_mode": is_offline_mode(),
        "whitelist_count": len(_config.whitelist),
        "allow_localhost": _config.allow_localhost,
        "interceptor_installed": _original_connect is not None,
    }


def check_network_allowed(host: str, port: int) -> bool:
    """Check if network access is allowed without attempting connection.

    Args:
        host: Target host
        port: Target port

    Returns:
        True if connection would be allowed
    """
    return _config.is_allowed(host, port)
