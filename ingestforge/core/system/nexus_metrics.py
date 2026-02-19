"""
Nexus Prometheus Metrics Definition.

Task 126: Security and performance metrics for federation.
NASA JPL Rule #9: Complete type hints.
"""

from prometheus_client import Counter, Histogram

# --- Security Metrics ---

# Counter for handshake failures
HANDSHAKE_FAILURES = Counter(
    "nexus_handshake_failures_total",
    "Total number of failed federated handshakes",
    ["reason"],
)

# Gauge for unauthorized access attempts (can also be a counter, but gauge allows point-in-time monitoring)
UNAUTHORIZED_ATTEMPTS = Counter(
    "nexus_unauthorized_access_attempts_total",
    "Total number of unauthorized federated access attempts",
    ["peer_id", "reason"],
)

# --- Performance Metrics ---

# Histogram for handshake latency
HANDSHAKE_LATENCY = Histogram(
    "nexus_handshake_latency_seconds",
    "Latency of federated handshakes in seconds",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0),
)


def track_handshake_failure(reason: str) -> None:
    """Increment handshake failure counter."""
    HANDSHAKE_FAILURES.labels(reason=reason).inc()


def track_unauthorized_attempt(peer_id: str, reason: str) -> None:
    """Increment unauthorized access counter."""
    UNAUTHORIZED_ATTEMPTS.labels(peer_id=peer_id, reason=reason).inc()


def get_handshake_timer():
    """Return a timer for handshake latency."""
    return HANDSHAKE_LATENCY.time()
