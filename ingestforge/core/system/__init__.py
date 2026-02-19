"""System and hardware telemetry services."""

from ingestforge.core.system.hardware import HardwareDetector, SystemResources
from ingestforge.core.system.resource_guard import (
    ResourceGuard,
    ResourceExhaustedError,
    ResourceStatus,
    check_resources,
    get_resource_guard,
    is_safe_to_proceed,
)

__all__ = [
    "HardwareDetector",
    "SystemResources",
    "ResourceGuard",
    "ResourceExhaustedError",
    "ResourceStatus",
    "check_resources",
    "get_resource_guard",
    "is_safe_to_proceed",
]
