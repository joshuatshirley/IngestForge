"""Resource Guard - Safety checks for system resources.

Monitors RAM, VRAM, and CPU usage to prevent system overload.
Raises ResourceExhaustedError when thresholds are exceeded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
DEFAULT_RAM_THRESHOLD: float = 85.0  # Percent (raised from 80 for transient spikes)
DEFAULT_VRAM_THRESHOLD: float = 90.0  # Percent
DEFAULT_CPU_THRESHOLD: float = 95.0  # Percent
CHECK_INTERVAL_SEC: float = 1.0
SUSTAINED_CHECK_COUNT: int = 3  # Require N consecutive high readings before stopping


class ResourceExhaustedError(Exception):
    """Raised when system resources exceed safe thresholds."""

    def __init__(self, resource: str, usage: float, threshold: float) -> None:
        self.resource = resource
        self.usage = usage
        self.threshold = threshold
        super().__init__(
            f"{resource} usage ({usage:.1f}%) exceeds threshold ({threshold:.1f}%)"
        )


@dataclass
class ResourceStatus:
    """Current resource usage snapshot."""

    ram_percent: float
    vram_percent: Optional[float]  # None if no GPU
    cpu_percent: float
    ram_available_gb: float
    vram_available_gb: Optional[float]

    @property
    def is_safe(self) -> bool:
        """Check if all resources are within safe limits.

        Note: This is a snapshot check. For sustained checks, use ResourceGuard.
        """
        return (
            self.ram_percent < DEFAULT_RAM_THRESHOLD
            and self.cpu_percent < DEFAULT_CPU_THRESHOLD
            and (
                self.vram_percent is None or self.vram_percent < DEFAULT_VRAM_THRESHOLD
            )
        )

    @property
    def is_using_gpu(self) -> bool:
        """Check if GPU is being used (VRAM > 0)."""
        return self.vram_percent is not None and self.vram_percent > 5.0


class ResourceGuard:
    """Monitor and guard system resources.

    Usage:
        guard = ResourceGuard()

        # Check before heavy operation
        guard.check_resources()  # Raises ResourceExhaustedError if unsafe

        # Or get status
        status = guard.get_status()
        if not status.is_safe:
            print("Resources low!")
    """

    def __init__(
        self,
        ram_threshold: float = DEFAULT_RAM_THRESHOLD,
        vram_threshold: float = DEFAULT_VRAM_THRESHOLD,
        cpu_threshold: float = DEFAULT_CPU_THRESHOLD,
        sustained_count: int = SUSTAINED_CHECK_COUNT,
    ) -> None:
        """Initialize resource guard.

        Args:
            ram_threshold: RAM usage threshold (percent)
            vram_threshold: VRAM usage threshold (percent)
            cpu_threshold: CPU usage threshold (percent)
            sustained_count: Number of consecutive high readings before raising
        """
        self._ram_threshold = max(10.0, min(99.0, ram_threshold))
        self._vram_threshold = max(10.0, min(99.0, vram_threshold))
        self._cpu_threshold = max(10.0, min(99.0, cpu_threshold))
        self._sustained_count = max(1, min(10, sustained_count))
        self._gpu_available: Optional[bool] = None
        # Track consecutive high readings for sustained spike detection
        self._ram_high_count: int = 0
        self._vram_high_count: int = 0
        self._cpu_high_count: int = 0

    def _get_ram_usage(self) -> tuple[float, float]:
        """Get RAM usage percent and available GB.

        Returns:
            Tuple of (percent_used, available_gb)
        """
        try:
            import psutil

            mem = psutil.virtual_memory()
            return mem.percent, mem.available / (1024**3)
        except Exception:
            return 0.0, 0.0

    def _get_cpu_usage(self) -> float:
        """Get CPU usage percent.

        Returns:
            CPU usage percent
        """
        try:
            import psutil

            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0

    def _get_vram_usage(self) -> tuple[Optional[float], Optional[float]]:
        """Get VRAM usage percent and available GB.

        Returns:
            Tuple of (percent_used, available_gb) or (None, None) if no GPU
        """
        if self._gpu_available is False:
            return None, None

        # Try NVIDIA GPU via pynvml
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            used_gb = info.used / (1024**3)
            total_gb = info.total / (1024**3)
            available_gb = (info.total - info.used) / (1024**3)
            percent = (info.used / info.total) * 100

            self._gpu_available = True
            return percent, available_gb

        except Exception:
            pass

        # Try torch for CUDA
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                available = total - allocated
                percent = (allocated / total) * 100

                self._gpu_available = True
                return percent, available / (1024**3)

        except Exception:
            pass

        # No GPU available
        self._gpu_available = False
        return None, None

    def get_status(self) -> ResourceStatus:
        """Get current resource status.

        Returns:
            ResourceStatus snapshot
        """
        ram_pct, ram_avail = self._get_ram_usage()
        cpu_pct = self._get_cpu_usage()
        vram_pct, vram_avail = self._get_vram_usage()

        return ResourceStatus(
            ram_percent=ram_pct,
            vram_percent=vram_pct,
            cpu_percent=cpu_pct,
            ram_available_gb=ram_avail,
            vram_available_gb=vram_avail,
        )

    def check_resources(self, raise_on_exceed: bool = True) -> ResourceStatus:
        """Check if resources are within safe limits.

        Only raises after sustained high usage (consecutive checks above threshold).
        This prevents stopping on brief spikes during model loading.

        Args:
            raise_on_exceed: Raise exception if sustained threshold exceeded

        Returns:
            ResourceStatus snapshot

        Raises:
            ResourceExhaustedError: If resources exceed threshold for sustained period
        """
        status = self.get_status()

        # Track RAM usage - only raise on sustained high usage
        if status.ram_percent >= self._ram_threshold:
            self._ram_high_count += 1
            if self._ram_high_count >= self._sustained_count:
                logger.warning(f"RAM usage sustained high: {status.ram_percent:.1f}%")
                if raise_on_exceed:
                    raise ResourceExhaustedError(
                        "RAM", status.ram_percent, self._ram_threshold
                    )
        else:
            self._ram_high_count = 0  # Reset counter on normal reading

        # Track VRAM usage - only raise on sustained high usage
        if (
            status.vram_percent is not None
            and status.vram_percent >= self._vram_threshold
        ):
            self._vram_high_count += 1
            if self._vram_high_count >= self._sustained_count:
                logger.warning(f"VRAM usage sustained high: {status.vram_percent:.1f}%")
                if raise_on_exceed:
                    raise ResourceExhaustedError(
                        "VRAM", status.vram_percent, self._vram_threshold
                    )
        else:
            self._vram_high_count = 0

        # Track CPU usage - only raise on sustained high usage
        if status.cpu_percent >= self._cpu_threshold:
            self._cpu_high_count += 1
            if self._cpu_high_count >= self._sustained_count:
                logger.warning(f"CPU usage sustained high: {status.cpu_percent:.1f}%")
                if raise_on_exceed:
                    raise ResourceExhaustedError(
                        "CPU", status.cpu_percent, self._cpu_threshold
                    )
        else:
            self._cpu_high_count = 0

        return status

    def reset_counters(self) -> None:
        """Reset the sustained high usage counters."""
        self._ram_high_count = 0
        self._vram_high_count = 0
        self._cpu_high_count = 0

    def log_status(self) -> None:
        """Log current resource status."""
        status = self.get_status()
        logger.info(
            f"Resources: RAM={status.ram_percent:.1f}% "
            f"CPU={status.cpu_percent:.1f}% "
            f"VRAM={status.vram_percent:.1f}%"
            if status.vram_percent
            else "VRAM=N/A"
        )


# Module-level singleton for convenience
_guard: Optional[ResourceGuard] = None


def get_resource_guard() -> ResourceGuard:
    """Get the singleton ResourceGuard instance."""
    global _guard
    if _guard is None:
        _guard = ResourceGuard()
    return _guard


def check_resources() -> ResourceStatus:
    """Check resources using the singleton guard.

    Raises:
        ResourceExhaustedError: If resources exceed threshold
    """
    return get_resource_guard().check_resources()


def is_safe_to_proceed() -> bool:
    """Quick check if it's safe to proceed with heavy operations."""
    try:
        status = get_resource_guard().check_resources(raise_on_exceed=False)
        return status.is_safe
    except Exception:
        return True  # Fail open if monitoring unavailable
