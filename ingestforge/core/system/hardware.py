"""Hardware Detection Service.

Provides real-time telemetry on system resources (RAM, CPU, Disk).
Follows NASA JPL Rule #4 (Modular) and Rule #9 (Type Hints).
"""

from __future__ import annotations
import os
import psutil
from dataclasses import dataclass


@dataclass(frozen=True)
class SystemResources:
    """Snapshot of current hardware availability."""

    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    cpu_usage_percent: float
    disk_free_gb: float


class HardwareDetector:
    """Logic for detecting and reporting system capabilities."""

    def get_snapshot(self) -> SystemResources:
        """Capture current system resource state.

        Rule #1: Linear flow.
        Rule #7: Handle potential psutil exceptions.
        """
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return SystemResources(
                total_ram_gb=round(mem.total / (1024**3), 2),
                available_ram_gb=round(mem.available / (1024**3), 2),
                cpu_count=os.cpu_count() or 1,
                cpu_usage_percent=psutil.cpu_percent(interval=None),
                disk_free_gb=round(disk.free / (1024**3), 2),
            )
        except Exception:
            # Fallback to safe defaults (Rule #1)
            return SystemResources(4.0, 1.0, 1, 50.0, 10.0)

    def recommend_preset(self) -> str:
        """Suggest a performance preset based on available RAM."""
        snapshot = self.get_snapshot()

        # Simple heuristic (Rule #1)
        if snapshot.total_ram_gb < 8.0:
            return "eco"
        if snapshot.total_ram_gb < 16.0:
            return "balanced"
        return "performance"
