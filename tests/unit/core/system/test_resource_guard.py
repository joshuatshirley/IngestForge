"""Unit tests for Resource Guard.

Tests the resource monitoring and safety checks."""

from unittest.mock import patch, MagicMock

import pytest

from ingestforge.core.system.resource_guard import (
    ResourceGuard,
    ResourceExhaustedError,
    ResourceStatus,
    check_resources,
    get_resource_guard,
    is_safe_to_proceed,
    DEFAULT_RAM_THRESHOLD,
    DEFAULT_VRAM_THRESHOLD,
    DEFAULT_CPU_THRESHOLD,
)


class TestResourceStatus:
    """Tests for ResourceStatus dataclass."""

    def test_is_safe_all_low(self) -> None:
        """Verify is_safe returns True when all resources are low."""
        status = ResourceStatus(
            ram_percent=50.0,
            vram_percent=50.0,
            cpu_percent=50.0,
            ram_available_gb=8.0,
            vram_available_gb=4.0,
        )
        assert status.is_safe is True

    def test_is_safe_ram_high(self) -> None:
        """Verify is_safe returns False when RAM is high."""
        status = ResourceStatus(
            ram_percent=90.0,  # Over 85% threshold
            vram_percent=50.0,
            cpu_percent=50.0,
            ram_available_gb=2.0,
            vram_available_gb=4.0,
        )
        assert status.is_safe is False

    def test_is_safe_vram_high(self) -> None:
        """Verify is_safe returns False when VRAM is high."""
        status = ResourceStatus(
            ram_percent=50.0,
            vram_percent=95.0,  # Over 90% threshold
            cpu_percent=50.0,
            ram_available_gb=8.0,
            vram_available_gb=1.0,
        )
        assert status.is_safe is False

    def test_is_safe_no_gpu(self) -> None:
        """Verify is_safe handles None VRAM gracefully."""
        status = ResourceStatus(
            ram_percent=50.0,
            vram_percent=None,  # No GPU
            cpu_percent=50.0,
            ram_available_gb=8.0,
            vram_available_gb=None,
        )
        assert status.is_safe is True


class TestResourceGuard:
    """Tests for ResourceGuard class."""

    def test_init_default_thresholds(self) -> None:
        """Verify default thresholds are set correctly."""
        guard = ResourceGuard()
        assert guard._ram_threshold == DEFAULT_RAM_THRESHOLD
        assert guard._vram_threshold == DEFAULT_VRAM_THRESHOLD
        assert guard._cpu_threshold == DEFAULT_CPU_THRESHOLD

    def test_init_custom_thresholds(self) -> None:
        """Verify custom thresholds are accepted."""
        guard = ResourceGuard(ram_threshold=70.0, vram_threshold=75.0)
        assert guard._ram_threshold == 70.0
        assert guard._vram_threshold == 75.0

    def test_init_clamps_thresholds(self) -> None:
        """Verify thresholds are clamped to valid range."""
        guard = ResourceGuard(ram_threshold=5.0, cpu_threshold=100.0)
        assert guard._ram_threshold == 10.0  # Clamped to min
        assert guard._cpu_threshold == 99.0  # Clamped to max

    def test_get_status_returns_resource_status(self) -> None:
        """Verify get_status returns ResourceStatus object."""
        guard = ResourceGuard()
        status = guard.get_status()
        assert isinstance(status, ResourceStatus)
        assert 0 <= status.ram_percent <= 100
        assert 0 <= status.cpu_percent <= 100

    def test_check_resources_normal(self) -> None:
        """Verify check_resources passes with normal usage."""
        guard = ResourceGuard(ram_threshold=99.0, cpu_threshold=99.0)
        status = guard.check_resources()  # Should not raise
        assert isinstance(status, ResourceStatus)

    def test_check_resources_raises_on_sustained_high_ram(self) -> None:
        """Verify check_resources raises when RAM exceeds threshold for sustained period."""
        guard = ResourceGuard(
            ram_threshold=1.0, sustained_count=3
        )  # Very low threshold
        # First two checks should warn but not raise
        guard.check_resources(raise_on_exceed=False)
        guard.check_resources(raise_on_exceed=False)
        # Third check should raise (sustained high)
        with pytest.raises(ResourceExhaustedError) as exc_info:
            guard.check_resources()
        assert exc_info.value.resource == "RAM"

    def test_check_resources_no_raise_option(self) -> None:
        """Verify check_resources can return without raising."""
        guard = ResourceGuard(ram_threshold=1.0)  # Very low threshold
        status = guard.check_resources(raise_on_exceed=False)
        assert isinstance(status, ResourceStatus)


class TestResourceExhaustedError:
    """Tests for ResourceExhaustedError exception."""

    def test_error_message(self) -> None:
        """Verify error message format."""
        error = ResourceExhaustedError("RAM", 85.0, 80.0)
        assert "RAM" in str(error)
        assert "85.0" in str(error)
        assert "80.0" in str(error)

    def test_error_attributes(self) -> None:
        """Verify error attributes are set correctly."""
        error = ResourceExhaustedError("VRAM", 90.5, 80.0)
        assert error.resource == "VRAM"
        assert error.usage == 90.5
        assert error.threshold == 80.0


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_resource_guard_singleton(self) -> None:
        """Verify get_resource_guard returns same instance."""
        guard1 = get_resource_guard()
        guard2 = get_resource_guard()
        assert guard1 is guard2

    def test_is_safe_to_proceed_returns_bool(self) -> None:
        """Verify is_safe_to_proceed returns boolean."""
        result = is_safe_to_proceed()
        assert isinstance(result, bool)

    def test_check_resources_function(self) -> None:
        """Verify check_resources module function works."""
        # Use high thresholds to avoid raising
        import ingestforge.core.system.resource_guard as rg

        original_guard = rg._guard
        rg._guard = ResourceGuard(ram_threshold=99.0, cpu_threshold=99.0)
        try:
            status = check_resources()
            assert isinstance(status, ResourceStatus)
        finally:
            rg._guard = original_guard


class TestVRAMDetection:
    """Tests for VRAM detection logic."""

    def test_vram_none_when_no_gpu(self) -> None:
        """Verify VRAM is None when no GPU available."""
        guard = ResourceGuard()
        # Force GPU unavailable
        guard._gpu_available = False
        percent, available = guard._get_vram_usage()
        assert percent is None
        assert available is None

    @patch("ingestforge.core.system.resource_guard.ResourceGuard._get_vram_usage")
    def test_vram_included_in_check(self, mock_vram: MagicMock) -> None:
        """Verify VRAM is included in resource check when sustained high."""
        mock_vram.return_value = (95.0, 2.0)  # High VRAM usage
        guard = ResourceGuard(vram_threshold=80.0, sustained_count=3)
        # Build up sustained high readings
        guard.check_resources(raise_on_exceed=False)
        guard.check_resources(raise_on_exceed=False)
        # Third check should raise
        with pytest.raises(ResourceExhaustedError) as exc_info:
            guard.check_resources()
        assert exc_info.value.resource == "VRAM"
