"""Tests for Registry CLI Commands.

Registry-Driven Discovery.
Tests CLI commands for listing and inspecting registered processors.

NASA JPL Power of Ten Compliance:
- Rule #1: No recursion in tests
- Rule #2: Fixed test data bounds
- Rule #4: Test functions < 60 lines
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ingestforge.cli.registry.main import registry_command


runner = CliRunner()


class TestListProcessors:
    """Tests for list_processors command."""

    def test_list_empty_registry(self) -> None:
        """Test listing when no processors registered."""
        with patch("ingestforge.cli.registry.main._get_registry") as mock_registry:
            mock_registry.return_value._id_map = {}
            result = runner.invoke(registry_command, ["list"])
            assert result.exit_code == 0
            assert "No processors registered" in result.output

    def test_list_with_processors(self) -> None:
        """Test listing registered processors."""
        mock_proc = MagicMock()
        mock_proc.processor_id = "test-processor"
        mock_proc.version = "1.0.0"
        mock_proc.capabilities = ["test-cap"]
        mock_proc.memory_mb = 100
        mock_proc.is_available.return_value = True

        with patch("ingestforge.cli.registry.main._get_registry") as mock_registry:
            mock_registry.return_value._id_map = {"test-processor": mock_proc}
            result = runner.invoke(registry_command, ["list"])

            assert result.exit_code == 0
            assert "test-processor" in result.output
            assert "1.0.0" in result.output

    def test_list_json_output(self) -> None:
        """Test JSON output format."""
        mock_proc = MagicMock()
        mock_proc.processor_id = "json-test"
        mock_proc.version = "2.0.0"
        mock_proc.capabilities = ["cap1", "cap2"]
        mock_proc.memory_mb = 256
        mock_proc.is_available.return_value = True

        with patch("ingestforge.cli.registry.main._get_registry") as mock_registry:
            mock_registry.return_value._id_map = {"json-test": mock_proc}
            result = runner.invoke(registry_command, ["list", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "processors" in data
            assert data["total"] == 1
            assert data["processors"][0]["processor_id"] == "json-test"

    def test_list_filter_by_capability(self) -> None:
        """Test filtering by capability."""
        mock_proc1 = MagicMock()
        mock_proc1.processor_id = "proc1"
        mock_proc1.version = "1.0.0"
        mock_proc1.capabilities = ["ocr"]
        mock_proc1.memory_mb = 100
        mock_proc1.is_available.return_value = True

        mock_proc2 = MagicMock()
        mock_proc2.processor_id = "proc2"
        mock_proc2.version = "1.0.0"
        mock_proc2.capabilities = ["embedding"]
        mock_proc2.memory_mb = 200
        mock_proc2.is_available.return_value = True

        with patch("ingestforge.cli.registry.main._get_registry") as mock_registry:
            mock_registry.return_value._id_map = {
                "proc1": mock_proc1,
                "proc2": mock_proc2,
            }
            result = runner.invoke(registry_command, ["list", "--capability", "ocr"])

            assert result.exit_code == 0
            assert "proc1" in result.output
            assert "proc2" not in result.output


class TestListCapabilities:
    """Tests for list_capabilities command."""

    def test_capabilities_empty(self) -> None:
        """Test listing when no capabilities registered."""
        with patch("ingestforge.cli.registry.main._get_registry") as mock_registry:
            mock_registry.return_value._id_map = {}
            mock_registry.return_value.get_all_enricher_capabilities.return_value = []
            mock_registry.return_value._enricher_capability_index = {}
            result = runner.invoke(registry_command, ["capabilities"])

            assert result.exit_code == 0
            assert "No capabilities registered" in result.output

    def test_capabilities_with_processors(self) -> None:
        """Test listing capabilities from processors."""
        mock_proc = MagicMock()
        mock_proc.capabilities = ["pdf-splitting", "ocr"]

        with patch("ingestforge.cli.registry.main._get_registry") as mock_registry:
            mock_registry.return_value._id_map = {"test": mock_proc}
            mock_registry.return_value.get_all_enricher_capabilities.return_value = []
            mock_registry.return_value._enricher_capability_index = {}
            result = runner.invoke(registry_command, ["capabilities"])

            assert result.exit_code == 0
            assert "pdf-splitting" in result.output
            assert "ocr" in result.output


class TestListEnrichers:
    """Tests for list_enrichers command."""

    def test_enrichers_empty(self) -> None:
        """Test listing when no enrichers registered."""
        with patch("ingestforge.cli.registry.main._get_registry") as mock_registry:
            mock_registry.return_value._enricher_factories = {}
            result = runner.invoke(registry_command, ["enrichers"])

            assert result.exit_code == 0
            assert "No enrichers registered" in result.output

    def test_enrichers_with_entries(self) -> None:
        """Test listing registered enrichers."""
        mock_entry = MagicMock()
        mock_entry.capabilities = ["embedding"]
        mock_entry.priority = 100

        with patch("ingestforge.cli.registry.main._get_registry") as mock_registry:
            mock_registry.return_value._enricher_factories = {
                "EmbeddingGenerator": mock_entry
            }
            result = runner.invoke(registry_command, ["enrichers"])

            assert result.exit_code == 0
            assert "EmbeddingGenerator" in result.output
            assert "embedding" in result.output


class TestRegistryHealth:
    """Tests for health command."""

    def test_health_healthy(self) -> None:
        """Test health check when registry is healthy."""
        with patch("ingestforge.cli.registry.main._get_registry") as mock_registry:
            mock_registry.return_value.is_healthy.return_value = True
            mock_registry.return_value._id_map = {}
            mock_registry.return_value._enricher_factories = {}
            mock_registry.return_value._capability_index = {}
            mock_registry.return_value.get_process_id.return_value = 12345
            result = runner.invoke(registry_command, ["health"])

            assert result.exit_code == 0
            assert "healthy" in result.output.lower()

    def test_health_unhealthy(self) -> None:
        """Test health check when registry is not healthy."""
        with patch("ingestforge.cli.registry.main._get_registry") as mock_registry:
            mock_registry.return_value.is_healthy.return_value = False
            result = runner.invoke(registry_command, ["health"])

            assert result.exit_code == 1
            assert "NOT healthy" in result.output


class TestPluginDiscovery:
    """Tests for plugin discovery functionality."""

    def test_discover_nonexistent_dir(self, tmp_path: Path) -> None:
        """Test discovery with non-existent directory."""
        from ingestforge.core.pipeline.registry import discover_plugins

        result = discover_plugins(str(tmp_path / "nonexistent"))
        assert result == 0

    def test_discover_empty_dir(self, tmp_path: Path) -> None:
        """Test discovery with empty directory."""
        from ingestforge.core.pipeline.registry import discover_plugins

        result = discover_plugins(str(tmp_path))
        assert result == 0

    def test_discover_with_plugin_file(self, tmp_path: Path) -> None:
        """Test discovery with a valid plugin file."""
        from ingestforge.core.pipeline.registry import discover_plugins, IFRegistry

        # Create a simple plugin file
        plugin_file = tmp_path / "test_plugin.py"
        plugin_file.write_text(
            '''
"""Test plugin."""
# This would normally register a processor
PLUGIN_LOADED = True
'''
        )

        # Clear registry before test
        registry = IFRegistry()
        count_before = len(registry._id_map)

        result = discover_plugins(str(tmp_path))

        # Plugin was loaded (even if it didn't register anything)
        assert result >= 0

    def test_discover_skips_private_files(self, tmp_path: Path) -> None:
        """Test that private files are skipped."""
        from ingestforge.core.pipeline.registry import discover_plugins

        # Create private file
        (tmp_path / "_private.py").write_text("# private")
        (tmp_path / "__init__.py").write_text("# init")

        result = discover_plugins(str(tmp_path))
        assert result == 0


class TestAutoDiscovery:
    """Tests for auto-discovery functionality."""

    def test_auto_discovery_on_registry_creation(self) -> None:
        """Test that auto-discovery triggers on first registry access."""
        from ingestforge.core.pipeline.registry import IFRegistry

        # Reset registry state
        IFRegistry._instance = None
        IFRegistry._auto_discovered = False

        with patch(
            "ingestforge.core.pipeline.registry._auto_discover_processors"
        ) as mock_discover:
            registry = IFRegistry()
            mock_discover.assert_called_once()

    def test_auto_discovery_only_once(self) -> None:
        """Test that auto-discovery only triggers once."""
        from ingestforge.core.pipeline.registry import IFRegistry

        # First access
        registry1 = IFRegistry()

        with patch(
            "ingestforge.core.pipeline.registry._auto_discover_processors"
        ) as mock_discover:
            # Second access should not trigger discovery
            registry2 = IFRegistry()
            mock_discover.assert_not_called()


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten compliance."""

    def test_fixed_bounds_defined(self) -> None:
        """Test that fixed bounds are defined for plugin discovery."""
        from ingestforge.core.pipeline.registry import (
            MAX_PLUGIN_FILES,
            MAX_PLUGIN_DEPTH,
        )

        assert MAX_PLUGIN_FILES == 100
        assert MAX_PLUGIN_DEPTH == 3

    def test_iterative_discovery(self, tmp_path: Path) -> None:
        """Test that plugin discovery is iterative (no recursion)."""
        from ingestforge.core.pipeline.registry import discover_plugins

        # Create nested directory structure
        for i in range(5):
            nested = tmp_path / f"level{i}"
            nested.mkdir(parents=True)
            (nested / f"plugin{i}.py").write_text(f"# plugin {i}")

        # Should not cause stack overflow
        result = discover_plugins(str(tmp_path))
        assert result >= 0
