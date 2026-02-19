"""Tests for Manual Link Storage.

Manual Graph Linker
Tests the ManualLinkManager and ManualLink classes.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from ingestforge.storage.manual_links import (
    ManualLink,
    ManualLinkManager,
    get_link_manager,
    validate_entity,
    validate_relation,
    MAX_ENTITY_LENGTH,
)


class TestManualLink:
    """Tests for ManualLink dataclass."""

    def test_create_link(self) -> None:
        """Test creating a manual link."""
        link = ManualLink(
            link_id="link_test123",
            source_entity="Einstein",
            target_entity="Relativity",
            relation="discovered",
        )

        assert link.link_id == "link_test123"
        assert link.source_entity == "Einstein"
        assert link.target_entity == "Relativity"
        assert link.relation == "discovered"
        assert link.confidence == 1.0
        assert link.notes == ""
        assert link.created_at  # Auto-generated

    def test_to_dict(self) -> None:
        """Test converting link to dictionary."""
        link = ManualLink(
            link_id="link_abc",
            source_entity="A",
            target_entity="B",
            relation="relates_to",
            notes="Test note",
        )

        data = link.to_dict()

        assert data["link_id"] == "link_abc"
        assert data["source_entity"] == "A"
        assert data["target_entity"] == "B"
        assert data["relation"] == "relates_to"
        assert data["notes"] == "Test note"

    def test_from_dict(self) -> None:
        """Test creating link from dictionary."""
        data = {
            "link_id": "link_xyz",
            "source_entity": "X",
            "target_entity": "Y",
            "relation": "causes",
            "confidence": 0.9,
            "notes": "Important",
            "created_at": "2026-01-01T00:00:00Z",
        }

        link = ManualLink.from_dict(data)

        assert link.link_id == "link_xyz"
        assert link.source_entity == "X"
        assert link.target_entity == "Y"
        assert link.relation == "causes"
        assert link.confidence == 0.9

    def test_generate_id(self) -> None:
        """Test ID generation."""
        id1 = ManualLink.generate_id()
        id2 = ManualLink.generate_id()

        assert id1.startswith("link_")
        assert id2.startswith("link_")
        assert id1 != id2


class TestValidation:
    """Tests for validation functions."""

    def test_validate_entity_valid(self) -> None:
        """Test valid entity."""
        result = validate_entity("  Einstein  ", "source")
        assert result == "Einstein"

    def test_validate_entity_empty(self) -> None:
        """Test empty entity raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_entity("", "source")

    def test_validate_entity_too_long(self) -> None:
        """Test too-long entity raises error."""
        long_entity = "x" * (MAX_ENTITY_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_entity(long_entity, "source")

    def test_validate_relation_valid(self) -> None:
        """Test valid relation."""
        result = validate_relation("  Related To  ")
        assert result == "related_to"

    def test_validate_relation_empty(self) -> None:
        """Test empty relation raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_relation("")


class TestManualLinkManager:
    """Tests for ManualLinkManager."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path: Path) -> Path:
        """Create temporary data directory."""
        data_dir = tmp_path / ".data"
        data_dir.mkdir()
        return data_dir

    @pytest.fixture
    def manager(self, temp_data_dir: Path) -> ManualLinkManager:
        """Create manager with temp directory."""
        return ManualLinkManager(temp_data_dir)

    def test_add_link(self, manager: ManualLinkManager) -> None:
        """Test adding a link."""
        link = manager.add("Einstein", "Relativity", "discovered")

        assert link.source_entity == "Einstein"
        assert link.target_entity == "Relativity"
        assert link.relation == "discovered"
        assert manager.count() == 1

    def test_add_link_with_notes(self, manager: ManualLinkManager) -> None:
        """Test adding link with notes."""
        link = manager.add("A", "B", "relates_to", notes="Important connection")

        assert link.notes == "Important connection"

    def test_add_duplicate_raises(self, manager: ManualLinkManager) -> None:
        """Test duplicate link raises error."""
        manager.add("A", "B", "relates_to")

        with pytest.raises(ValueError, match="already exists"):
            manager.add("A", "B", "relates_to")

    def test_get_link(self, manager: ManualLinkManager) -> None:
        """Test getting link by ID."""
        created = manager.add("A", "B", "test")

        retrieved = manager.get(created.link_id)

        assert retrieved is not None
        assert retrieved.link_id == created.link_id

    def test_get_nonexistent(self, manager: ManualLinkManager) -> None:
        """Test getting nonexistent link."""
        result = manager.get("nonexistent")
        assert result is None

    def test_get_for_entity(self, manager: ManualLinkManager) -> None:
        """Test getting links for an entity."""
        manager.add("A", "B", "r1")
        manager.add("A", "C", "r2")
        manager.add("D", "E", "r3")

        links = manager.get_for_entity("A")

        assert len(links) == 2

    def test_delete_link(self, manager: ManualLinkManager) -> None:
        """Test deleting a link."""
        link = manager.add("A", "B", "test")

        deleted = manager.delete(link.link_id)

        assert deleted is True
        assert manager.count() == 0

    def test_delete_nonexistent(self, manager: ManualLinkManager) -> None:
        """Test deleting nonexistent link."""
        deleted = manager.delete("nonexistent")
        assert deleted is False

    def test_list_all(self, manager: ManualLinkManager) -> None:
        """Test listing all links."""
        manager.add("A", "B", "r1")
        manager.add("C", "D", "r2")
        manager.add("E", "F", "r3")

        links = manager.list_all(limit=2)

        assert len(links) == 2

    def test_persistence(self, temp_data_dir: Path) -> None:
        """Test links persist across manager instances."""
        # Create and add
        manager1 = ManualLinkManager(temp_data_dir)
        link = manager1.add("A", "B", "test")

        # Create new instance
        manager2 = ManualLinkManager(temp_data_dir)

        # Should find the link
        retrieved = manager2.get(link.link_id)
        assert retrieved is not None
        assert retrieved.source_entity == "A"

    def test_to_graph_edges(self, manager: ManualLinkManager) -> None:
        """Test converting to graph edge format."""
        manager.add("A", "B", "relates_to", notes="Test")

        edges = manager.to_graph_edges()

        assert len(edges) == 1
        assert edges[0]["source"] == "A"
        assert edges[0]["target"] == "B"
        assert edges[0]["relation"] == "relates_to"
        assert edges[0]["manual"] is True


class TestGetLinkManager:
    """Tests for get_link_manager factory."""

    def test_get_manager_default_dir(self, tmp_path: Path) -> None:
        """Test getting manager with default directory."""
        with patch("ingestforge.storage.manual_links.Path") as mock_path:
            mock_path.cwd.return_value = tmp_path

            manager = get_link_manager()

            assert manager.data_dir == tmp_path / ".data"

    def test_get_manager_custom_dir(self, tmp_path: Path) -> None:
        """Test getting manager with custom directory."""
        custom_dir = tmp_path / "custom"

        manager = get_link_manager(custom_dir)

        assert manager.data_dir == custom_dir
