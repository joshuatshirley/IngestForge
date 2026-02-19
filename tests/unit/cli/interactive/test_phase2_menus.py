"""Unit tests for Phase 2 menu features.

Tests the new submenus and handlers added in Phase 2:
- Organization (tags, bookmarks, annotations)
- Legal Research (court opinions)
- Security Research (CVE)
- Extended Export (pack, unpack, outline)
- Extended Citations (writing cite commands)"""

from typing import Set


from ingestforge.cli.interactive.menu import (
    # New submenus
    ORGANIZE_SUBMENU,
    LEGAL_SUBMENU,
    SECURITY_SUBMENU,
    CITATION_SUBMENU,
    EXPORT_SUBMENU,
    # Updated menus
    DISCOVERY_MENU,
)


class TestOrganizeSubmenu:
    """Tests for Organization submenu."""

    def test_has_required_items(self) -> None:
        """Verify Organization submenu has all required items."""
        keys: Set[str] = {item[0] for item in ORGANIZE_SUBMENU}
        required: Set[str] = {"1", "2", "3", "4", "5", "6", "7", "b"}
        assert keys == required

    def test_tag_commands_present(self) -> None:
        """Verify tag-related commands are present."""
        labels: Set[str] = {item[1] for item in ORGANIZE_SUBMENU}
        assert "Add Tag" in labels
        assert "List Tags" in labels

    def test_bookmark_commands_present(self) -> None:
        """Verify bookmark commands are present."""
        labels: Set[str] = {item[1] for item in ORGANIZE_SUBMENU}
        assert "Bookmark" in labels
        assert "Bookmarks" in labels

    def test_annotation_commands_present(self) -> None:
        """Verify annotation commands are present."""
        labels: Set[str] = {item[1] for item in ORGANIZE_SUBMENU}
        assert "Annotate" in labels
        assert "Annotations" in labels


class TestLegalSubmenu:
    """Tests for Legal Research submenu."""

    def test_has_required_items(self) -> None:
        """Verify Legal submenu has court commands."""
        keys: Set[str] = {item[0] for item in LEGAL_SUBMENU}
        required: Set[str] = {"1", "2", "3", "4", "b"}
        assert keys == required

    def test_court_commands_present(self) -> None:
        """Verify all court commands are present."""
        labels: Set[str] = {item[1] for item in LEGAL_SUBMENU}
        assert "Search Opinions" in labels
        assert "Case Details" in labels
        assert "Download Opinion" in labels
        assert "Jurisdictions" in labels


class TestSecuritySubmenu:
    """Tests for Security Research submenu."""

    def test_has_required_items(self) -> None:
        """Verify Security submenu has CVE commands."""
        keys: Set[str] = {item[0] for item in SECURITY_SUBMENU}
        required: Set[str] = {"1", "2", "b"}
        assert keys == required

    def test_cve_commands_present(self) -> None:
        """Verify CVE commands are present."""
        labels: Set[str] = {item[1] for item in SECURITY_SUBMENU}
        assert "Search CVE" in labels
        assert "CVE Details" in labels


class TestCitationSubmenu:
    """Tests for extended Citation submenu."""

    def test_has_required_items(self) -> None:
        """Verify Citation submenu has all required items."""
        keys: Set[str] = {item[0] for item in CITATION_SUBMENU}
        required: Set[str] = {"1", "2", "3", "4", "5", "6", "7", "8", "b"}
        assert keys == required

    def test_writing_cite_commands_present(self) -> None:
        """Verify writing cite commands are present (Phase 2 additions)."""
        labels: Set[str] = {item[1] for item in CITATION_SUBMENU}
        assert "Check in Doc" in labels
        assert "Format in Doc" in labels
        assert "Insert" in labels

    def test_core_citation_commands_present(self) -> None:
        """Verify core citation commands remain."""
        labels: Set[str] = {item[1] for item in CITATION_SUBMENU}
        assert "Extract" in labels
        assert "Bibliography" in labels
        assert "Validate" in labels


class TestExportSubmenu:
    """Tests for extended Export submenu."""

    def test_has_required_items(self) -> None:
        """Verify Export submenu has all required items."""
        keys: Set[str] = {item[0] for item in EXPORT_SUBMENU}
        required: Set[str] = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "b"}
        assert keys == required

    def test_pack_commands_present(self) -> None:
        """Verify pack/unpack commands are present (Phase 2 additions)."""
        labels: Set[str] = {item[1] for item in EXPORT_SUBMENU}
        assert "Create Package" in labels
        assert "Import Package" in labels
        assert "Package Info" in labels

    def test_outline_command_present(self) -> None:
        """Verify outline export is present (Phase 2 addition)."""
        labels: Set[str] = {item[1] for item in EXPORT_SUBMENU}
        assert "Outline" in labels

    def test_core_export_commands_remain(self) -> None:
        """Verify core export commands remain."""
        labels: Set[str] = {item[1] for item in EXPORT_SUBMENU}
        assert "Markdown" in labels
        assert "JSON" in labels
        assert "PDF" in labels


class TestDataMenuUpdates:
    """Tests for INGEST_ORGANIZE_MENU (Phase 3 renamed from DATA_MENU)."""

    def test_organize_option_added(self) -> None:
        """Verify Organize option is in INGEST_ORGANIZE_MENU."""
        from ingestforge.cli.interactive.menu import INGEST_ORGANIZE_MENU

        labels: Set[str] = {item[1] for item in INGEST_ORGANIZE_MENU}
        assert "Organize" in labels

    def test_has_correct_item_count(self) -> None:
        """Verify INGEST_ORGANIZE_MENU has correct number of items."""
        from ingestforge.cli.interactive.menu import INGEST_ORGANIZE_MENU

        # 6 options + back = 7 (Phase 3 consolidated)
        assert len(INGEST_ORGANIZE_MENU) == 7


class TestDiscoveryMenuUpdates:
    """Tests for DISCOVERY_MENU updates."""

    def test_legal_option_added(self) -> None:
        """Verify Legal Research option was added."""
        labels: Set[str] = {item[1] for item in DISCOVERY_MENU}
        assert "Legal Research" in labels

    def test_security_option_added(self) -> None:
        """Verify Security Research option was added."""
        labels: Set[str] = {item[1] for item in DISCOVERY_MENU}
        assert "Security Research" in labels

    def test_has_correct_item_count(self) -> None:
        """Verify DISCOVERY_MENU has correct number of items."""
        # 7 options + back = 8
        assert len(DISCOVERY_MENU) == 8


class TestMenuStructureIntegrity:
    """Tests for menu structure integrity."""

    def test_all_menus_have_back_option(self) -> None:
        """Verify all submenus have a back option."""
        menus: list = [
            ORGANIZE_SUBMENU,
            LEGAL_SUBMENU,
            SECURITY_SUBMENU,
            CITATION_SUBMENU,
            EXPORT_SUBMENU,
        ]
        for menu in menus:
            keys: Set[str] = {item[0] for item in menu}
            assert "b" in keys, f"Menu missing back option: {menu}"

    def test_no_duplicate_keys_in_menus(self) -> None:
        """Verify no duplicate keys in any menu."""
        menus: list = [
            ("ORGANIZE_SUBMENU", ORGANIZE_SUBMENU),
            ("LEGAL_SUBMENU", LEGAL_SUBMENU),
            ("SECURITY_SUBMENU", SECURITY_SUBMENU),
            ("CITATION_SUBMENU", CITATION_SUBMENU),
            ("EXPORT_SUBMENU", EXPORT_SUBMENU),
        ]
        for name, menu in menus:
            keys: list = [item[0] for item in menu]
            assert len(keys) == len(set(keys)), f"Duplicate keys in {name}"

    def test_menu_item_structure(self) -> None:
        """Verify all menu items have correct structure (key, label, desc, action)."""
        menus: list = [
            ORGANIZE_SUBMENU,
            LEGAL_SUBMENU,
            SECURITY_SUBMENU,
            CITATION_SUBMENU,
            EXPORT_SUBMENU,
        ]
        for menu in menus:
            for item in menu:
                assert len(item) == 4, f"Invalid item structure: {item}"
                key, label, desc, action = item
                assert isinstance(key, str) and len(key) == 1
                assert isinstance(label, str) and len(label) > 0
                assert isinstance(desc, str)
                assert isinstance(action, str)
