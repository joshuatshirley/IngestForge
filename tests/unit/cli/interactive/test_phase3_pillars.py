"""Unit tests for Phase 3 pillar restructure.

Tests the 4-pillar + System menu structure:
- [1] Ingest & Organize
- [2] Search & Discover
- [3] Analyze & Understand
- [4] Create & Export
- [0] System"""

from typing import Set


from ingestforge.cli.interactive.menu import (
    # New 4-pillar menus
    MAIN_MENU,
    INGEST_ORGANIZE_MENU,
    ADD_DOCS_SUBMENU,
    SEARCH_DISCOVER_MENU,
    ANALYZE_MENU,
    CREATE_EXPORT_MENU,
    SYSTEM_MENU,
    # Handlers
    handle_ingest_organize_menu,
    handle_search_discover_menu,
    handle_analyze_menu,
    handle_create_export_menu,
    handle_system_menu,
)


class TestMainMenuStructure:
    """Tests for main menu 4-pillar structure."""

    def test_main_menu_has_four_pillars_plus_system(self) -> None:
        """Verify main menu has 4 pillars + System + Status + Quit."""
        assert len(MAIN_MENU) == 7  # 4 pillars + system + status + quit

    def test_main_menu_pillar_keys(self) -> None:
        """Verify main menu uses correct keys."""
        keys: Set[str] = {item[0] for item in MAIN_MENU}
        expected: Set[str] = {"1", "2", "3", "4", "0", "s", "q"}
        assert keys == expected

    def test_main_menu_pillar_labels(self) -> None:
        """Verify main menu has correct pillar labels."""
        labels: Set[str] = {item[1] for item in MAIN_MENU}
        assert "Ingest & Organize" in labels
        assert "Search & Discover" in labels
        assert "Analyze & Understand" in labels
        assert "Create & Export" in labels
        assert "System" in labels


class TestIngestOrganizePillar:
    """Tests for [1] Ingest & Organize pillar."""

    def test_has_required_items(self) -> None:
        """Verify pillar has all required items."""
        keys: Set[str] = {item[0] for item in INGEST_ORGANIZE_MENU}
        required: Set[str] = {"1", "2", "3", "4", "5", "6", "b"}
        assert keys == required

    def test_has_add_docs_option(self) -> None:
        """Verify Add Documents option exists."""
        labels: Set[str] = {item[1] for item in INGEST_ORGANIZE_MENU}
        assert "Add Documents" in labels

    def test_has_organize_option(self) -> None:
        """Verify Organize option exists."""
        labels: Set[str] = {item[1] for item in INGEST_ORGANIZE_MENU}
        assert "Organize" in labels

    def test_add_docs_submenu_structure(self) -> None:
        """Verify Add Documents submenu has file/youtube/audio/batch."""
        labels: Set[str] = {item[1] for item in ADD_DOCS_SUBMENU}
        assert "File/Folder" in labels
        assert "YouTube Video" in labels
        assert "Audio File" in labels
        assert "Batch Import" in labels


class TestSearchDiscoverPillar:
    """Tests for [2] Search & Discover pillar."""

    def test_has_required_items(self) -> None:
        """Verify pillar has all required items."""
        keys: Set[str] = {item[0] for item in SEARCH_DISCOVER_MENU}
        required: Set[str] = {"1", "2", "3", "4", "5", "6", "7", "b"}
        assert keys == required

    def test_has_query_option(self) -> None:
        """Verify Query option exists (moved from Core)."""
        labels: Set[str] = {item[1] for item in SEARCH_DISCOVER_MENU}
        assert "Query Knowledge Base" in labels

    def test_has_agent_option(self) -> None:
        """Verify Agent option exists (moved from Core)."""
        labels: Set[str] = {item[1] for item in SEARCH_DISCOVER_MENU}
        assert "Agent Research" in labels

    def test_has_legal_and_security(self) -> None:
        """Verify Legal and Security research options exist."""
        labels: Set[str] = {item[1] for item in SEARCH_DISCOVER_MENU}
        assert "Legal Research" in labels
        assert "Security Research" in labels


class TestAnalyzePillar:
    """Tests for [3] Analyze & Understand pillar."""

    def test_has_required_items(self) -> None:
        """Verify pillar has all required items."""
        keys: Set[str] = {item[0] for item in ANALYZE_MENU}
        required: Set[str] = {"1", "2", "3", "4", "5", "6", "7", "b"}
        assert keys == required

    def test_has_all_analysis_types(self) -> None:
        """Verify all analysis types are present."""
        labels: Set[str] = {item[1] for item in ANALYZE_MENU}
        assert "Comprehension" in labels
        assert "Argument Analysis" in labels
        assert "Literary Analysis" in labels
        assert "Content Analysis" in labels
        assert "Code Analysis" in labels
        assert "Fact Checker" in labels


class TestCreateExportPillar:
    """Tests for [4] Create & Export pillar."""

    def test_has_required_items(self) -> None:
        """Verify pillar has all required items."""
        keys: Set[str] = {item[0] for item in CREATE_EXPORT_MENU}
        required: Set[str] = {"1", "2", "3", "4", "5", "b"}
        assert keys == required

    def test_has_study_tools(self) -> None:
        """Verify Study Tools option exists."""
        labels: Set[str] = {item[1] for item in CREATE_EXPORT_MENU}
        assert "Study Tools" in labels

    def test_has_export_option(self) -> None:
        """Verify Export option exists."""
        labels: Set[str] = {item[1] for item in CREATE_EXPORT_MENU}
        assert "Export" in labels

    def test_has_research_summary(self) -> None:
        """Verify Research Summary option exists (new in Phase 3)."""
        labels: Set[str] = {item[1] for item in CREATE_EXPORT_MENU}
        assert "Research Summary" in labels


class TestSystemPillar:
    """Tests for [0] System pillar."""

    def test_has_required_items(self) -> None:
        """Verify pillar has all required items."""
        keys: Set[str] = {item[0] for item in SYSTEM_MENU}
        required: Set[str] = {"1", "2", "3", "4", "5", "6", "7", "8", "b"}
        assert keys == required

    def test_has_quick_start(self) -> None:
        """Verify Quick Start is in System (moved from Core)."""
        labels: Set[str] = {item[1] for item in SYSTEM_MENU}
        assert "Quick Start" in labels

    def test_has_admin_options(self) -> None:
        """Verify admin options are present."""
        labels: Set[str] = {item[1] for item in SYSTEM_MENU}
        assert "LLM Settings" in labels
        assert "Configuration" in labels
        assert "Storage" in labels
        assert "Maintenance" in labels
        assert "API Server" in labels


class TestHandlerFunctionsExist:
    """Tests that all pillar handlers exist."""

    def test_ingest_organize_handler_exists(self) -> None:
        """Verify ingest/organize handler is callable."""
        assert callable(handle_ingest_organize_menu)

    def test_search_discover_handler_exists(self) -> None:
        """Verify search/discover handler is callable."""
        assert callable(handle_search_discover_menu)

    def test_analyze_handler_exists(self) -> None:
        """Verify analyze handler is callable."""
        assert callable(handle_analyze_menu)

    def test_create_export_handler_exists(self) -> None:
        """Verify create/export handler is callable."""
        assert callable(handle_create_export_menu)

    def test_system_handler_exists(self) -> None:
        """Verify system handler is callable."""
        assert callable(handle_system_menu)


class TestMenuStructureIntegrity:
    """Tests for overall menu structure integrity."""

    def test_all_menus_have_back_option(self) -> None:
        """Verify all pillar menus have a back option."""
        menus = [
            INGEST_ORGANIZE_MENU,
            SEARCH_DISCOVER_MENU,
            ANALYZE_MENU,
            CREATE_EXPORT_MENU,
            SYSTEM_MENU,
        ]
        for menu in menus:
            keys: Set[str] = {item[0] for item in menu}
            assert "b" in keys

    def test_no_duplicate_keys_in_pillar_menus(self) -> None:
        """Verify no duplicate keys in any pillar menu."""
        menus = [
            ("INGEST_ORGANIZE_MENU", INGEST_ORGANIZE_MENU),
            ("SEARCH_DISCOVER_MENU", SEARCH_DISCOVER_MENU),
            ("ANALYZE_MENU", ANALYZE_MENU),
            ("CREATE_EXPORT_MENU", CREATE_EXPORT_MENU),
            ("SYSTEM_MENU", SYSTEM_MENU),
        ]
        for name, menu in menus:
            keys = [item[0] for item in menu]
            assert len(keys) == len(set(keys)), f"Duplicate keys in {name}"

    def test_pillar_reduction(self) -> None:
        """Verify we went from 6 pillars to 4 + System."""
        # Main menu should have: 4 pillars + system + status + quit = 7 items
        pillar_count = sum(
            1 for item in MAIN_MENU if item[0] in {"1", "2", "3", "4", "0"}
        )
        assert pillar_count == 5  # 4 pillars + 1 system
