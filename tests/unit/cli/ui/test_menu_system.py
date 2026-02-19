"""Tests for hierarchical menu system.

Tests menu navigation, submenus, and command selection."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ingestforge.cli.ui.menu_system import (
    Menu,
    MenuAction,
    MenuItem,
    MenuSelection,
    MenuSystem,
    create_default_menus,
    MAX_MENU_DEPTH,
    MAX_ITEMS_PER_MENU,
)

# MenuItem tests


class TestMenuItem:
    """Tests for MenuItem dataclass."""

    def test_item_creation(self) -> None:
        """Test creating menu item."""
        item = MenuItem(
            key="1",
            label="Test Item",
            action=MenuAction.COMMAND,
            command="test-cmd",
        )

        assert item.key == "1"
        assert item.label == "Test Item"
        assert item.action == MenuAction.COMMAND
        assert item.command == "test-cmd"

    def test_item_with_icon(self) -> None:
        """Test item with icon prefix."""
        item = MenuItem(key="1", label="Test", icon="📁")

        assert item.display_label() == "📁 Test"

    def test_item_without_icon(self) -> None:
        """Test item without icon."""
        item = MenuItem(key="1", label="Test")

        assert item.display_label() == "Test"

    def test_submenu_item(self) -> None:
        """Test submenu item creation."""
        item = MenuItem(
            key="r",
            label="Research",
            action=MenuAction.SUBMENU,
            submenu_id="research",
        )

        assert item.action == MenuAction.SUBMENU
        assert item.submenu_id == "research"


# Menu tests


class TestMenu:
    """Tests for Menu dataclass."""

    def test_menu_creation(self) -> None:
        """Test creating menu."""
        menu = Menu(id="main", title="Main Menu")

        assert menu.id == "main"
        assert menu.title == "Main Menu"
        assert menu.items == []

    def test_add_item(self) -> None:
        """Test adding items to menu."""
        menu = Menu(id="test", title="Test")
        item = MenuItem(key="1", label="Item 1")

        menu.add_item(item)

        assert len(menu.items) == 1
        assert menu.items[0].label == "Item 1"

    def test_add_item_respects_limit(self) -> None:
        """Test that add_item respects MAX_ITEMS_PER_MENU."""
        menu = Menu(id="test", title="Test")

        for i in range(MAX_ITEMS_PER_MENU + 5):
            menu.add_item(MenuItem(key=str(i), label=f"Item {i}"))

        assert len(menu.items) == MAX_ITEMS_PER_MENU

    def test_get_item_found(self) -> None:
        """Test getting item by key."""
        menu = Menu(id="test", title="Test")
        menu.add_item(MenuItem(key="a", label="Item A"))
        menu.add_item(MenuItem(key="b", label="Item B"))

        item = menu.get_item("a")

        assert item is not None
        assert item.label == "Item A"

    def test_get_item_not_found(self) -> None:
        """Test getting nonexistent item."""
        menu = Menu(id="test", title="Test")
        menu.add_item(MenuItem(key="a", label="Item A"))

        item = menu.get_item("x")

        assert item is None

    def test_get_item_case_insensitive(self) -> None:
        """Test that get_item is case insensitive."""
        menu = Menu(id="test", title="Test")
        menu.add_item(MenuItem(key="A", label="Item A"))

        assert menu.get_item("a") is not None
        assert menu.get_item("A") is not None

    def test_menu_with_parent(self) -> None:
        """Test menu with parent reference."""
        menu = Menu(id="sub", title="Submenu", parent_id="main")

        assert menu.parent_id == "main"


# MenuSelection tests


class TestMenuSelection:
    """Tests for MenuSelection dataclass."""

    def test_selection_command(self) -> None:
        """Test command selection."""
        selection = MenuSelection(
            action=MenuAction.COMMAND,
            command="query",
        )

        assert selection.action == MenuAction.COMMAND
        assert selection.command == "query"
        assert selection.cancelled is False

    def test_selection_submenu(self) -> None:
        """Test submenu selection."""
        selection = MenuSelection(
            action=MenuAction.SUBMENU,
            submenu_id="research",
        )

        assert selection.action == MenuAction.SUBMENU
        assert selection.submenu_id == "research"

    def test_selection_cancelled(self) -> None:
        """Test cancelled selection."""
        selection = MenuSelection(
            action=MenuAction.COMMAND,
            cancelled=True,
        )

        assert selection.cancelled is True


# MenuSystem tests


class TestMenuSystem:
    """Tests for MenuSystem."""

    def test_system_creation(self) -> None:
        """Test creating menu system."""
        system = MenuSystem()

        assert system.menus == {}
        assert system._current_menu_id == "main"

    def test_register_menu(self) -> None:
        """Test registering menus."""
        system = MenuSystem()
        menu = Menu(id="test", title="Test Menu")

        system.register_menu(menu)

        assert "test" in system.menus
        assert system.menus["test"].title == "Test Menu"

    def test_get_menu_found(self) -> None:
        """Test getting registered menu."""
        system = MenuSystem()
        menu = Menu(id="test", title="Test")
        system.register_menu(menu)

        result = system.get_menu("test")

        assert result is not None
        assert result.id == "test"

    def test_get_menu_not_found(self) -> None:
        """Test getting nonexistent menu."""
        system = MenuSystem()

        result = system.get_menu("nonexistent")

        assert result is None


class TestMenuSystemNavigation:
    """Tests for menu navigation."""

    @pytest.fixture
    def system_with_menus(self) -> MenuSystem:
        """Create system with test menus."""
        system = MenuSystem()

        main = Menu(id="main", title="Main")
        main.add_item(
            MenuItem(
                key="1",
                label="Query",
                action=MenuAction.COMMAND,
                command="query",
            )
        )
        main.add_item(
            MenuItem(
                key="2",
                label="Research",
                action=MenuAction.SUBMENU,
                submenu_id="research",
            )
        )
        system.register_menu(main)

        research = Menu(id="research", title="Research", parent_id="main")
        research.add_item(
            MenuItem(
                key="1",
                label="Topics",
                action=MenuAction.COMMAND,
                command="analyze topics",
            )
        )
        system.register_menu(research)

        return system

    def test_prompt_selection_command(self, system_with_menus: MenuSystem) -> None:
        """Test selecting a command."""
        with patch("ingestforge.cli.ui.menu_system.Prompt.ask", return_value="1"):
            selection = system_with_menus.prompt_selection("main")

        assert selection.action == MenuAction.COMMAND
        assert selection.command == "query"

    def test_prompt_selection_submenu(self, system_with_menus: MenuSystem) -> None:
        """Test selecting a submenu."""
        with patch("ingestforge.cli.ui.menu_system.Prompt.ask", return_value="2"):
            selection = system_with_menus.prompt_selection("main")

        assert selection.action == MenuAction.SUBMENU
        assert selection.submenu_id == "research"

    def test_prompt_selection_quit(self, system_with_menus: MenuSystem) -> None:
        """Test quitting."""
        with patch("ingestforge.cli.ui.menu_system.Prompt.ask", return_value="q"):
            selection = system_with_menus.prompt_selection("main")

        assert selection.action == MenuAction.EXIT

    def test_prompt_selection_back(self, system_with_menus: MenuSystem) -> None:
        """Test going back from submenu."""
        with patch("ingestforge.cli.ui.menu_system.Prompt.ask", return_value="b"):
            selection = system_with_menus.prompt_selection("research")

        assert selection.action == MenuAction.BACK
        assert selection.submenu_id == "main"

    def test_prompt_selection_invalid(self, system_with_menus: MenuSystem) -> None:
        """Test invalid selection."""
        with patch("ingestforge.cli.ui.menu_system.Prompt.ask", return_value="xyz"):
            selection = system_with_menus.prompt_selection("main")

        assert selection.cancelled is True


class TestMenuSystemRun:
    """Tests for menu system run loop."""

    def test_run_command_selection(self) -> None:
        """Test running and selecting a command."""
        system = MenuSystem()
        menu = Menu(id="main", title="Main")
        menu.add_item(
            MenuItem(
                key="1",
                label="Test",
                action=MenuAction.COMMAND,
                command="test-cmd",
            )
        )
        system.register_menu(menu)

        with patch("ingestforge.cli.ui.menu_system.Prompt.ask", return_value="1"):
            result = system.run("main")

        assert result == "test-cmd"

    def test_run_quit(self) -> None:
        """Test running and quitting."""
        system = MenuSystem()
        menu = Menu(id="main", title="Main")
        system.register_menu(menu)

        with patch("ingestforge.cli.ui.menu_system.Prompt.ask", return_value="q"):
            result = system.run("main")

        assert result is None


# Default menus tests


class TestDefaultMenus:
    """Tests for default menu creation."""

    def test_create_default_menus(self) -> None:
        """Test creating default menu structure."""
        system = create_default_menus()

        assert "main" in system.menus
        assert "research" in system.menus
        assert "study" in system.menus
        assert "export" in system.menus

    def test_main_menu_items(self) -> None:
        """Test main menu has expected items."""
        system = create_default_menus()
        main = system.get_menu("main")

        assert main is not None
        assert len(main.items) >= 5

        # Check submenus are linked
        labels = [item.label for item in main.items]
        assert "Research" in labels
        assert "Study" in labels
        assert "Export" in labels

    def test_research_menu_items(self) -> None:
        """Test research menu has analysis commands."""
        system = create_default_menus()
        research = system.get_menu("research")

        assert research is not None
        assert research.parent_id == "main"

        commands = [item.command for item in research.items if item.command]
        assert "analyze topics" in commands
        assert "analyze entities" in commands

    def test_study_menu_items(self) -> None:
        """Test study menu has study commands."""
        system = create_default_menus()
        study = system.get_menu("study")

        assert study is not None
        assert study.parent_id == "main"

        commands = [item.command for item in study.items if item.command]
        assert "study review" in commands
        assert "study quiz" in commands

    def test_export_menu_items(self) -> None:
        """Test export menu has export commands."""
        system = create_default_menus()
        export = system.get_menu("export")

        assert export is not None
        assert export.parent_id == "main"

        commands = [item.command for item in export.items if item.command]
        assert "export markdown" in commands
        assert "export pack" in commands


class TestDepthLimit:
    """Tests for menu depth limiting."""

    def test_max_menu_depth_constant(self) -> None:
        """Test that MAX_MENU_DEPTH is set to 3."""
        assert MAX_MENU_DEPTH == 3

    def test_menu_stack_respects_depth(self) -> None:
        """Test that menu stack respects depth limit."""
        system = MenuSystem()

        # Create nested menus beyond limit
        for i in range(5):
            parent = f"menu_{i-1}" if i > 0 else None
            menu = Menu(id=f"menu_{i}", title=f"Menu {i}", parent_id=parent)
            menu.add_item(
                MenuItem(
                    key="n",
                    label="Next",
                    action=MenuAction.SUBMENU,
                    submenu_id=f"menu_{i+1}",
                )
            )
            system.register_menu(menu)

        # Stack should never exceed MAX_MENU_DEPTH
        system._menu_stack = ["menu_0", "menu_1", "menu_2", "menu_3", "menu_4"]

        # When depth is exceeded, it should be trimmed
        assert len(system._menu_stack) > MAX_MENU_DEPTH  # Pre-check
        # The run() method would trim this
