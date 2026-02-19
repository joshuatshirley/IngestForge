"""Hierarchical Menu System for IngestForge CLI.

Provides a nested menu structure for navigating CLI commands
with submenus for Research and Study functionality."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

MAX_MENU_DEPTH = 3
MAX_ITEMS_PER_MENU = 20
MAX_MENU_ITERATIONS = 1000  # Prevent infinite loops in menu system


class MenuAction(str, Enum):
    """Actions that can be taken from a menu."""

    COMMAND = "command"  # Execute a command
    SUBMENU = "submenu"  # Open a submenu
    BACK = "back"  # Go back to parent menu
    EXIT = "exit"  # Exit the menu system


@dataclass
class MenuItem:
    """A single item in a menu."""

    key: str  # Shortcut key (1, 2, a, b, etc.)
    label: str  # Display label
    action: MenuAction = MenuAction.COMMAND
    command: Optional[str] = None  # CLI command to execute
    submenu_id: Optional[str] = None  # ID of submenu to open
    description: str = ""  # Help text
    icon: str = ""  # Optional icon prefix

    def display_label(self) -> str:
        """Get label for display."""
        if self.icon:
            return f"{self.icon} {self.label}"
        return self.label


@dataclass
class Menu:
    """A menu containing items and optional submenus."""

    id: str
    title: str
    items: List[MenuItem] = field(default_factory=list)
    parent_id: Optional[str] = None
    description: str = ""

    def add_item(self, item: MenuItem) -> None:
        """Add item to menu (respects MAX_ITEMS_PER_MENU)."""
        if len(self.items) < MAX_ITEMS_PER_MENU:
            self.items.append(item)

    def get_item(self, key: str) -> Optional[MenuItem]:
        """Get item by key."""
        for item in self.items:
            if item.key.lower() == key.lower():
                return item
        return None


@dataclass
class MenuSelection:
    """Result of a menu selection."""

    action: MenuAction
    command: Optional[str] = None
    submenu_id: Optional[str] = None
    cancelled: bool = False


class MenuSystem:
    """Hierarchical menu system with keyboard navigation.

    Provides nested menus for organizing CLI commands with
    a maximum depth of 3 levels (Rule #1 compliance).
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialize menu system.

        Args:
            console: Rich console for output (default: new Console)
        """
        self.console = console or Console()
        self.menus: dict[str, Menu] = {}
        self._current_menu_id: str = "main"
        self._menu_stack: List[str] = []

    def register_menu(self, menu: Menu) -> None:
        """Register a menu in the system.

        Args:
            menu: Menu to register
        """
        self.menus[menu.id] = menu

    def get_menu(self, menu_id: str) -> Optional[Menu]:
        """Get menu by ID."""
        return self.menus.get(menu_id)

    def display_menu(self, menu_id: str) -> None:
        """Display a menu.

        Args:
            menu_id: ID of menu to display
        """
        menu = self.get_menu(menu_id)
        if not menu:
            self.console.print(f"[red]Menu not found: {menu_id}[/red]")
            return

        # Build table of options
        table = Table(
            title=menu.title,
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Key", style="yellow", width=6)
        table.add_column("Option", style="white")
        table.add_column("Description", style="dim")

        for item in menu.items:
            key_display = f"[{item.key}]"
            label = item.display_label()
            if item.action == MenuAction.SUBMENU:
                label += " →"
            table.add_row(key_display, label, item.description)

        # Add navigation options
        if menu.parent_id:
            table.add_row("[b]", "Back", "Return to previous menu")
        table.add_row("[q]", "Quit", "Exit menu")

        self.console.print()
        self.console.print(table)

    def prompt_selection(self, menu_id: str) -> MenuSelection:
        """Prompt user for menu selection.

        Args:
            menu_id: ID of current menu

        Returns:
            MenuSelection with user's choice
        """
        menu = self.get_menu(menu_id)
        if not menu:
            return MenuSelection(action=MenuAction.EXIT, cancelled=True)

        self.display_menu(menu_id)

        # Build valid choices
        valid_keys = [item.key.lower() for item in menu.items]
        valid_keys.extend(["q", "quit", "exit"])
        if menu.parent_id:
            valid_keys.extend(["b", "back"])

        choice = (
            Prompt.ask(
                "\n[cyan]Enter choice[/cyan]",
                default="",
            )
            .strip()
            .lower()
        )

        # Handle navigation
        if choice in ("q", "quit", "exit"):
            return MenuSelection(action=MenuAction.EXIT)

        if choice in ("b", "back") and menu.parent_id:
            return MenuSelection(
                action=MenuAction.BACK,
                submenu_id=menu.parent_id,
            )

        # Handle item selection
        item = menu.get_item(choice)
        if not item:
            self.console.print("[yellow]Invalid selection[/yellow]")
            return MenuSelection(action=MenuAction.COMMAND, cancelled=True)

        if item.action == MenuAction.SUBMENU:
            return MenuSelection(
                action=MenuAction.SUBMENU,
                submenu_id=item.submenu_id,
            )

        return MenuSelection(
            action=MenuAction.COMMAND,
            command=item.command,
        )

    def run(self, start_menu: str = "main") -> Optional[str]:
        """Run the menu system interactively.

        Args:
            start_menu: ID of starting menu

        Returns:
            Command to execute, or None if cancelled
        """
        self._current_menu_id = start_menu
        self._menu_stack = [start_menu]

        for _ in range(MAX_MENU_ITERATIONS):
            # Check depth limit (Rule #1)
            if len(self._menu_stack) > MAX_MENU_DEPTH:
                self._menu_stack.pop()
                continue

            selection = self.prompt_selection(self._current_menu_id)

            if selection.cancelled:
                continue

            if selection.action == MenuAction.EXIT:
                return None

            if selection.action == MenuAction.BACK:
                if self._menu_stack:
                    self._menu_stack.pop()
                if self._menu_stack:
                    self._current_menu_id = self._menu_stack[-1]
                else:
                    return None
                continue

            if selection.action == MenuAction.SUBMENU and selection.submenu_id:
                self._current_menu_id = selection.submenu_id
                self._menu_stack.append(selection.submenu_id)
                continue

            if selection.action == MenuAction.COMMAND:
                return selection.command
        raise AssertionError(f"Menu loop exceeded {MAX_MENU_ITERATIONS} iterations")


def create_default_menus() -> MenuSystem:
    """Create default IngestForge menu structure.

    Returns:
        Configured MenuSystem with all menus
    """
    system = MenuSystem()

    # Main menu
    main_menu = Menu(
        id="main",
        title="IngestForge Main Menu",
        description="Main navigation menu",
    )
    main_menu.add_item(
        MenuItem(
            key="1",
            label="Ingest Documents",
            action=MenuAction.COMMAND,
            command="ingest",
            description="Add documents to knowledge base",
        )
    )
    main_menu.add_item(
        MenuItem(
            key="2",
            label="Query Knowledge",
            action=MenuAction.COMMAND,
            command="query",
            description="Search your knowledge base",
        )
    )
    main_menu.add_item(
        MenuItem(
            key="3",
            label="Research",
            action=MenuAction.SUBMENU,
            submenu_id="research",
            description="Research and analysis tools",
        )
    )
    main_menu.add_item(
        MenuItem(
            key="4",
            label="Study",
            action=MenuAction.SUBMENU,
            submenu_id="study",
            description="Study and review tools",
        )
    )
    main_menu.add_item(
        MenuItem(
            key="5",
            label="Export",
            action=MenuAction.SUBMENU,
            submenu_id="export",
            description="Export and share content",
        )
    )
    main_menu.add_item(
        MenuItem(
            key="s",
            label="Status",
            action=MenuAction.COMMAND,
            command="status",
            description="Show project status",
        )
    )
    system.register_menu(main_menu)

    # Research submenu
    research_menu = Menu(
        id="research",
        title="Research Tools",
        parent_id="main",
        description="Tools for research and analysis",
    )
    research_menu.add_item(
        MenuItem(
            key="1",
            label="Analyze Topics",
            action=MenuAction.COMMAND,
            command="analyze topics",
            description="Extract topic clusters",
        )
    )
    research_menu.add_item(
        MenuItem(
            key="2",
            label="Analyze Entities",
            action=MenuAction.COMMAND,
            command="analyze entities",
            description="Extract named entities",
        )
    )
    research_menu.add_item(
        MenuItem(
            key="3",
            label="Analyze Relationships",
            action=MenuAction.COMMAND,
            command="analyze relationships",
            description="Find connections",
        )
    )
    research_menu.add_item(
        MenuItem(
            key="4",
            label="Find Duplicates",
            action=MenuAction.COMMAND,
            command="analyze duplicates",
            description="Detect duplicate content",
        )
    )
    research_menu.add_item(
        MenuItem(
            key="5",
            label="Similarity Search",
            action=MenuAction.COMMAND,
            command="analyze similarity",
            description="Find similar chunks",
        )
    )
    system.register_menu(research_menu)

    # Study submenu
    study_menu = Menu(
        id="study",
        title="Study Tools",
        parent_id="main",
        description="Tools for studying and reviewing",
    )
    study_menu.add_item(
        MenuItem(
            key="1",
            label="Review Cards",
            action=MenuAction.COMMAND,
            command="study review",
            description="Review due flashcards",
        )
    )
    study_menu.add_item(
        MenuItem(
            key="2",
            label="Take Quiz",
            action=MenuAction.COMMAND,
            command="study quiz",
            description="Test your knowledge",
        )
    )
    study_menu.add_item(
        MenuItem(
            key="3",
            label="View Glossary",
            action=MenuAction.COMMAND,
            command="study glossary",
            description="Browse terminology",
        )
    )
    study_menu.add_item(
        MenuItem(
            key="4",
            label="Study Progress",
            action=MenuAction.COMMAND,
            command="study progress",
            description="View statistics",
        )
    )
    system.register_menu(study_menu)

    # Export submenu
    export_menu = Menu(
        id="export",
        title="Export Tools",
        parent_id="main",
        description="Export and sharing options",
    )
    export_menu.add_item(
        MenuItem(
            key="1",
            label="Export Markdown",
            action=MenuAction.COMMAND,
            command="export markdown",
            description="Export to Markdown",
        )
    )
    export_menu.add_item(
        MenuItem(
            key="2",
            label="Export JSON",
            action=MenuAction.COMMAND,
            command="export json",
            description="Export to JSON",
        )
    )
    export_menu.add_item(
        MenuItem(
            key="3",
            label="Export Outline",
            action=MenuAction.COMMAND,
            command="export outline",
            description="Export with thesis structure",
        )
    )
    export_menu.add_item(
        MenuItem(
            key="4",
            label="Pack Corpus",
            action=MenuAction.COMMAND,
            command="export pack",
            description="Create portable package",
        )
    )
    export_menu.add_item(
        MenuItem(
            key="5",
            label="Unpack Corpus",
            action=MenuAction.COMMAND,
            command="export unpack",
            description="Restore from package",
        )
    )
    system.register_menu(export_menu)

    return system


def show_interactive_menu() -> Optional[str]:
    """Show interactive menu and return selected command.

    Returns:
        Selected command string, or None if cancelled
    """
    system = create_default_menus()
    return system.run()
