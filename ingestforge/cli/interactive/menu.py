"""Interactive Menu - 6-Pillar navigation for IngestForge.

Provides a user-friendly menu interface organized into 6 workflow pillars.
Any feature is accessible within 2 keystrokes maximum."""
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import questionary
import typer
from questionary import Style
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

_MAX_MENU_ITERATIONS: int = 1000
_MAX_MAIN_MENU_ITERATIONS: int = 10000
MenuItem = Tuple[str, str, str, str]  # (key, label, description, action)
MenuList = List[MenuItem]
HandlerDict = Dict[str, Callable[[], None]]
CommandDict = Dict[str, str]
QuickBarItem = Tuple[str, str, str]  # (key, label, command)

# =============================================================================
# QUICK BAR - Single keystroke access to common operations (Rule #6: Constants)
# =============================================================================

QUICK_BAR: List[QuickBarItem] = [
    ("q", "Query", "query"),
    ("i", "Ingest", "ingest"),
    ("a", "Agent", "agent"),
    ("s", "Status", "status"),
    ("f", "Study", "flashcards"),
    ("e", "Export", "export"),
    ("g", "GUI", "gui"),
    ("?", "Help", "help"),
]

# Quick bar keys for fast lookup (Rule #6: Derived constant)
QUICK_BAR_KEYS: set = {item[0] for item in QUICK_BAR}
console: Console = Console()

# Style for questionary menus (Rule #6: acceptable constant)
MENU_STYLE: Style = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:cyan bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
    ]
)

# =============================================================================
# MENU DEFINITIONS - 4 Pillar Structure + System (Phase 3 Redesign)
# =============================================================================

MAIN_MENU: MenuList = [
    (
        "1",
        "Ingest & Organize",
        "Documents, Libraries, Tags, Transform",
        "ingest_organize",
    ),
    (
        "2",
        "Search & Discover",
        "Query, Academic, Legal, Security, Agent",
        "search_discover",
    ),
    (
        "3",
        "Analyze & Understand",
        "Comprehension, Arguments, Literary, Code",
        "analyze",
    ),
    (
        "4",
        "Create & Export",
        "Study Tools, Writing, Citations, Export",
        "create_export",
    ),
    ("0", "System", "Config, LLM, Storage, Maintenance, API", "system"),
    ("s", "Status", "View current project state", "status"),
    ("q", "Quit", "Exit IngestForge", "quit"),
]

# [1] INGEST & ORGANIZE - Data operations
INGEST_ORGANIZE_MENU: MenuList = [
    ("1", "Add Documents", "Ingest files / YouTube / Audio", "add_docs"),
    ("2", "Manage Documents", "List / Delete / Re-ingest / Clear", "docs"),
    ("3", "Libraries", "List / Query / Move / Delete", "libraries"),
    ("4", "Organize", "Tags / Bookmarks / Annotations", "organize"),
    ("5", "Transform", "Clean / Enrich / Filter / Merge / Split", "transform"),
    ("6", "Index Management", "Info / List / Rebuild / Delete", "index"),
    ("b", "Back", "Return to main menu", "back"),
]

# Add Documents submenu
ADD_DOCS_SUBMENU: MenuList = [
    ("1", "File/Folder", "Process local files", "ingest"),
    ("2", "YouTube Video", "Download and transcribe", "youtube"),
    ("3", "Audio File", "Transcribe with Whisper", "audio"),
    ("4", "Batch Import", "Process multiple files", "batch"),
    ("b", "Back", "Return to previous menu", "back"),
]

# [2] SEARCH & DISCOVER - Query and research
SEARCH_DISCOVER_MENU: MenuList = [
    ("1", "Query Knowledge Base", "Hybrid search with AI answer", "query"),
    ("2", "Interactive Shell", "REPL mode with history", "shell"),
    ("3", "Agent Research", "ReAct reasoning with tools", "agent"),
    ("4", "Academic Search", "Semantic Scholar / CrossRef / arXiv", "academic"),
    ("5", "Legal Research", "Court opinions / Cases", "legal"),
    ("6", "Security Research", "CVE search / Details", "security"),
    ("7", "Research Audit", "Audit KB / Verify citations", "audit"),
    ("b", "Back", "Return to main menu", "back"),
]

# [3] ANALYZE & UNDERSTAND - Analysis tools
ANALYZE_MENU: MenuList = [
    ("1", "Comprehension", "Explain / Compare / Connect concepts", "comprehension"),
    ("2", "Argument Analysis", "Conflicts / Counter / Debate / Gaps", "argument"),
    ("3", "Literary Analysis", "Arc / Character / Symbols / Themes", "literary"),
    (
        "4",
        "Content Analysis",
        "Topics / Entities / Relationships / Timeline",
        "content",
    ),
    ("5", "Code Analysis", "Analyze / Document / Explain / Map", "code"),
    ("6", "Fact Checker", "Contradiction detection / Evidence linking", "factcheck"),
    ("7", "Stored Analyses", "View / Search / Manage stored analyses", "stored"),
    ("b", "Back", "Return to main menu", "back"),
]

# [4] CREATE & EXPORT - Writing, study, and export
CREATE_EXPORT_MENU: MenuList = [
    ("1", "Study Tools", "Flashcards / Quiz / Notes / Review", "study"),
    (
        "2",
        "Academic Writing",
        "Draft / Outline / Paraphrase / Quote",
        "academic_writing",
    ),
    (
        "3",
        "Citation Tools",
        "Bibliography / Extract / Format / Validate",
        "citation_tools",
    ),
    ("4", "Export", "Markdown / JSON / PDF / Package", "export"),
    ("5", "Research Summary", "Multi-agent summarization", "research_summary"),
    ("b", "Back", "Return to main menu", "back"),
]

# [0] SYSTEM - Administration
SYSTEM_MENU: MenuList = [
    ("1", "Quick Start", "Wizard: init -> ingest -> query", "quick_start"),
    ("2", "LLM Settings", "Model selection", "llm"),
    ("3", "Configuration", "Show / List / Set / Reset / Validate", "config"),
    ("4", "Storage", "Health / Migrate / Stats", "storage"),
    ("5", "Maintenance", "Backup / Restore / Cleanup / Optimize", "maintenance"),
    ("6", "Monitor", "Diagnostics / Health / Logs / Metrics", "monitor"),
    ("7", "Workflow", "Batch / Pipeline / Schedule", "workflow"),
    ("8", "API Server", "Start / Stop REST API server", "api"),
    ("b", "Back", "Return to main menu", "back"),
]

# Legacy aliases for backward compatibility (will be removed in future)
CORE_MENU: MenuList = SEARCH_DISCOVER_MENU
DATA_MENU: MenuList = INGEST_ORGANIZE_MENU
DISCOVERY_MENU: MenuList = SEARCH_DISCOVER_MENU
ANALYSIS_MENU: MenuList = ANALYZE_MENU
WRITING_MENU: MenuList = CREATE_EXPORT_MENU
ADMIN_MENU: MenuList = SYSTEM_MENU

DOCS_SUBMENU: MenuList = [
    ("1", "List Documents", "View all ingested documents", "list"),
    ("2", "Delete Document", "Remove a document", "delete"),
    ("3", "Re-ingest", "Delete and re-process", "reingest"),
    ("4", "Clear All", "Remove ALL documents", "clear"),
    ("b", "Back", "Return to previous menu", "back"),
]

LIBRARY_SUBMENU: MenuList = [
    ("1", "List Libraries", "View all libraries", "list"),
    ("2", "Query Library", "Search within library", "query"),
    ("3", "Move Document", "Move doc to library", "move"),
    ("4", "Delete Library", "Remove all docs from library", "delete"),
    ("b", "Back", "Return to previous menu", "back"),
]

TRANSFORM_SUBMENU: MenuList = [
    ("1", "Clean Text", "Clean and normalize", "clean"),
    ("2", "Enrich", "Add metadata", "enrich"),
    ("3", "Filter Chunks", "Filter by criteria", "filter"),
    ("4", "Merge", "Merge/deduplicate", "merge"),
    ("5", "Split", "Split documents", "split"),
    ("b", "Back", "Return to previous menu", "back"),
]

EXPORT_SUBMENU: MenuList = [
    ("1", "Markdown", "Export to .md", "markdown"),
    ("2", "JSON", "Export to .json", "json"),
    ("3", "PDF", "Export to .pdf", "pdf"),
    ("4", "Outline", "Structured outline document", "outline"),
    ("5", "Context", "Export RAG context", "context"),
    ("6", "Study Package", "Full study folder", "folder"),
    ("7", "Knowledge Graph", "Interactive D3.js visualization", "graph"),
    ("8", "Create Package", "Portable corpus pack", "pack"),
    ("9", "Import Package", "Unpack corpus", "unpack"),
    ("0", "Package Info", "Show pack metadata", "info"),
    ("b", "Back", "Return to previous menu", "back"),
]

# Organization submenu - tags, bookmarks, annotations (Phase 2)
ORGANIZE_SUBMENU: MenuList = [
    ("1", "Add Tag", "Tag chunks for organization", "tag"),
    ("2", "List Tags", "View all tags", "tags"),
    ("3", "Bookmark", "Save chunk as bookmark", "bookmark"),
    ("4", "Bookmarks", "View saved bookmarks", "bookmarks"),
    ("5", "Annotate", "Add note to chunk", "annotate"),
    ("6", "Annotations", "Manage annotations", "annotations"),
    ("7", "Mark Read", "Mark chunks as read/unread", "mark"),
    ("b", "Back", "Return to previous menu", "back"),
]

# Legal research submenu (Phase 2)
LEGAL_SUBMENU: MenuList = [
    ("1", "Search Opinions", "Search CourtListener", "court"),
    ("2", "Case Details", "Get case information", "detail"),
    ("3", "Download Opinion", "Download case text", "download"),
    ("4", "Jurisdictions", "List court codes", "list"),
    ("b", "Back", "Return to previous menu", "back"),
]

# Security research submenu (Phase 2)
SECURITY_SUBMENU: MenuList = [
    ("1", "Search CVE", "Search vulnerabilities", "cve"),
    ("2", "CVE Details", "Get CVE by ID", "cve-get"),
    ("b", "Back", "Return to previous menu", "back"),
]

# Citation tools submenu (Phase 2)
CITATION_SUBMENU: MenuList = [
    ("1", "Extract", "Extract citations", "extract"),
    ("2", "Format", "Format citations", "format"),
    ("3", "Bibliography", "Generate bibliography", "bibliography"),
    ("4", "Validate", "Check citations", "validate"),
    ("5", "Citation Graph", "Visualize citations", "graph"),
    ("6", "Check in Doc", "Verify doc citations", "cite-check"),
    ("7", "Format in Doc", "Format doc citations", "cite-format"),
    ("8", "Insert", "Insert citations", "cite-insert"),
    ("b", "Back", "Return to previous menu", "back"),
]

# Banner row definitions (Rule #4: Extracted data for smaller functions)
# Anvil design with INGESTFORGE ASCII art and fireball
_BANNER_ROWS: List[List[Tuple[str, str]]] = [
    [("        ", ""), ("__", "color(94)")],
    [
        ("        ", ""),
        ("||", "color(130)"),
        (" _   _   ____  _____  ____  _____  _____   ", "color(51)"),
        ("(    ", "color(196)"),
        ("____   ____  _____", "color(51)"),
    ],
    [
        ("      ", ""),
        ("__", "color(250)"),
        ("||", "color(130)"),
        ("| \\ | | / ___|| ____|/ ___||_   _||  ___| ", "color(45)"),
        (") \\  ", "color(196)"),
        ("|  _ \\ / ___|| ____|", "color(45)"),
    ],
    [
        ("      ", ""),
        ("|  |", "color(252)"),
        ("|  \\| || |  _ |  _|  \\___ \\  | |  | |_   ", "color(39)"),
        ("/ ", "color(196)"),
        (",", "color(214)"),
        (" ) ", "color(196)"),
        ("| |_) | |  _ |  _|", "color(39)"),
    ],
    [
        ("      ", ""),
        ("|  |", "color(255)"),
        ("| |\\  || |_| || |___  ___) | | |  |  _| ", "color(33)"),
        ("| ", "color(196)"),
        ("(", "color(208)"),
        ("_", "color(226)"),
        (")", "color(208)"),
        (" |", "color(196)"),
        ("|  _ <| |_| || |___", "color(33)"),
    ],
    [
        ("      ", ""),
        ("\\  |", "color(252)"),
        ("|_| \\_| \\____||_____||____/  |_|  |_|    ", "color(27)"),
        ("\\___/ ", "color(196)"),
        ("|_| \\_\\\\____||_____|", "color(27)"),
    ],
    [
        ("       ", ""),
        ("\\ |", "color(250)"),
        ("                         ", ""),
        ("Document Chunking ", "color(250)"),
        ("&", "color(208)"),
        (" Vector Smithing", "color(250)"),
    ],
    [("        ", ""), ("\\|", "color(247)")],
]

# =============================================================================
# BANNER AND DISPLAY (Rule #4: Functions <60 lines)
# =============================================================================


def clear_screen() -> None:
    """Clear console screen cross-platform."""
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")
    try:
        console.clear()
    except Exception as e:
        from ingestforge.core.logging import get_logger

        get_logger(__name__).debug(f"Console clear failed: {e}")


def display_banner() -> None:
    """Display IngestForge banner with anvil and fireball design."""
    console.print()
    for row_data in _BANNER_ROWS:
        row = Text()
        for text, style in row_data:
            row.append(text, style=style)
        console.print(row)

    console.print()


def display_dashboard() -> None:
    """Display storage status dashboard panel."""
    from ingestforge.cli.interactive.storage_cache import StorageCache

    cache = StorageCache()
    status: str
    stats_line: str
    if cache.is_ready():
        status, stats_line = _format_ready_status(cache)
    elif cache.is_loading():
        status = "[yellow]Loading...[/yellow]"
        stats_line = "[dim]Initializing storage...[/dim]"
    elif cache.get_error():
        error_msg: str = cache.get_error() or ""
        status = f"[red]{error_msg[:40]}[/red]"
        stats_line = "[dim]Storage unavailable[/dim]"
    else:
        status = "[dim]Not initialized[/dim]"
        stats_line = "[dim]Starting...[/dim]"

    content: str = f"Storage: {status}\n{stats_line}"
    panel = Panel(
        content, title="[bold]Dashboard[/bold]", border_style="dim", padding=(0, 1)
    )
    console.print(panel)


def display_quick_bar() -> None:
    """Display Quick Bar with single-key shortcuts.

    Rule #4: Function <60 lines
    Rule #9: Full type hints

    The Quick Bar provides instant access to common operations
    without navigating through menus.
    """
    # Build compact display string
    items: List[str] = []
    for key, label, _ in QUICK_BAR:
        items.append(f"[cyan][{key.upper()}][/cyan]{label}")

    # Join with separator and wrap in panel
    bar_text: str = " ".join(items)

    panel = Panel(
        bar_text,
        title="[yellow]Quick Access[/yellow]",
        border_style="yellow",
        padding=(0, 1),
    )
    console.print(panel)


def _format_ready_status(cache: Any) -> Tuple[str, str]:
    """Format status when storage is ready. Rule #4: Extracted helper."""
    stats: Dict[str, int] = cache.get_cached_stats()
    status: str = "[green]Ready[/green]"

    if not cache.are_stats_ready():
        return status, "[dim]Scanning storage...[/dim]"

    doc_str: str = str(stats["docs"]) if stats["docs"] >= 0 else "?"
    stats_line: str = (
        f"Docs: [cyan]{doc_str}[/cyan] | "
        f"Libraries: [cyan]{stats['libraries']:,}[/cyan] | "
        f"Chunks: [cyan]{stats['chunks']:,}[/cyan]"
    )
    return status, stats_line


def display_menu(title: str, items: MenuList) -> str:
    """Display menu with arrow key navigation using questionary.

    Rule #7: Validate parameters
    Rule #9: Full type hints
    """
    assert title, "Menu title cannot be empty"
    assert items, "Menu items cannot be empty"

    console.print()
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print()

    choices: List[str] = []
    key_map: Dict[str, str] = {}

    for key, label, description, _ in items:
        display: str = f"{label} - {description}"
        choices.append(display)
        key_map[display] = key

    try:
        result: Optional[str] = questionary.select(
            "Select option:",
            choices=choices,
            style=MENU_STYLE,
        ).ask()

        if result is None:
            return "b"
        return key_map.get(result, "b")
    except (KeyboardInterrupt, EOFError):
        return "b"


# =============================================================================
# UTILITY FUNCTIONS (Rule #4: Small, focused helpers)
# =============================================================================


def _open_file_browser(
    title: str = "Select File",
    file_types: Optional[List[Tuple[str, str]]] = None,
    initial_dir: Optional[str] = None,
) -> Optional[str]:
    """Open a file browser dialog to select a file.

    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        title: Dialog title
        file_types: List of (description, pattern) tuples e.g. [("PDF", "*.pdf")]
        initial_dir: Starting directory

    Returns:
        Selected file path or None if cancelled
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        # Create hidden root window
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        # Default file types
        if file_types is None:
            file_types = [
                ("All supported", "*.pdf *.epub *.txt *.docx *.md *.mp3 *.wav *.m4a"),
                ("PDF files", "*.pdf"),
                ("EPUB files", "*.epub"),
                ("Text files", "*.txt *.md"),
                ("Word documents", "*.docx"),
                ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
                ("All files", "*.*"),
            ]

        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=file_types,
            initialdir=initial_dir or str(Path.home()),
        )
        root.destroy()

        return file_path if file_path else None

    except ImportError:
        console.print("[yellow]File browser not available (tkinter required)[/yellow]")
        return None
    except Exception as e:
        console.print(f"[yellow]File browser error: {e}[/yellow]")
        return None


def _open_folder_browser(
    title: str = "Select Folder",
    initial_dir: Optional[str] = None,
) -> Optional[str]:
    """Open a folder browser dialog to select a directory.

    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        title: Dialog title
        initial_dir: Starting directory

    Returns:
        Selected folder path or None if cancelled
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        # Create hidden root window
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        folder_path = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir or str(Path.home()),
        )
        root.destroy()

        return folder_path if folder_path else None

    except ImportError:
        console.print(
            "[yellow]Folder browser not available (tkinter required)[/yellow]"
        )
        return None
    except Exception as e:
        console.print(f"[yellow]Folder browser error: {e}[/yellow]")
        return None


def _browse_for_path(for_folder: bool = False) -> Optional[str]:
    """Prompt user to browse for file or folder.

    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        for_folder: True to browse for folder, False for file

    Returns:
        Selected path or None
    """
    if for_folder:
        return _open_folder_browser("Select Folder to Ingest")
    return _open_file_browser("Select File to Ingest")


def run_command(cmd: str) -> int:
    """Run an IngestForge command (simple commands).

    Rule #7: Validate cmd parameter
    """
    assert cmd, "Command cannot be empty"

    import subprocess

    console.print(f"\n[dim]Running: ingestforge {cmd}[/dim]\n")
    result = subprocess.run(
        ["python", "-m", "ingestforge.cli.main"] + cmd.split(),
        cwd=Path.cwd(),
    )
    return result.returncode


def run_command_with_args(*args: str) -> int:
    """Run an IngestForge command with proper argument handling.

    Rule #7: Validate args
    """
    assert args, "At least one argument required"

    import subprocess

    cmd_display: str = " ".join(args)
    console.print(f"\n[dim]Running: ingestforge {cmd_display}[/dim]\n")
    result = subprocess.run(
        ["python", "-m", "ingestforge.cli.main"] + list(args),
        cwd=Path.cwd(),
    )
    return result.returncode


def _validate_path(path: str) -> Optional[Path]:
    """Validate that a path exists.

    Rule #7: Parameter validation with user feedback
    """
    if not path or not path.strip():
        console.print("[red]Error: Path cannot be empty[/red]")
        return None

    resolved: Path = Path(path.strip()).resolve()
    if not resolved.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        return None
    return resolved


def _validate_required(value: str, field_name: str) -> bool:
    """Validate that a required field is not empty.

    Rule #7: Parameter validation
    """
    assert field_name, "Field name required for error message"

    if not value or not value.strip():
        console.print(f"[red]Error: {field_name} cannot be empty[/red]")
        return False
    return True


def _prompt_continue(back_label: str = "main") -> str:
    """Prompt for next action after command."""
    console.print("\n" + "-" * 60)
    nav_text: str = f"[b]ack to {back_label} | [m]ore | [q]uit"
    console.print(f"[cyan]{escape(nav_text)}[/cyan]")
    try:
        choice: str = Prompt.ask("Select", default="m")
        return choice.lower().strip()
    except (EOFError, KeyboardInterrupt):
        return "b"


def _get_storage() -> Optional["StorageBackend"]:
    """Get storage backend from cache.

    Rule #9: Return typed StorageBackend protocol
    """
    from ingestforge.cli.interactive.storage_cache import StorageCache

    cache = StorageCache()

    if not cache.is_ready() and not cache.is_loading():
        cache.start_loading()

    storage: Any = cache.get_storage(timeout=30.0)
    if storage is None:
        error: str = cache.get_error() or "Unknown error"
        console.print(f"[red]Storage error: {error}[/red]")
    return storage


def _get_document_list() -> List[Tuple[str, str, int]]:
    """Get list of documents with chunk counts.

    Rule #2: Bounded iteration over chunks
    """
    storage = _get_storage()
    if not storage:
        return []

    chunks = storage.get_all_chunks()
    doc_map: Dict[str, Tuple[str, int]] = {}
    iteration_count: int = 0
    max_chunks: int = 100_000

    for chunk in chunks:
        iteration_count += 1
        if iteration_count > max_chunks:
            break

        doc_id: str = chunk.document_id
        if doc_id not in doc_map:
            doc_map[doc_id] = (chunk.source_file, 0)
        source, count = doc_map[doc_id]
        doc_map[doc_id] = (source, count + 1)

    return [(doc_id, source, count) for doc_id, (source, count) in doc_map.items()]


def _parse_int_choice(choice: str, max_val: int) -> Optional[int]:
    """Parse user choice to integer index.

    Rule #4: Extracted validation helper
    Rule #7: Validate input
    """
    assert max_val > 0, "Max value must be positive"

    try:
        idx: int = int(choice) - 1
        if 0 <= idx < max_val:
            return idx
        console.print("[red]Invalid selection: out of range[/red]")
        return None
    except ValueError:
        console.print("[red]Invalid input: Please enter a number[/red]")
        return None


# =============================================================================
# [1] CORE INTERACTIONS HANDLERS
# =============================================================================


def handle_quick_start() -> None:
    """Handle quick start workflow."""
    console.print("\n[bold cyan]Quick Start Wizard[/bold cyan]\n")

    project_dir: Path = Path.cwd() / ".ingestforge"
    if project_dir.exists():
        console.print("[green]OK[/green] Project already initialized")
    else:
        name: str = Prompt.ask("Project name", default="my_project")
        if not _validate_required(name, "Project name"):
            input("Press Enter to continue...")
            return
        run_command(f"init {name}")

    docs_path: str = Prompt.ask("\nPath to documents (or Enter to skip)", default="")
    if docs_path:
        path: Optional[Path] = _validate_path(docs_path)
        if path is None:
            input("Press Enter to continue...")
            return
        # Only add --recursive for directories
        if path.is_dir():
            run_command_with_args("ingest", str(path), "--recursive")
        else:
            run_command_with_args("ingest", str(path))

    console.print("\n[green]Quick start complete![/green]")
    input("Press Enter to continue...")


def handle_shell() -> None:
    """Start interactive shell."""
    import subprocess

    console.print("\n[cyan]Starting interactive shell...[/cyan]\n")
    subprocess.run(
        [sys.executable, "-m", "ingestforge.cli.main", "interactive", "shell"],
        cwd=Path.cwd(),
    )


def handle_query() -> None:
    """Handle query interface."""
    console.print("\n[bold cyan]Query Knowledge Base[/bold cyan]\n")
    query: str = Prompt.ask("Enter your question")
    query = query.strip().strip('"').strip("'")
    if not _validate_required(query, "Question"):
        input("\nPress Enter to continue...")
        return
    run_command_with_args("query", query)
    input("\nPress Enter to continue...")


def handle_agent() -> None:
    """Handle ReAct agent mode."""
    console.print("\n[bold cyan]Agent Mode (ReAct Reasoning)[/bold cyan]\n")
    console.print("The agent will autonomously research a topic using tools.\n")
    topic: str = Prompt.ask("Research topic or question")
    if not _validate_required(topic, "Topic"):
        input("\nPress Enter to continue...")
        return
    max_steps: str = Prompt.ask("Max reasoning steps", default="10")
    run_command_with_args("agent", "run", topic, "--max-steps", max_steps)
    input("\nPress Enter to continue...")


def handle_core_menu() -> None:
    """Handle [1] Core Interactions submenu."""
    handlers: HandlerDict = {
        "1": handle_quick_start,
        "2": handle_shell,
        "3": handle_query,
        "4": handle_agent,
    }
    _run_submenu("Core Interactions", CORE_MENU, handlers)


# =============================================================================
# [2] DATA & WORKSPACE HANDLERS
# =============================================================================


def handle_ingest() -> None:
    """Handle document ingestion with file browser support."""
    console.print("\n[bold cyan]Ingest Documents[/bold cyan]\n")

    # Offer browse or manual entry
    browse_choice: str = (
        questionary.select(
            "How would you like to select files?",
            choices=[
                "Browse for file...",
                "Browse for folder...",
                "Type path manually",
            ],
            style=MENU_STYLE,
        ).ask()
        or "Type path manually"
    )

    path_input: Optional[str] = None
    is_folder_selection: bool = False

    if browse_choice == "Browse for file...":
        path_input = _browse_for_path(for_folder=False)
        if path_input:
            console.print(f"[green]Selected: {path_input}[/green]")
    elif browse_choice == "Browse for folder...":
        path_input = _browse_for_path(for_folder=True)
        is_folder_selection = True
        if path_input:
            console.print(f"[green]Selected: {path_input}[/green]")
    else:
        path_input = Prompt.ask("Path to documents or file")

    if not path_input:
        console.print("[yellow]No path selected[/yellow]")
        input("\nPress Enter to continue...")
        return

    path: Optional[Path] = _validate_path(path_input)
    if path is None:
        input("\nPress Enter to continue...")
        return

    # Only ask for recursive if path is a directory
    cmd: List[str] = ["ingest", str(path)]
    if path.is_dir():
        recursive: bool = Prompt.ask("Recursive? (y/n)", default="y").lower() == "y"
        if recursive:
            cmd.append("--recursive")
    # Single file - no recursive option needed

    run_command_with_args(*cmd)
    input("\nPress Enter to continue...")


def _should_exit_submenu(menu_name: str) -> bool:
    """Check if user wants to exit submenu. Rule #1: Extracted to reduce nesting."""
    cont: str = _prompt_continue(menu_name)
    if cont == "q":
        raise SystemExit(0)
    return cont == "b"


def _execute_docs_action(choice: str) -> None:
    """Execute document menu action. Rule #1: Max 3 nesting levels."""
    actions: Dict[str, Callable[[], None]] = {
        "1": _show_document_list,
        "2": _delete_document,
        "3": _reingest_document,
        "4": _clear_all_documents,
    }
    action = actions.get(choice)
    if action:
        action()


def _execute_library_action(choice: str) -> None:
    """Execute library menu action. Rule #1: Max 3 nesting levels."""
    actions: Dict[str, Callable[[], None]] = {
        "1": _list_libraries,
        "2": _query_library,
        "3": _move_to_library,
        "4": _delete_library,
    }
    action = actions.get(choice)
    if action:
        action()


def _execute_discovery_action(choice: str) -> None:
    """Execute discovery menu action. Rule #1: Max 3 nesting levels."""
    action_map: Dict[str, Callable[[], None]] = {
        "1": handle_academic_search,
        "2": handle_arxiv,
        "5": handle_legal_research,
        "6": handle_security_research,
    }

    if choice in action_map:
        action_map[choice]()
        return

    if choice == "3":
        _run_simple_commands(
            "Citation Graphs",
            {
                "1": ("Paper Citations", "discovery scholar-citations", "Paper ID"),
                "2": ("Paper References", "discovery scholar-references", "Paper ID"),
            },
        )
    elif choice == "4":
        _run_simple_commands(
            "Researchers",
            {
                "1": ("Find Scholars", "discovery scholars", "Field/Topic"),
                "2": ("Timeline", "discovery timeline", "Topic"),
            },
        )
    elif choice == "7":
        _run_simple_commands(
            "Research Audit",
            {
                "1": ("Audit KB", "research audit", None),
                "2": ("Verify Citations", "research verify", None),
            },
        )


def _execute_prompted_analysis_action(choice: str) -> None:
    """Execute analysis action requiring prompt. Rule #1: Max 3 nesting levels."""
    if choice == "2":
        query: str = Prompt.ask("Search query")
        if not query:
            return
        run_command_with_args("analysis", "search", query)
        return

    if choice == "3":
        analysis_id: str = Prompt.ask("Analysis ID")
        if not analysis_id:
            return
        run_command_with_args("analysis", "show", analysis_id)
        return

    if choice == "4":
        analysis_id = Prompt.ask("Analysis ID to delete")
        if not analysis_id:
            return
        run_command_with_args("analysis", "delete", analysis_id)
        return

    if choice == "5":
        doc_path: str = Prompt.ask("Document path or name")
        if not doc_path:
            return
        run_command_with_args("analysis", "for-document", doc_path)


def handle_docs_submenu() -> None:
    """Handle document management. Rule #1: Max 3 nesting levels."""
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu("Manage Documents", DOCS_SUBMENU)
        if choice == "b":
            break

        # Execute action
        _execute_docs_action(choice)

        # Handle continuation
        if _should_exit_submenu("docs menu"):
            break


def _show_document_list() -> None:
    """List all documents."""
    console.print("\n[bold cyan]Documents[/bold cyan]\n")
    docs: List[Tuple[str, str, int]] = _get_document_list()
    if not docs:
        console.print("[yellow]No documents in storage[/yellow]")
        return

    table = Table(title=f"{len(docs)} Documents")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Source", style="white")
    table.add_column("Chunks", style="green", justify="right")

    for i, (_, source, count) in enumerate(docs, 1):
        disp: str = source if len(source) < 50 else "..." + source[-47:]
        table.add_row(str(i), disp, str(count))
    console.print(table)


def _delete_document() -> None:
    """Delete a document."""
    console.print("\n[bold cyan]Delete Document[/bold cyan]\n")
    docs: List[Tuple[str, str, int]] = _get_document_list()
    if not docs:
        console.print("[yellow]No documents[/yellow]")
        return

    for i, (_, source, count) in enumerate(docs, 1):
        console.print(f"  [{i}] {Path(source).name} ({count} chunks)")

    choice: str = Prompt.ask("\nDocument # to delete (Enter to cancel)", default="")
    if not choice:
        return

    idx: Optional[int] = _parse_int_choice(choice, len(docs))
    if idx is None:
        return

    doc_id, source, _ = docs[idx]
    if Prompt.ask(f"Delete {Path(source).name}? (y/n)", default="n") == "y":
        storage = _get_storage()
        if storage:
            deleted: int = storage.delete_document(doc_id)
            console.print(f"[green]Deleted {deleted} chunks[/green]")


def _reingest_document() -> None:
    """Re-ingest a document."""
    console.print("\n[bold cyan]Re-ingest Document[/bold cyan]\n")
    docs: List[Tuple[str, str, int]] = _get_document_list()
    if not docs:
        console.print("[yellow]No documents[/yellow]")
        return

    for i, (_, source, count) in enumerate(docs, 1):
        console.print(f"  [{i}] {Path(source).name} ({count} chunks)")

    choice: str = Prompt.ask("\nDocument # to re-ingest (Enter to cancel)", default="")
    if not choice:
        return

    idx: Optional[int] = _parse_int_choice(choice, len(docs))
    if idx is None:
        return

    doc_id, source, _ = docs[idx]
    if not Path(source).exists():
        console.print(f"[red]File not found: {source}[/red]")
        return

    storage = _get_storage()
    if storage:
        storage.delete_document(doc_id)
    run_command_with_args("ingest", source)


def _clear_all_documents() -> None:
    """Clear all documents."""
    console.print("\n[bold cyan]Clear All Documents[/bold cyan]\n")
    storage = _get_storage()
    if not storage:
        return

    total: int = storage.count()
    if total == 0:
        console.print("[yellow]Storage is empty[/yellow]")
        return

    console.print(f"[yellow]WARNING: Delete ALL {total} chunks?[/yellow]")
    if Prompt.ask("Type 'DELETE' to confirm", default="") == "DELETE":
        storage.clear()
        console.print("[green]Cleared[/green]")


def handle_libraries_submenu() -> None:
    """Handle library management. Rule #1: Max 3 nesting levels."""
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu("Libraries", LIBRARY_SUBMENU)
        if choice == "b":
            break

        # Execute action
        _execute_library_action(choice)

        # Handle continuation - reuses _should_exit_submenu helper
        if _should_exit_submenu("libraries"):
            break


def _list_libraries() -> None:
    """List libraries."""
    console.print("\n[bold cyan]Libraries[/bold cyan]\n")
    storage = _get_storage()
    if not storage:
        return

    libs: List[str] = storage.get_libraries()
    table = Table()
    table.add_column("Library", style="cyan")
    table.add_column("Chunks", style="green", justify="right")

    total: int = 0
    for lib in libs:
        count: int = storage.count_by_library(lib)
        total += count
        table.add_row(lib, str(count))

    table.add_row("-" * 15, "-" * 6)
    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
    console.print(table)


def _query_library() -> None:
    """Query within a library."""
    storage = _get_storage()
    if not storage:
        return

    libs: List[str] = storage.get_libraries()
    for i, lib in enumerate(libs, 1):
        console.print(f"  [{i}] {lib}")

    choice: str = Prompt.ask("Library #", default="")
    if not choice:
        return

    idx: Optional[int] = _parse_int_choice(choice, len(libs))
    if idx is None:
        return

    query: str = Prompt.ask("Question")
    if query:
        run_command_with_args("query", query, "--library", libs[idx])


def _move_to_library() -> None:
    """Move document to library."""
    storage = _get_storage()
    if not storage:
        return

    docs: List[Tuple[str, str, int]] = _get_document_list()
    if not docs:
        console.print("[yellow]No documents[/yellow]")
        return

    for i, (_, source, _) in enumerate(docs, 1):
        console.print(f"  [{i}] {Path(source).name}")

    doc_choice: str = Prompt.ask("Document #", default="")
    if not doc_choice:
        return

    idx: Optional[int] = _parse_int_choice(doc_choice, len(docs))
    if idx is None:
        return

    doc_id: str = docs[idx][0]
    lib_name: str = Prompt.ask("Target library name")
    if lib_name:
        moved: int = storage.move_document_to_library(doc_id, lib_name)
        console.print(f"[green]Moved {moved} chunks[/green]")


def _delete_library() -> None:
    """Delete a library."""
    storage = _get_storage()
    if not storage:
        return

    libs: List[str] = storage.get_libraries()
    for i, lib in enumerate(libs, 1):
        count: int = storage.count_by_library(lib)
        console.print(f"  [{i}] {lib} ({count} chunks)")

    choice: str = Prompt.ask("Library #", default="")
    if not choice:
        return

    idx: Optional[int] = _parse_int_choice(choice, len(libs))
    if idx is None:
        return

    lib: str = libs[idx]
    if Prompt.ask(f"Delete all from '{lib}'? (y/n)", default="n") == "y":
        deleted: int = storage.delete_by_library(lib)
        console.print(f"[green]Deleted {deleted} chunks[/green]")


def handle_transform_submenu() -> None:
    """Handle transform operations with enhanced clean options."""
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu("Transform", TRANSFORM_SUBMENU)

        if choice == "b":
            break

        _execute_transform_action(choice)

        if choice != "b" and _prompt_continue("transform") in ("b", "q"):
            break


def _execute_transform_action(choice: str) -> None:
    """Execute transform action. Rule #4: <60 lines."""
    if choice == "1":
        _handle_transform_clean()
    elif choice == "2":
        _handle_simple_transform("enrich")
    elif choice == "3":
        _handle_simple_transform("filter")
    elif choice == "4":
        _handle_simple_transform("merge")
    elif choice == "5":
        _handle_simple_transform("split")


def _handle_simple_transform(cmd: str) -> None:
    """Handle simple transform commands. Rule #4: <60 lines."""
    file_path: str = Prompt.ask("File path")
    if not file_path:
        return
    run_command_with_args("transform", cmd, file_path)


def _handle_transform_clean() -> None:
    """Handle transform clean with OCR options and file browser. Rule #4: <60 lines."""
    console.print("\n[bold cyan]Transform Clean[/bold cyan]\n")
    console.print("[dim]Advanced text cleaning with OCR support[/dim]\n")

    # Offer browse or manual entry
    browse: bool = (
        questionary.confirm("Browse for file?", default=True, style=MENU_STYLE).ask()
        or False
    )

    file_path: Optional[str] = None
    if browse:
        file_path = _open_file_browser(
            title="Select File to Clean",
            file_types=[
                ("Text files", "*.txt *.md"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            console.print(f"[green]Selected: {file_path}[/green]")
    else:
        file_path = Prompt.ask("File path")

    if not file_path:
        console.print("[yellow]No file selected[/yellow]")
        return

    # Ask about OCR cleanup options
    console.print("\n[yellow]OCR Cleanup Options:[/yellow]")
    group_para: bool = (
        questionary.confirm(
            "Join broken paragraphs?", default=False, style=MENU_STYLE
        ).ask()
        or False
    )
    clean_bullet: bool = (
        questionary.confirm(
            "Normalize bullet characters?", default=False, style=MENU_STYLE
        ).ask()
        or False
    )
    clean_prefix: bool = (
        questionary.confirm(
            "Remove page numbers/headers?", default=False, style=MENU_STYLE
        ).ask()
        or False
    )

    # Build command with options
    cmd_parts: List[str] = ["transform", "clean", f'"{file_path}"']
    if group_para:
        cmd_parts.append("--group-paragraphs")
    if clean_bullet:
        cmd_parts.append("--clean-bullets")
    if clean_prefix:
        cmd_parts.append("--clean-prefix-postfix")

    # Ask for output file
    output: str = Prompt.ask("Output file (optional)", default="")
    if output:
        cmd_parts.extend(["-o", f'"{output}"'])

    run_command(" ".join(cmd_parts))


def handle_index_submenu() -> None:
    """Handle index operations. Rule #1: Max 3 nesting levels."""
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        menu: MenuList = [
            ("1", "Index Info", "Show index details", "info"),
            ("2", "List Indexes", "List all indexes", "list"),
            ("3", "Rebuild Index", "Rebuild index", "rebuild"),
            ("4", "Delete Index", "Delete index", "delete"),
            ("b", "Back", "Return", "back"),
        ]
        choice: str = display_menu("Index Management", menu)

        if choice == "b":
            break

        _handle_index_choice(choice)

        if choice != "b" and _prompt_continue("index") in ("b", "q"):
            break


def _handle_index_choice(choice: str) -> None:
    """Handle index menu choice. Rule #4: Extracted helper."""
    if choice == "2":
        run_command("index list")
        return

    if choice not in ("1", "3", "4"):
        return

    index_name: str = Prompt.ask("Index name", default="default")
    if choice == "1":
        run_command_with_args("index", "info", index_name)
    elif choice == "3":
        run_command_with_args("index", "rebuild", index_name)
    elif choice == "4":
        run_command_with_args("index", "delete", index_name)


def handle_youtube() -> None:
    """Handle YouTube transcript ingestion."""
    console.print("\n[bold cyan]Ingest YouTube Video[/bold cyan]\n")
    url: str = Prompt.ask("YouTube URL (e.g., https://youtube.com/watch?v=...)")
    if not _validate_required(url, "URL"):
        input("\nPress Enter to continue...")
        return
    # Pass URL directly to ingest - it auto-detects YouTube URLs
    run_command_with_args("ingest", url)
    input("\nPress Enter to continue...")


def handle_audio() -> None:
    """Handle audio file transcription with Whisper."""
    console.print("\n[bold cyan]Ingest Audio (Whisper Transcription)[/bold cyan]\n")

    # Check if faster-whisper is installed before proceeding
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        console.print("[red]Error: faster-whisper not installed[/red]\n")
        console.print("Install with: [cyan]pip install faster-whisper[/cyan]")
        console.print("\n[dim]Note: Requires ~1GB download for the whisper model[/dim]")
        input("\nPress Enter to continue...")
        return

    console.print("[dim]Supported: mp3, wav, m4a, flac, ogg, webm[/dim]\n")

    # Offer browse or manual entry
    browse: bool = (
        questionary.confirm(
            "Browse for audio file?", default=True, style=MENU_STYLE
        ).ask()
        or False
    )

    path_input: Optional[str] = None
    if browse:
        path_input = _open_file_browser(
            title="Select Audio File",
            file_types=[
                ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.webm"),
                ("MP3", "*.mp3"),
                ("WAV", "*.wav"),
                ("M4A", "*.m4a"),
                ("All files", "*.*"),
            ],
        )
        if path_input:
            console.print(f"[green]Selected: {path_input}[/green]")
    else:
        path_input = Prompt.ask("Path to audio file")

    if not path_input:
        console.print("[yellow]No file selected[/yellow]")
        input("\nPress Enter to continue...")
        return

    path: Optional[Path] = _validate_path(path_input)
    if path is None:
        input("\nPress Enter to continue...")
        return

    # Audio files are auto-detected by extension in ingest
    run_command_with_args("ingest", str(path))
    input("\nPress Enter to continue...")


def handle_data_menu() -> None:
    """Handle [2] Data & Workspace submenu. Rule #1: Max 3 nesting levels."""
    handlers: HandlerDict = {
        "1": handle_ingest,
        "2": handle_youtube,
        "3": handle_audio,
        "4": handle_docs_submenu,
        "5": handle_libraries_submenu,
        "6": handle_organize_submenu,
        "7": handle_transform_submenu,
        "8": handle_index_submenu,
    }
    _run_submenu("Data & Workspace", DATA_MENU, handlers)


# =============================================================================
# ORGANIZATION HANDLERS (Phase 2)
# =============================================================================


def handle_organize_submenu() -> None:
    """Handle organization submenu for tags, bookmarks, annotations."""
    commands: CommandDict = {
        "1": "tag",
        "2": "tags",
        "3": "bookmark",
        "4": "bookmarks",
        "5": "annotate",
        "6": "annotations",
        "7": "mark",
    }
    _run_organize_menu("Organization", ORGANIZE_SUBMENU, commands)


def _run_organize_menu(title: str, menu: MenuList, commands: CommandDict) -> None:
    """Run organization menu with appropriate prompts. Rule #4: <60 lines."""
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu(title, menu)

        if choice == "b":
            break

        _execute_organize_action(choice, commands)


def _execute_organize_action(choice: str, commands: CommandDict) -> None:
    """Execute organization action. Rule #1: Max 3 nesting levels."""
    if choice not in commands:
        return

    cmd: str = commands[choice]

    # Commands that need chunk_id prompt
    chunk_id_cmds: set = {"tag", "bookmark", "annotate", "mark"}
    # Commands that run without prompts
    list_cmds: set = {"tags", "bookmarks", "annotations"}

    if cmd in list_cmds:
        run_command(cmd)
        input("\nPress Enter to continue...")
        return

    if cmd in chunk_id_cmds:
        chunk_id: str = Prompt.ask("Chunk ID")
        if not chunk_id:
            return
        if cmd == "tag":
            tag_name: str = Prompt.ask("Tag name")
            run_command_with_args(cmd, chunk_id, tag_name)
        elif cmd == "bookmark":
            note: str = Prompt.ask("Note (optional)", default="")
            if note:
                run_command(f'{cmd} "{chunk_id}" --note "{note}"')
            else:
                run_command_with_args(cmd, chunk_id)
        elif cmd == "annotate":
            text: str = Prompt.ask("Annotation text")
            run_command_with_args(cmd, chunk_id, text)
        elif cmd == "mark":
            status: str = Prompt.ask("Status", choices=["read", "unread"])
            run_command(f'{cmd} "{chunk_id}" --{status}')
        input("\nPress Enter to continue...")


# =============================================================================
# [3] SEARCH & DISCOVERY HANDLERS
# =============================================================================


def handle_academic_search() -> None:
    """Academic search options."""
    menu: MenuList = [
        ("1", "Semantic Scholar", "Search Semantic Scholar", "scholar"),
        ("2", "CrossRef", "Search by DOI", "crossref"),
        ("3", "Educational", "Find learning materials", "edu"),
        ("b", "Back", "Return", "back"),
    ]
    commands: Dict[str, Tuple[str, str]] = {
        "1": ("discovery scholar", "Search query"),
        "2": ("discovery crossref-search", "Search query"),
        "3": ("discovery educational", "Topic"),
    }
    _run_prompted_submenu("Academic Search", menu, commands)


def handle_arxiv() -> None:
    """arXiv portal. Rule #1: Max 3 nesting levels."""
    menu: MenuList = [
        ("1", "Search arXiv", "Search papers", "search"),
        ("2", "Download Paper", "Download by ID", "download"),
        ("b", "Back", "Return", "back"),
    ]
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu("arXiv Portal", menu)

        if choice == "b":
            break

        _handle_arxiv_choice(choice)

        if choice != "b" and _prompt_continue("arxiv") in ("b", "q"):
            break


def _handle_arxiv_choice(choice: str) -> None:
    """Handle arXiv menu choice. Rule #4: Extracted helper."""
    if choice == "1":
        query: str = Prompt.ask("Search query")
        if not query:
            return
        run_command_with_args("discovery", "arxiv", query)
    elif choice == "2":
        paper_id: str = Prompt.ask("arXiv ID (e.g., 2301.00001)")
        if not paper_id:
            return
        run_command_with_args("discovery", "arxiv-download", paper_id)


def handle_legal_research() -> None:
    """Legal research via CourtListener. Rule #4: <60 lines."""
    commands: CommandDict = {
        "1": "discovery court",
        "2": "discovery court-detail",
        "3": "discovery court-download",
        "4": "discovery court-list",
    }
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu("Legal Research", LEGAL_SUBMENU)

        if choice == "b":
            break

        _execute_legal_action(choice, commands)


def _execute_legal_action(choice: str, commands: CommandDict) -> None:
    """Execute legal research action. Rule #1: Max 3 nesting levels."""
    if choice not in commands:
        return

    cmd: str = commands[choice]

    if choice == "1":
        query: str = Prompt.ask("Search query (case name, topic)")
        if not query:
            return
        run_command(f'{cmd} "{query}"')
    elif choice == "2":
        cluster_id: str = Prompt.ask("Cluster ID")
        if not cluster_id:
            return
        run_command_with_args("discovery", "court-detail", cluster_id)
    elif choice == "3":
        cluster_id = Prompt.ask("Cluster ID to download")
        if not cluster_id:
            return
        run_command_with_args("discovery", "court-download", cluster_id)
    elif choice == "4":
        run_command(cmd)

    input("\nPress Enter to continue...")


def handle_security_research() -> None:
    """Security/CVE research. Rule #4: <60 lines."""
    commands: CommandDict = {
        "1": "discovery cve",
        "2": "discovery cve-get",
    }
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu("Security Research", SECURITY_SUBMENU)

        if choice == "b":
            break

        _execute_security_action(choice, commands)


def _execute_security_action(choice: str, commands: CommandDict) -> None:
    """Execute security research action. Rule #1: Max 3 nesting levels."""
    if choice not in commands:
        return

    if choice == "1":
        query: str = Prompt.ask("Product/software name")
        if not query:
            return
        run_command(f'discovery cve "{query}"')
    elif choice == "2":
        cve_id: str = Prompt.ask("CVE ID (e.g., CVE-2023-12345)")
        if not cve_id:
            return
        run_command_with_args("discovery", "cve-get", cve_id)

    input("\nPress Enter to continue...")


def handle_discovery_menu() -> None:
    """Handle [3] Search & Discovery submenu. Rule #1: Max 3 nesting levels."""
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu("Search & Discovery", DISCOVERY_MENU)
        if choice == "b":
            break

        # Execute action - extracted to helper
        _execute_discovery_action(choice)


# =============================================================================
# [4] DEEP ANALYSIS HANDLERS
# =============================================================================


def handle_comprehension() -> None:
    """Comprehension tools."""
    _run_simple_commands(
        "Comprehension",
        {
            "1": ("Explain Concept", "comprehension explain", "Concept"),
            "2": (
                "Compare Concepts",
                "comprehension compare",
                "Concepts (comma-separated)",
            ),
            "3": ("Connect Concepts", "comprehension connect", "Concepts"),
        },
    )


def handle_argument() -> None:
    """Argument analysis."""
    _run_simple_commands(
        "Argument Analysis",
        {
            "1": ("Detect Conflicts", "argument conflicts", "Topic"),
            "2": ("Counter Arguments", "argument counter", "Claim"),
            "3": ("Debate Analysis", "argument debate", "Claim"),
            "4": ("Find Gaps", "argument gaps", "Topic"),
            "5": ("Find Support", "argument support", "Claim"),
        },
    )


def handle_literary() -> None:
    """Literary analysis."""
    _run_simple_commands(
        "Literary Analysis",
        {
            "1": ("Story Arc", "lit arc", "Work title"),
            "2": ("Characters", "lit character", "Work title"),
            "3": ("Outline", "lit outline", "Work title"),
            "4": ("Symbols", "lit symbols", "Work title"),
            "5": ("Themes", "lit themes", "Work title"),
        },
    )


def handle_content_analysis() -> None:
    """Content analysis."""
    commands: CommandDict = {
        "1": "analyze topics",
        "2": "analyze entities",
        "3": "analyze relationships",
        "4": "analyze connections",
        "5": "analyze timeline",
        "6": "analyze similarity",
        "7": "analyze duplicates",
        "8": "analyze knowledge-graph",
    }
    menu: MenuList = [
        ("1", "Topics", "Extract topics", "topics"),
        ("2", "Entities", "Analyze entities", "ent"),
        ("3", "Relationships", "Entity relationships", "rel"),
        ("4", "Connections", "Lateral connections", "conn"),
        ("5", "Timeline", "Build timeline", "time"),
        ("6", "Similarity", "Find similar docs", "sim"),
        ("7", "Duplicates", "Detect duplicates", "dup"),
        ("8", "Knowledge Graph", "Build graph", "graph"),
        ("b", "Back", "Return", "back"),
    ]
    _run_command_submenu("Content Analysis", menu, commands)


def handle_code_analysis() -> None:
    """Code analysis."""
    _run_simple_commands(
        "Code Analysis",
        {
            "1": ("Analyze Code", "code analyze", "File path"),
            "2": ("Document Code", "code document", "File path"),
            "3": ("Explain Code", "code explain", "File path"),
            "4": ("Code Map", "code map", "Directory path"),
        },
    )


def handle_stored_analyses() -> None:
    """Manage stored analyses."""
    menu: MenuList = [
        ("1", "List Analyses", "View all stored analyses", "list"),
        ("2", "Search Analyses", "Semantic search", "search"),
        ("3", "Show Analysis", "View specific analysis", "show"),
        ("4", "Delete Analysis", "Remove an analysis", "delete"),
        ("5", "For Document", "Analyses for a document", "doc"),
        ("6", "Statistics", "Analysis storage stats", "stats"),
        ("b", "Back", "Return", "back"),
    ]

    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu("Stored Analyses", menu)

        if choice == "b":
            break

        _handle_stored_analysis_choice(choice)

        if choice != "b" and _prompt_continue("stored analyses") in ("b", "q"):
            break


def _handle_stored_analysis_choice(choice: str) -> None:
    """Handle stored analysis menu choice. Rule #1: Max 3 nesting levels."""
    simple_commands: Dict[str, str] = {
        "1": "analysis list",
        "6": "analysis stats",
    }

    # Handle simple commands with no prompts
    if choice in simple_commands:
        run_command(simple_commands[choice])
        return

    # Handle commands requiring prompts
    _execute_prompted_analysis_action(choice)


def handle_factcheck() -> None:
    """Handle fact checking and contradiction detection."""
    console.print("\n[bold cyan]Fact Checker[/bold cyan]\n")
    _run_simple_commands(
        "Fact Checker",
        {
            "1": ("Find Contradictions", "analyze contradictions", "Topic to check"),
            "2": ("Link Evidence", "analyze evidence", "Claim text"),
            "3": ("Agent Research", "agent run", "Research query"),
        },
    )


def handle_analysis_menu() -> None:
    """Handle [4] Deep Analysis submenu. Rule #1: Max 3 nesting levels."""
    handlers: HandlerDict = {
        "1": handle_comprehension,
        "2": handle_argument,
        "3": handle_literary,
        "4": handle_content_analysis,
        "5": handle_code_analysis,
        "6": handle_factcheck,
        "7": handle_stored_analyses,
    }
    _run_submenu("Deep Analysis", ANALYSIS_MENU, handlers)


# =============================================================================
# [5] WRITING & STUDY HANDLERS
# =============================================================================


def handle_academic_writing() -> None:
    """Academic writing tools."""
    _run_simple_commands(
        "Academic Writing",
        {
            "1": ("Generate Draft", "writing draft", "Topic"),
            "2": ("Create Outline", "writing outline", "Topic"),
            "3": ("Paraphrase", "writing paraphrase", "Text to paraphrase"),
            "4": ("Find Quotes", "writing quote", "Topic"),
            "5": ("Rewrite", "writing rewrite", "Text to rewrite"),
            "6": ("Simplify", "writing simplify", "Text to simplify"),
            "7": ("Thesis Help", "writing thesis", "Thesis statement"),
        },
    )


def handle_citation_tools() -> None:
    """Citation tools including document cite commands."""
    commands: CommandDict = {
        "1": "citation extract",
        "2": "citation format",
        "3": "citation bibliography",
        "4": "citation validate",
        "5": "citation graph",
        "6": "writing cite check",
        "7": "writing cite format",
        "8": "writing cite insert",
    }
    _run_citation_menu("Citation Tools", CITATION_SUBMENU, commands)


def _run_citation_menu(title: str, menu: MenuList, commands: CommandDict) -> None:
    """Run citation menu with prompts. Rule #4: <60 lines."""
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu(title, menu)

        if choice == "b":
            break

        _execute_citation_action(choice, commands)


def _execute_citation_action(choice: str, commands: CommandDict) -> None:
    """Execute citation action. Rule #1: Max 3 nesting levels."""
    if choice not in commands:
        return

    cmd: str = commands[choice]

    # Commands that need document path
    doc_cmds: set = {"6", "7", "8"}  # writing cite check/format/insert
    # Commands that run without prompts
    no_prompt: set = {"1", "3", "4", "5"}  # extract, bibliography, validate, graph

    if choice in no_prompt:
        run_command(cmd)
        input("\nPress Enter to continue...")
        return

    if choice == "2":  # citation format needs style
        style: str = Prompt.ask(
            "Style", choices=["apa", "mla", "chicago"], default="apa"
        )
        run_command(f"citation format --style {style}")
        input("\nPress Enter to continue...")
        return

    if choice in doc_cmds:
        doc_path: str = Prompt.ask("Document path")
        if not doc_path:
            return
        run_command(f'{cmd} "{doc_path}"')
        input("\nPress Enter to continue...")


def handle_study_tools() -> None:
    """Study tools with SM-2 spaced repetition."""
    _run_simple_commands(
        "Study Tools",
        {
            "1": ("Flashcards", "study flashcards", "Topic"),
            "2": ("Glossary", "study glossary", "Topic"),
            "3": ("Study Notes", "study notes", "Topic"),
            "4": ("Quiz", "study quiz", "Topic"),
            "5": ("Review Due Cards", "study review", None),
            "6": ("Study Stats", "study stats", None),
            "7": ("Schedule Info", "study schedule", "Topic"),
        },
    )


def handle_export_submenu() -> None:
    """Handle export options."""
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu("Export", EXPORT_SUBMENU)

        if choice == "b":
            break

        _handle_export_choice(choice)

        if choice != "b" and _prompt_continue("export") in ("b", "q"):
            break


def _handle_export_choice(choice: str) -> None:
    """Handle export menu choice. Rule #1: Max 3 nesting levels."""
    # Simple file exports
    simple_exports: Dict[str, Tuple[str, str]] = {
        "1": ("markdown", "export.md"),
        "2": ("json", "export.json"),
        "3": ("pdf", "export.pdf"),
        "4": ("outline", "outline.md"),
    }

    if choice in simple_exports:
        fmt, default = simple_exports[choice]
        out: str = Prompt.ask("Output file", default=default)
        run_command(f"export {fmt} {out}")
        return

    if choice == "5":
        query: str = Prompt.ask("Context query")
        if not query:
            return
        run_command_with_args("export", "context", "--query", query)
        return

    if choice == "6":
        out = Prompt.ask("Output folder", default="./study_package")
        run_command_with_args("export", "folder-export", out)
        return

    if choice == "7":
        console.print("\n[bold cyan]Knowledge Graph Export[/bold cyan]\n")
        out = Prompt.ask("Output HTML file", default="knowledge_graph.html")
        topic: str = Prompt.ask("Filter by topic (optional)", default="")
        if topic:
            run_command_with_args("export", "knowledge-graph", out, "--topic", topic)
        else:
            run_command_with_args("export", "knowledge-graph", out)
        return

    # Package operations (Phase 2)
    if choice == "8":
        out = Prompt.ask("Package filename", default="corpus.pack")
        run_command_with_args("export", "pack", out)
        return

    if choice == "9":
        pack_file: str = Prompt.ask("Package file to import")
        if not pack_file:
            return
        run_command_with_args("export", "unpack", pack_file)
        return

    if choice == "0":
        pack_file = Prompt.ask("Package file")
        if not pack_file:
            return
        run_command_with_args("export", "info", pack_file)


def handle_writing_menu() -> None:
    """Handle [5] Writing & Study submenu. Rule #1: Max 3 nesting levels."""
    handlers: HandlerDict = {
        "1": handle_academic_writing,
        "2": handle_citation_tools,
        "3": handle_study_tools,
        "4": handle_export_submenu,
    }
    _run_submenu("Writing & Study", WRITING_MENU, handlers)


# =============================================================================
# [6] SYSTEM ADMIN HANDLERS
# =============================================================================


def _get_current_model(config_path: Path) -> str:
    """Read current model name from config.

    Rule #7: Validate config_path
    """
    assert config_path is not None, "Config path required"

    if not config_path.exists():
        return "None"

    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_path = config.get("llm", {}).get("llamacpp", {}).get("model_path")
        return Path(model_path).name if model_path else "None"
    except Exception:
        return "None"


def _scan_models() -> List[Tuple[str, str, float]]:
    """Scan for available .gguf models."""
    models_dir: Path = Path.cwd() / ".data" / "models"
    models: List[Tuple[str, str, float]] = []

    if not models_dir.exists():
        return models

    for f in models_dir.glob("*.gguf"):
        size: float = f.stat().st_size / (1024**3)
        models.append((f.name, str(f), size))

    models.sort(key=lambda x: x[2])
    return models


def _update_model_config(config_path: Path, model_path: str) -> None:
    """Update config file with new model path.

    Rule #7: Validate parameters
    """
    assert config_path.exists(), "Config file must exist"
    assert model_path, "Model path required"

    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "llm" not in config:
        config["llm"] = {}
    if "llamacpp" not in config["llm"]:
        config["llm"]["llamacpp"] = {}

    config["llm"]["llamacpp"]["model_path"] = model_path
    config["llm"]["default_provider"] = "llamacpp"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def handle_llm_settings() -> None:
    """Configure LLM settings."""
    console.print("\n[bold cyan]LLM Configuration[/bold cyan]\n")

    config_path: Path = Path.cwd() / "ingestforge.yaml"
    current_model: str = _get_current_model(config_path)
    models: List[Tuple[str, str, float]] = _scan_models()

    if not models:
        console.print("[yellow]No models in .data/models/[/yellow]")
        input("Press Enter to continue...")
        return

    _display_model_table(models, current_model)
    _handle_model_selection(config_path, models)
    input("\nPress Enter to continue...")


def _display_model_table(models: List[Tuple[str, str, float]], current: str) -> None:
    """Display model selection table. Rule #4: Extracted helper."""
    table = Table(title="Available Models")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Model", style="white")
    table.add_column("Size", style="green")
    table.add_column("Status", style="yellow")

    for i, (name, _, size) in enumerate(models, 1):
        status: str = "[green]Active[/green]" if name == current else ""
        table.add_row(str(i), name, f"{size:.1f} GB", status)

    console.print(table)


def _handle_model_selection(
    config_path: Path, models: List[Tuple[str, str, float]]
) -> None:
    """Handle model selection input. Rule #4: Extracted helper."""
    choice: str = Prompt.ask("\nModel # to switch (Enter to keep)", default="")
    if not choice:
        return

    idx: Optional[int] = _parse_int_choice(choice, len(models))
    if idx is None:
        return

    try:
        _, path, _ = models[idx]
        _update_model_config(config_path, path)
        console.print(f"[green]Switched to {models[idx][0]}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def handle_config_submenu() -> None:
    """Configuration management."""
    commands: CommandDict = {
        "1": "config show",
        "2": "config list",
        "3": "config validate",
        "4": "config reset",
    }
    menu: MenuList = [
        ("1", "Show Config", "Display current config", "show"),
        ("2", "List All", "List all settings", "list"),
        ("3", "Validate", "Validate config", "val"),
        ("4", "Reset", "Reset to defaults", "reset"),
        ("b", "Back", "Return", "back"),
    ]
    _run_command_submenu("Configuration", menu, commands)


def handle_storage_submenu() -> None:
    """Storage management."""
    commands: CommandDict = {
        "1": "storage health",
        "2": "storage stats",
        "3": "storage migrate",
    }
    menu: MenuList = [
        ("1", "Health Check", "Check storage health", "health"),
        ("2", "Statistics", "Show storage stats", "stats"),
        ("3", "Migrate", "Migrate storage", "migrate"),
        ("b", "Back", "Return", "back"),
    ]
    _run_command_submenu("Storage", menu, commands)


def handle_maintenance_submenu() -> None:
    """Maintenance tools."""
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        menu: MenuList = [
            ("1", "Backup", "Backup project", "backup"),
            ("2", "Restore", "Restore from backup", "restore"),
            ("3", "Cleanup", "Clean temp files", "cleanup"),
            ("4", "Optimize", "Optimize storage", "optimize"),
            ("b", "Back", "Return", "back"),
        ]
        choice: str = display_menu("Maintenance", menu)

        if choice == "b":
            break

        _handle_maintenance_choice(choice)

        if choice != "b" and _prompt_continue("maintenance") in ("b", "q"):
            break


def _handle_maintenance_choice(choice: str) -> None:
    """Handle maintenance menu choice. Rule #1: Max 3 nesting levels."""
    if choice == "1":
        out: str = Prompt.ask("Backup file", default="backup.zip")
        run_command_with_args("maintenance", "backup", out)
        return

    if choice == "2":
        inp: str = Prompt.ask("Backup file to restore")
        if not inp:
            return
        run_command_with_args("maintenance", "restore", inp)
        return

    if choice == "3":
        run_command("maintenance cleanup")
    elif choice == "4":
        run_command("maintenance optimize")


def handle_monitor_submenu() -> None:
    """Monitor tools."""
    commands: CommandDict = {
        "1": "monitor diagnostics",
        "2": "monitor health",
        "3": "monitor logs",
        "4": "monitor metrics",
    }
    menu: MenuList = [
        ("1", "Diagnostics", "Run diagnostics", "diag"),
        ("2", "Health Check", "System health", "health"),
        ("3", "View Logs", "View logs", "logs"),
        ("4", "Metrics", "Show metrics", "metrics"),
        ("b", "Back", "Return", "back"),
    ]
    _run_command_submenu("Monitor", menu, commands)


def handle_workflow_submenu() -> None:
    """Workflow automation."""
    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        menu: MenuList = [
            ("1", "Batch Operations", "Run batch ops", "batch"),
            ("2", "Pipeline", "Execute pipeline", "pipeline"),
            ("3", "Schedule", "Schedule automation", "schedule"),
            ("b", "Back", "Return", "back"),
        ]
        choice: str = display_menu("Workflow", menu)

        if choice == "b":
            break

        _handle_workflow_choice(choice)

        if choice != "b" and _prompt_continue("workflow") in ("b", "q"):
            break


def _handle_workflow_choice(choice: str) -> None:
    """Handle workflow menu choice. Rule #4: Extracted helper."""
    if choice == "1":
        files: str = Prompt.ask("Files pattern (e.g., *.pdf)")
        if files:
            run_command_with_args("workflow", "batch", files)
    elif choice == "2":
        pipeline: str = Prompt.ask("Pipeline name")
        if pipeline:
            run_command_with_args("workflow", "pipeline", pipeline)
    elif choice == "3":
        run_command("workflow schedule")


def handle_api_server() -> None:
    """Handle API server management."""
    console.print("\n[bold cyan]API Server[/bold cyan]\n")
    _run_simple_commands(
        "API Server",
        {
            "1": ("Start Server", "api start", None),
            "2": ("Stop Server", "api stop", None),
            "3": ("Server Status", "api status", None),
            "4": ("View Docs", "api docs", None),
        },
    )


def handle_admin_menu() -> None:
    """Handle [6] System Admin submenu. Rule #1: Max 3 nesting levels."""
    handlers: HandlerDict = {
        "1": handle_llm_settings,
        "2": handle_config_submenu,
        "3": handle_storage_submenu,
        "4": handle_maintenance_submenu,
        "5": handle_monitor_submenu,
        "6": handle_workflow_submenu,
        "7": handle_api_server,
    }
    _run_submenu("System Admin", ADMIN_MENU, handlers)


# =============================================================================
# GENERIC SUBMENU HELPERS (Rule #4: Small, reusable functions)
# =============================================================================


def _run_submenu(title: str, menu: MenuList, handlers: HandlerDict) -> None:
    """Run a submenu with function handlers.

    Rule #7: Validate parameters
    """
    assert title, "Title required"
    assert menu, "Menu required"
    assert handlers, "Handlers required"

    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu(title, menu)

        if choice == "b":
            break
        if choice == "q":
            raise SystemExit(0)

        handler: Optional[Callable[[], None]] = handlers.get(choice)
        if handler:
            handler()


def _run_command_submenu(title: str, menu: MenuList, commands: CommandDict) -> None:
    """Run a submenu that executes commands.

    Rule #7: Validate parameters
    """
    assert title, "Title required"
    assert menu, "Menu required"
    assert commands, "Commands required"

    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu(title, menu)

        if choice == "b":
            break

        cmd: Optional[str] = commands.get(choice)
        if cmd:
            run_command(cmd)
            if _prompt_continue(title.lower()) in ("b", "q"):
                break


def _run_prompted_submenu(
    title: str, menu: MenuList, commands: Dict[str, Tuple[str, str]]
) -> None:
    """Run submenu with prompted commands.

    Rule #7: Validate parameters
    """
    assert title, "Title required"
    assert menu, "Menu required"
    assert commands, "Commands required"

    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        choice: str = display_menu(title, menu)

        if choice == "b":
            break

        if choice in commands:
            cmd, prompt_text = commands[choice]
            value: str = Prompt.ask(prompt_text)
            if value:
                run_command_with_args(*cmd.split(), value)
            if _prompt_continue(title.lower()) in ("b", "q"):
                break


def _run_simple_commands(
    title: str, commands: Dict[str, Tuple[str, str, Optional[str]]]
) -> None:
    """Run a simple command menu with prompts.

    Rule #7: Validate parameters
    """
    assert title, "Title required"
    assert commands, "Commands required"

    for _ in range(_MAX_MENU_ITERATIONS):
        clear_screen()
        display_banner()

        # Build menu from commands
        menu: MenuList = [
            (str(i), label, "", key)
            for i, (key, (label, _, _)) in enumerate(commands.items(), 1)
        ]
        menu.append(("b", "Back", "Return", "back"))

        _display_simple_menu(title, menu)

        choice: str = Prompt.ask("\n[cyan]Select[/cyan]", default="b")
        if choice == "b":
            break

        _execute_simple_command(choice, commands, title)


def _display_simple_menu(title: str, menu: MenuList) -> None:
    """Display simple menu panel. Rule #4: Extracted helper."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="cyan bold", width=4)
    table.add_column("Option", style="white")

    for key, label, _, _ in menu:
        table.add_row(f"[{key}]", label)

    panel = Panel(table, title=f"[bold]{title}[/bold]", border_style="cyan")
    console.print(panel)


def _execute_simple_command(
    choice: str, commands: Dict[str, Tuple[str, str, Optional[str]]], title: str
) -> bool:
    """Execute command from simple menu. Rule #4: Extracted helper."""
    keys: List[str] = list(commands.keys())
    idx: Optional[int] = _parse_int_choice(choice, len(keys))

    if idx is None:
        return False

    key: str = keys[idx]
    _, cmd, prompt_text = commands[key]

    if prompt_text:
        value: str = Prompt.ask(prompt_text)
        if value:
            run_command_with_args(*cmd.split(), value)
    else:
        run_command(cmd)

    return _prompt_continue(title.lower()) not in ("b", "q")


def handle_status() -> None:
    """Show project status."""
    run_command("status")
    input("\nPress Enter to continue...")


# =============================================================================
# QUICK BAR HANDLERS (Rule #4: Small, focused functions)
# =============================================================================


def _quickbar_query() -> None:
    """Quick Bar: Query knowledge base.

    Rule #4: Function <60 lines
    Rule #9: Full type hints
    """
    console.print("\n[bold cyan]Quick Query[/bold cyan]\n")
    query: str = Prompt.ask("Enter your question")
    query = query.strip().strip('"').strip("'")
    if not _validate_required(query, "Question"):
        input("\nPress Enter to continue...")
        return
    run_command_with_args("query", query)
    input("\nPress Enter to continue...")


def _quickbar_ingest() -> None:
    """Quick Bar: Ingest documents.

    Rule #4: Function <60 lines
    Rule #9: Full type hints
    """
    console.print("\n[bold cyan]Quick Ingest[/bold cyan]\n")
    path_input: str = Prompt.ask("Path to file or folder")
    path: Optional[Path] = _validate_path(path_input)
    if path is None:
        input("\nPress Enter to continue...")
        return

    recursive: bool = path.is_dir()
    cmd: List[str] = ["ingest", str(path)]
    if recursive:
        cmd.append("--recursive")
    run_command_with_args(*cmd)
    input("\nPress Enter to continue...")


def _quickbar_agent() -> None:
    """Quick Bar: Run agent research.

    Rule #4: Function <60 lines
    Rule #9: Full type hints
    """
    console.print("\n[bold cyan]Agent Research[/bold cyan]\n")
    console.print("[dim]The agent will autonomously research your topic.[/dim]\n")
    topic: str = Prompt.ask("Research topic or question")
    if not _validate_required(topic, "Topic"):
        input("\nPress Enter to continue...")
        return
    run_command_with_args("agent", "run", topic)
    input("\nPress Enter to continue...")


def _quickbar_status() -> None:
    """Quick Bar: Show project status.

    Rule #4: Function <60 lines
    Rule #9: Full type hints
    """
    run_command("status")
    input("\nPress Enter to continue...")


def _quickbar_flashcards() -> None:
    """Quick Bar: Generate flashcards.

    Rule #4: Function <60 lines
    Rule #9: Full type hints
    """
    console.print("\n[bold cyan]Quick Flashcards[/bold cyan]\n")
    topic: str = Prompt.ask("Topic for flashcards")
    if not _validate_required(topic, "Topic"):
        input("\nPress Enter to continue...")
        return
    run_command_with_args("study", "flashcards", topic)
    input("\nPress Enter to continue...")


def _quickbar_export() -> None:
    """Quick Bar: Quick export to markdown.

    Rule #4: Function <60 lines
    Rule #9: Full type hints
    """
    console.print("\n[bold cyan]Quick Export[/bold cyan]\n")
    console.print("[dim]Export formats: markdown, json, pdf[/dim]\n")

    format_choice: str = Prompt.ask(
        "Format", choices=["markdown", "json", "pdf"], default="markdown"
    )
    output_file: str = Prompt.ask(
        "Output filename",
        default=f"export.{format_choice[:2] if format_choice != 'markdown' else 'md'}",
    )
    run_command_with_args("export", format_choice, output_file)
    input("\nPress Enter to continue...")


def _quickbar_gui() -> None:
    """Quick Bar: Launch GUI.

    Rule #4: Function <60 lines
    Rule #9: Full type hints
    """
    console.print("\n[cyan]Launching GUI...[/cyan]\n")
    from ingestforge.cli.interactive.gui_menu import main as gui_main

    gui_main()


def _quickbar_help() -> None:
    """Quick Bar: Show help and available commands.

    Rule #4: Function <60 lines
    Rule #9: Full type hints
    """
    console.print("\n[bold cyan]IngestForge Quick Reference[/bold cyan]\n")

    # Quick Bar section
    console.print("[bold yellow]Quick Bar Keys:[/bold yellow]")
    for key, label, _ in QUICK_BAR:
        console.print(f"  [cyan][{key.upper()}][/cyan] {label}")

    console.print("\n[bold yellow]Main Menu Pillars:[/bold yellow]")
    console.print("  [cyan][1][/cyan] Ingest & Organize - Documents, Libraries, Tags")
    console.print("  [cyan][2][/cyan] Search & Discover - Query, Academic, Agent")
    console.print("  [cyan][3][/cyan] Analyze & Understand - Comprehension, Arguments")
    console.print("  [cyan][4][/cyan] Create & Export - Study, Writing, Citations")
    console.print("  [cyan][0][/cyan] System - Config, LLM, Storage, API")

    console.print("\n[bold yellow]CLI Commands:[/bold yellow]")
    console.print("  [dim]ingestforge --help       Full command reference[/dim]")
    console.print("  [dim]ingestforge doctor       System diagnostics[/dim]")
    console.print("  [dim]ingestforge quickstart   New user wizard[/dim]")

    input("\nPress Enter to continue...")


# Quick Bar handler dispatch (Rule #6: Module constant)
QUICK_BAR_HANDLERS: Dict[str, Callable[[], None]] = {
    "q": _quickbar_query,
    "i": _quickbar_ingest,
    "a": _quickbar_agent,
    "s": _quickbar_status,
    "f": _quickbar_flashcards,
    "e": _quickbar_export,
    "g": _quickbar_gui,
    "?": _quickbar_help,
}


def handle_quick_bar(key: str) -> bool:
    """Handle Quick Bar key press.

    Rule #4: Function <60 lines
    Rule #7: Validate parameters
    Rule #9: Full type hints

    Args:
        key: The key pressed (lowercase)

    Returns:
        True if key was handled, False otherwise
    """
    assert key, "Key cannot be empty"

    handler: Optional[Callable[[], None]] = QUICK_BAR_HANDLERS.get(key.lower())
    if handler:
        handler()
        return True
    return False


# =============================================================================
# PHASE 3: 4-PILLAR HANDLERS
# =============================================================================


def handle_ingest_organize_menu() -> None:
    """Handle [1] Ingest & Organize pillar. Rule #4: <60 lines."""
    handlers: HandlerDict = {
        "1": handle_add_docs_submenu,
        "2": handle_docs_submenu,
        "3": handle_libraries_submenu,
        "4": handle_organize_submenu,
        "5": handle_transform_submenu,
        "6": handle_index_submenu,
    }
    _run_submenu("Ingest & Organize", INGEST_ORGANIZE_MENU, handlers)


def handle_add_docs_submenu() -> None:
    """Handle Add Documents submenu. Rule #4: <60 lines."""
    handlers: HandlerDict = {
        "1": handle_ingest,
        "2": handle_youtube,
        "3": handle_audio,
        "4": _handle_batch_ingest,
    }
    _run_submenu("Add Documents", ADD_DOCS_SUBMENU, handlers)


def _handle_batch_ingest() -> None:
    """Handle batch import. Rule #4: <60 lines."""
    console.print("\n[bold cyan]Batch Import[/bold cyan]\n")
    folder: str = Prompt.ask("Folder path containing files")
    if not folder:
        return
    run_command_with_args("workflow", "batch", folder)
    input("\nPress Enter to continue...")


def handle_search_discover_menu() -> None:
    """Handle [2] Search & Discover pillar. Rule #4: <60 lines."""
    handlers: HandlerDict = {
        "1": handle_query,
        "2": handle_shell,
        "3": handle_agent,
        "4": handle_academic_search,
        "5": handle_legal_research,
        "6": handle_security_research,
        "7": _handle_research_audit,
    }
    _run_submenu("Search & Discover", SEARCH_DISCOVER_MENU, handlers)


def _handle_research_audit() -> None:
    """Handle research audit. Rule #4: <60 lines."""
    _run_simple_commands(
        "Research Audit",
        {
            "1": ("Audit KB", "research audit", None),
            "2": ("Verify Citations", "research verify", None),
        },
    )


def handle_analyze_menu() -> None:
    """Handle [3] Analyze & Understand pillar. Rule #4: <60 lines."""
    handlers: HandlerDict = {
        "1": handle_comprehension,
        "2": handle_argument,
        "3": handle_literary,
        "4": handle_content_analysis,
        "5": handle_code_analysis,
        "6": handle_factcheck,
        "7": handle_stored_analyses,
    }
    _run_submenu("Analyze & Understand", ANALYZE_MENU, handlers)


def handle_create_export_menu() -> None:
    """Handle [4] Create & Export pillar. Rule #4: <60 lines."""
    handlers: HandlerDict = {
        "1": handle_study_tools,
        "2": handle_academic_writing,
        "3": handle_citation_tools,
        "4": handle_export_submenu,
        "5": _handle_research_summary,
    }
    _run_submenu("Create & Export", CREATE_EXPORT_MENU, handlers)


def _handle_research_summary() -> None:
    """Handle multi-agent research summary. Rule #4: <60 lines."""
    console.print("\n[bold cyan]Research Summary[/bold cyan]\n")
    topic: str = Prompt.ask("Research topic")
    if not topic:
        return
    run_command_with_args("research", "summarize", topic)
    input("\nPress Enter to continue...")


def handle_system_menu() -> None:
    """Handle [0] System pillar. Rule #4: <60 lines."""
    handlers: HandlerDict = {
        "1": handle_quick_start,
        "2": handle_llm_settings,
        "3": handle_config_submenu,
        "4": handle_storage_submenu,
        "5": handle_maintenance_submenu,
        "6": handle_monitor_submenu,
        "7": handle_workflow_submenu,
        "8": handle_api_server,
    }
    _run_submenu("System", SYSTEM_MENU, handlers)


# =============================================================================
# MAIN MENU RUNNER
# =============================================================================


def _suppress_logging_for_menu() -> None:
    """Suppress console logging to prevent questionary menu corruption.

    RichHandler output interferes with questionary's terminal control,
    causing garbled menu displays. This disables all console handlers
    during interactive menu mode.
    """
    import logging

    # Set all relevant loggers to CRITICAL to suppress INFO/WARNING
    for logger_name in [
        "",
        "ingestforge",
        "ingestforge.storage",
        "ingestforge.pipeline",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        # Also disable any existing handlers
        for handler in logger.handlers[:]:
            if hasattr(handler, "stream"):
                handler.setLevel(logging.CRITICAL)


def _get_quick_input() -> str:
    """Get user input, supporting Quick Bar single-key shortcuts.

    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Returns:
        User input (single key for quick bar, or 'menu' for full menu)
    """
    console.print()
    console.print("[dim]Press Quick Bar key, or Enter for full menu[/dim]")

    try:
        # Use simple input to allow single-key detection
        choice: str = (
            Prompt.ask(
                "[cyan]>[/cyan]",
                default="",
                show_default=False,
            )
            .strip()
            .lower()
        )

        # Empty input means show full menu
        if not choice:
            return "menu"

        return choice

    except (KeyboardInterrupt, EOFError):
        return "quit"


def run_menu() -> int:
    """Run the main interactive menu with Quick Bar support.

    Rule #2: Fixed upper bound for main loop
    Rule #4: Function <60 lines
    """
    from ingestforge.cli.interactive.storage_cache import StorageCache

    # Suppress logging to prevent questionary menu corruption
    _suppress_logging_for_menu()

    # Start background loading immediately
    cache = StorageCache()
    cache.start_loading()

    # Phase 3: 4 Pillars + System structure
    handlers: HandlerDict = {
        "1": handle_ingest_organize_menu,
        "2": handle_search_discover_menu,
        "3": handle_analyze_menu,
        "4": handle_create_export_menu,
        "0": handle_system_menu,
        "s": handle_status,
    }
    for _ in range(_MAX_MAIN_MENU_ITERATIONS):
        clear_screen()
        display_banner()
        display_quick_bar()
        display_dashboard()

        # Get user input (Quick Bar or menu navigation)
        quick_input: str = _get_quick_input()

        # Handle quit
        if quick_input in ("quit", "x"):
            console.print("\n[cyan]Goodbye![/cyan]\n")
            return 0

        # Handle Quick Bar keys
        if quick_input in QUICK_BAR_KEYS:
            handle_quick_bar(quick_input)
            continue

        # Handle numeric pillar shortcuts (1-6, 0)
        if quick_input in handlers:
            handlers[quick_input]()
            continue

        # Show full menu for 'menu' or unrecognized input
        if quick_input == "menu" or quick_input not in QUICK_BAR_KEYS:
            choice: str = display_menu("Main Menu", MAIN_MENU)

            if choice == "q":
                console.print("\n[cyan]Goodbye![/cyan]\n")
                return 0

            handler: Optional[Callable[[], None]] = handlers.get(choice)
            if handler:
                handler()

    return 0


def command() -> None:
    """Launch interactive menu."""
    try:
        exit_code: int = run_menu()
        raise typer.Exit(code=exit_code)
    except KeyboardInterrupt:
        console.print("\n[cyan]Goodbye![/cyan]\n")
        raise typer.Exit(code=0)


if __name__ == "__main__":
    command()
