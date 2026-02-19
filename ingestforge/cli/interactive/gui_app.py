"""GUI Application - Main IngestForge GUI Class.

Extracted from gui_menu.py for modularity (JPL-004.1).

NASA JPL Rule #4: Functions limited to 60 lines.
NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from tkinter import Tk, Frame, Label, Button, LEFT, RIGHT, BOTH, X, Y, BOTTOM

from ingestforge.cli.interactive.gui_theme import Colors, Fonts
from ingestforge.cli.interactive.gui_widgets import StyledEntry, NavButton
from ingestforge.cli.interactive.gui_state import ViewType
from ingestforge.cli.interactive.gui_views_base import BaseView
from ingestforge.cli.interactive.gui_views_dashboard import DashboardView
from ingestforge.cli.interactive.gui_views_import import ImportView
from ingestforge.cli.interactive.gui_views_processing import ProcessingView
from ingestforge.cli.interactive.gui_views_query import QueryView
from ingestforge.cli.interactive.gui_views_analyze import AnalyzeView, LiteraryView
from ingestforge.cli.interactive.gui_views_study import StudyView
from ingestforge.cli.interactive.gui_views_monitor import MonitorView
from ingestforge.cli.interactive.gui_views_settings import SettingsView


class IngestForgeGUI:
    """Main GUI application for IngestForge."""

    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("IngestForge v1.0")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg=Colors.BG_PRIMARY)

        # Enable modern dark title bar on Windows
        self._set_dark_title_bar()

        # State
        self._pipeline = None
        self._config = None
        self._llm_client = None
        self.current_view: ViewType = ViewType.DASHBOARD
        self.views: Dict[ViewType, BaseView] = {}
        self.nav_buttons: Dict[ViewType, NavButton] = {}

        # Build UI
        self._create_layout()
        self._create_views()

        # Show initial view
        self.show_view(ViewType.DASHBOARD)

        # Try to initialize backend
        self._init_backend()

    def _set_dark_title_bar(self) -> None:
        """Enable dark title bar on Windows 10/11."""
        try:
            import ctypes

            # Windows 10 1809+ / Windows 11 dark mode attribute
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19  # For older Windows 10 builds

            # Get window handle
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())

            # Try newer attribute first (Windows 10 2004+, Windows 11)
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(ctypes.c_int(1)),
                ctypes.sizeof(ctypes.c_int),
            )

            # If failed, try older attribute (Windows 10 1809-1903)
            if result != 0:
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_USE_IMMERSIVE_DARK_MODE_OLD,
                    ctypes.byref(ctypes.c_int(1)),
                    ctypes.sizeof(ctypes.c_int),
                )

            # Force window to redraw with new title bar
            self.root.update()

        except Exception as e:
            # Not on Windows or DWM not available - log and continue
            import logging

            logging.debug(f"Dark title bar not available: {e}")

    def _create_layout(self) -> None:
        """Create main layout structure."""
        # Header
        self._create_header()

        # Main container
        main = Frame(self.root, bg=Colors.BG_PRIMARY)
        main.pack(fill=BOTH, expand=True)

        # Sidebar
        self._create_sidebar(main)

        # Content area
        self.content = Frame(main, bg=Colors.BG_PRIMARY)
        self.content.pack(side=RIGHT, fill=BOTH, expand=True)

        # Status bar
        self._create_status_bar()

    def _create_header(self) -> None:
        """Create header with logo and title."""
        header = Frame(self.root, bg=Colors.BG_HEADER, height=60)
        header.pack(fill=X)
        header.pack_propagate(False)

        # Logo and title
        logo_frame = Frame(header, bg=Colors.BG_HEADER)
        logo_frame.pack(side=LEFT, padx=20, pady=10)

        Label(
            logo_frame,
            text="⚙️",
            font=("Segoe UI", 24),
            fg=Colors.ACCENT_PRIMARY,
            bg=Colors.BG_HEADER,
        ).pack(side=LEFT)

        Label(
            logo_frame,
            text="IngestForge",
            font=Fonts.H1,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_HEADER,
        ).pack(side=LEFT, padx=(10, 0))

        # Right side controls
        controls = Frame(header, bg=Colors.BG_HEADER)
        controls.pack(side=RIGHT, padx=20)

        # Search (placeholder)
        search = StyledEntry(controls, placeholder="Search...", width=25)
        search.pack(side=LEFT, padx=(0, 20))

        # Help button
        Button(
            controls,
            text="?",
            font=Fonts.BODY_BOLD,
            bg=Colors.BG_TERTIARY,
            fg=Colors.TEXT_PRIMARY,
            relief="flat",
            width=3,
            cursor="hand2",
        ).pack(side=LEFT, padx=5)

        # Settings button
        Button(
            controls,
            text="⚙",
            font=Fonts.BODY_BOLD,
            bg=Colors.BG_TERTIARY,
            fg=Colors.TEXT_PRIMARY,
            relief="flat",
            width=3,
            cursor="hand2",
            command=lambda: self.show_view(ViewType.SETTINGS),
        ).pack(side=LEFT, padx=5)

    def _create_sidebar(self, parent: Frame) -> None:
        """Create navigation sidebar."""
        sidebar = Frame(parent, bg=Colors.BG_SECONDARY, width=200)
        sidebar.pack(side=LEFT, fill=Y)
        sidebar.pack_propagate(False)

        # Navigation sections
        nav_items = [
            ("Documents", ViewType.DOCUMENTS, "📄"),
            ("Query", ViewType.QUERY, "🔍"),
            ("Analyze", ViewType.ANALYZE, "📊"),
            ("Literary", ViewType.LITERARY, "📚"),
            ("Study", ViewType.STUDY, "📝"),
            ("Export", ViewType.EXPORT, "📤"),
            None,  # Separator
            ("Monitor", ViewType.MONITOR, "📈"),
            ("Settings", ViewType.SETTINGS, "⚙️"),
        ]

        # Home/Dashboard button
        home_btn = NavButton(
            sidebar,
            text="Dashboard",
            icon="🏠",
            active=True,
            command=lambda: self.show_view(ViewType.DASHBOARD),
        )
        home_btn.pack(fill=X, pady=(15, 5))
        self.nav_buttons[ViewType.DASHBOARD] = home_btn

        for item in nav_items:
            if item is None:
                # Separator
                Frame(sidebar, bg=Colors.BG_PRIMARY, height=1).pack(fill=X, pady=15)
                continue

            text, view_type, icon = item
            btn = NavButton(
                sidebar,
                text=text,
                icon=icon,
                command=lambda vt=view_type: self.show_view(vt),
            )
            btn.pack(fill=X)
            self.nav_buttons[view_type] = btn

    def _create_status_bar(self) -> None:
        """Create status bar."""
        status = Frame(self.root, bg=Colors.BG_TERTIARY, height=30)
        status.pack(fill=X, side=BOTTOM)
        status.pack_propagate(False)

        # Status text
        self.status_label = Label(
            status,
            text="Ready",
            font=Fonts.SMALL,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_TERTIARY,
        )
        self.status_label.pack(side=LEFT, padx=15)

        # Separators and info
        Label(
            status,
            text="│",
            font=Fonts.SMALL,
            fg=Colors.TEXT_DIMMED,
            bg=Colors.BG_TERTIARY,
        ).pack(side=LEFT)

        self.llm_status = Label(
            status,
            text="LLM: llama.cpp (local)",
            font=Fonts.SMALL,
            fg=Colors.SUCCESS,
            bg=Colors.BG_TERTIARY,
        )
        self.llm_status.pack(side=LEFT, padx=15)

        Label(
            status,
            text="│",
            font=Fonts.SMALL,
            fg=Colors.TEXT_DIMMED,
            bg=Colors.BG_TERTIARY,
        ).pack(side=LEFT)

        self.storage_status = Label(
            status,
            text="Storage: ChromaDB",
            font=Fonts.SMALL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_TERTIARY,
        )
        self.storage_status.pack(side=LEFT, padx=15)

        # Doc count on right
        self.doc_count = Label(
            status,
            text="0 docs",
            font=Fonts.SMALL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_TERTIARY,
        )
        self.doc_count.pack(side=RIGHT, padx=15)

    def _create_views(self) -> None:
        """Create all views."""
        self.views[ViewType.DASHBOARD] = DashboardView(self.content, self)
        self.views[ViewType.DOCUMENTS] = DashboardView(
            self.content, self
        )  # Reuse for now
        self.views[ViewType.IMPORT] = ImportView(self.content, self)
        self.views[ViewType.PROCESSING] = ProcessingView(self.content, self)
        self.views[ViewType.QUERY] = QueryView(self.content, self)
        self.views[ViewType.ANALYZE] = AnalyzeView(self.content, self)
        self.views[ViewType.LITERARY] = LiteraryView(self.content, self)
        self.views[ViewType.STUDY] = StudyView(self.content, self)
        self.views[ViewType.EXPORT] = AnalyzeView(self.content, self)  # Placeholder
        self.views[ViewType.MONITOR] = MonitorView(self.content, self)
        self.views[ViewType.SETTINGS] = SettingsView(self.content, self)

    def show_view(self, view_type: ViewType) -> None:
        """Show specified view."""
        # Hide current view
        if self.current_view in self.views:
            self.views[self.current_view].on_hide()
            self.views[self.current_view].pack_forget()

        # Update nav buttons
        for vt, btn in self.nav_buttons.items():
            btn.set_active(vt == view_type)

        # Show new view
        self.current_view = view_type
        if view_type in self.views:
            self.views[view_type].pack(fill=BOTH, expand=True)
            self.views[view_type].on_show()

    def start_processing(self, files: List[Path], options: Dict) -> None:
        """Start document processing."""
        self.show_view(ViewType.PROCESSING)
        processing_view = self.views[ViewType.PROCESSING]
        processing_view.start_processing(files, options)

    def _init_backend(self) -> None:
        """Initialize backend components gracefully."""
        try:
            from ingestforge.core.config_loaders import load_config
            from ingestforge.core.pipeline.pipeline import Pipeline

            self._config = load_config()

            # Validate config has required attributes
            if not hasattr(self._config, "project") or not self._config.project:
                raise ValueError("Invalid configuration: missing project settings")

            self._pipeline = Pipeline(self._config)

            # Update status
            self.set_status("Project loaded")

            # Try to get LLM client
            self._init_llm_client()

        except FileNotFoundError:
            self.set_status("No project - run 'ingestforge init' first")
        except Exception as e:
            self.set_status(f"Project error: {str(e)[:50]}")
            import logging

            logging.debug(f"Backend initialization failed: {e}")

    def _init_llm_client(self) -> None:
        """Initialize LLM client - always prefer llama.cpp unless explicitly changed."""
        try:
            from ingestforge.llm.factory import get_llm_client

            # Always try llama.cpp first (local, no API costs)
            try:
                self._llm_client = get_llm_client(self._config, provider="llamacpp")
                self.llm_status.config(text="LLM: llama.cpp (local)", fg=Colors.SUCCESS)
                return
            except Exception as e:
                import logging

                logging.debug(f"llama.cpp not available: {e}")

            # Only use cloud provider if explicitly configured as default
            configured_provider = getattr(
                self._config.llm, "default_provider", "llamacpp"
            )
            if configured_provider in ("claude", "openai", "gemini"):
                # User explicitly configured cloud - ask for confirmation
                self._llm_client = None
                self.llm_status.config(
                    text=f"LLM: {configured_provider} (cloud - click Settings)",
                    fg=Colors.WARNING,
                )
            else:
                self._llm_client = None
                self.llm_status.config(
                    text="LLM: No local model found", fg=Colors.WARNING
                )

        except Exception as e:
            self.llm_status.config(text="LLM: Error", fg=Colors.ERROR)
            import logging

            logging.debug(f"LLM client initialization failed: {e}")

    def get_pipeline(self) -> Optional[Any]:
        """Get pipeline instance."""
        return self._pipeline

    def get_pipeline_status(self) -> Dict:
        """Get pipeline status."""
        if self._pipeline:
            return self._pipeline.get_status()
        return {}

    def get_llm_client(self) -> Optional[Any]:
        """Get LLM client."""
        return self._llm_client

    def set_status(self, text: str) -> None:
        """Set status bar text."""
        self.status_label.config(text=text)

    def run(self) -> None:
        """Start the application."""
        self.root.mainloop()


def main() -> None:
    """Launch the GUI application."""
    app = IngestForgeGUI()
    app.run()


if __name__ == "__main__":
    main()
