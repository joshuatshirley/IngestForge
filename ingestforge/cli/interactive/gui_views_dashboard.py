"""GUI Views - Dashboard View.

Extracted from gui_menu.py for modularity (JPL-004.1).

NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from tkinter import Frame, Label, Button, LEFT, RIGHT, X, Y, W, BOTH

from ingestforge.cli.interactive.gui_views_base import BaseView
from ingestforge.cli.interactive.gui_theme import Colors, Fonts
from ingestforge.cli.interactive.gui_widgets import StatCard
from ingestforge.cli.interactive.gui_state import ViewType

if TYPE_CHECKING:
    from ingestforge.cli.interactive.gui_app import IngestForgeGUI


class DashboardView(BaseView):
    """Dashboard home view with stats and quick actions."""

    def __init__(self, parent: Any, app: IngestForgeGUI) -> None:
        super().__init__(parent, app)
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create all dashboard widgets."""
        self._create_header()
        self._create_stats_row()
        self._create_content_area()

    def _create_header(self) -> None:
        """Create dashboard header."""
        header = Frame(self, bg=Colors.BG_PRIMARY)
        header.pack(fill=X, padx=20, pady=20)

        Label(
            header,
            text="Dashboard",
            font=Fonts.H1,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_PRIMARY,
        ).pack(side=LEFT)

        Label(
            header,
            text="Last updated: just now",
            font=Fonts.SMALL,
            fg=Colors.TEXT_DIMMED,
            bg=Colors.BG_PRIMARY,
        ).pack(side=RIGHT)

    def _create_stats_row(self) -> None:
        """Create statistics cards row."""
        stats_frame = Frame(self, bg=Colors.BG_PRIMARY)
        stats_frame.pack(fill=X, padx=20, pady=(0, 20))

        self.doc_card = StatCard(stats_frame, "📄", "0", "Documents", "+0 this week")
        self.doc_card.pack(side=LEFT, fill=X, expand=True, padx=(0, 10))

        self.chunk_card = StatCard(stats_frame, "🔢", "0", "Chunks", "avg 0/doc")
        self.chunk_card.pack(side=LEFT, fill=X, expand=True, padx=5)

        self.storage_card = StatCard(stats_frame, "💾", "0 MB", "Storage", "ChromaDB")
        self.storage_card.pack(side=LEFT, fill=X, expand=True, padx=(10, 0))

    def _create_content_area(self) -> None:
        """Create main content area with activity and actions."""
        content = Frame(self, bg=Colors.BG_PRIMARY)
        content.pack(fill=BOTH, expand=True, padx=20)

        # Left column - Recent Activity
        left_col = Frame(content, bg=Colors.BG_PRIMARY)
        left_col.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        self._create_activity_panel(left_col)

        # Right column - Quick Actions + LLM Status
        right_col = Frame(content, bg=Colors.BG_PRIMARY)
        right_col.pack(side=RIGHT, fill=Y, padx=(10, 0))
        self._create_quick_actions(right_col)
        self._create_llm_status(right_col)

    def _create_activity_panel(self, parent: Frame) -> None:
        """Create recent activity panel."""
        panel = Frame(parent, bg=Colors.BG_SECONDARY)
        panel.pack(fill=BOTH, expand=True)

        # Header
        header = Frame(panel, bg=Colors.BG_SECONDARY)
        header.pack(fill=X, padx=15, pady=(15, 10))

        Label(
            header,
            text="Recent Activity",
            font=Fonts.H3,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT)

        # Activity list
        self.activity_frame = Frame(panel, bg=Colors.BG_SECONDARY)
        self.activity_frame.pack(fill=BOTH, expand=True, padx=15, pady=(0, 15))

        # Placeholder
        Label(
            self.activity_frame,
            text="No recent activity",
            font=Fonts.BODY,
            fg=Colors.TEXT_DIMMED,
            bg=Colors.BG_SECONDARY,
        ).pack(pady=30)

    def _create_quick_actions(self, parent: Frame) -> None:
        """Create quick actions panel."""
        panel = Frame(parent, bg=Colors.BG_SECONDARY, width=250)
        panel.pack(fill=X, pady=(0, 10))
        panel.pack_propagate(False)

        Label(
            panel,
            text="Quick Actions",
            font=Fonts.H3,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=15, pady=(15, 10))

        actions = [
            ("📁 Import Documents", lambda: self.app.show_view(ViewType.IMPORT)),
            ("❓ Ask a Question", lambda: self.app.show_view(ViewType.QUERY)),
            ("📊 Analyze Topics", lambda: self.app.show_view(ViewType.ANALYZE)),
            ("📝 Generate Quiz", lambda: self.app.show_view(ViewType.STUDY)),
        ]

        for text, cmd in actions:
            btn = Button(
                panel,
                text=text,
                command=cmd,
                font=Fonts.BODY,
                bg=Colors.BG_TERTIARY,
                fg=Colors.TEXT_PRIMARY,
                activebackground=Colors.ACCENT_TERTIARY,
                activeforeground=Colors.BG_PRIMARY,
                relief="flat",
                anchor="w",
                padx=10,
                pady=8,
                cursor="hand2",
            )
            btn.pack(fill=X, padx=15, pady=3)

    def _create_llm_status(self, parent: Frame) -> None:
        """Create LLM status panel."""
        panel = Frame(parent, bg=Colors.BG_SECONDARY, width=250)
        panel.pack(fill=X)
        panel.pack_propagate(False)

        Label(
            panel,
            text="LLM Status",
            font=Fonts.H3,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=15, pady=(15, 10))

        # Active provider
        self.llm_status_frame = Frame(panel, bg=Colors.BG_SECONDARY)
        self.llm_status_frame.pack(fill=X, padx=15, pady=(0, 15))

        # Default: no provider
        self._update_llm_display("llama.cpp", "Qwen2.5-14B-Instruct", True)

    def _update_llm_display(
        self, provider: str, model: str = "", active: bool = True
    ) -> None:
        """Update LLM status display."""
        for widget in self.llm_status_frame.winfo_children():
            widget.destroy()

        indicator = "●" if active else "○"
        color = Colors.SUCCESS if active else Colors.TEXT_DIMMED

        Label(
            self.llm_status_frame,
            text=f"{indicator} {provider}",
            font=Fonts.BODY,
            fg=color,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W)

        if model:
            Label(
                self.llm_status_frame,
                text=f"   {model}",
                font=Fonts.SMALL,
                fg=Colors.TEXT_SECONDARY,
                bg=Colors.BG_SECONDARY,
            ).pack(anchor=W)

    def on_show(self) -> None:
        """Refresh dashboard data."""
        self._refresh_stats()

    def _refresh_stats(self) -> None:
        """Refresh dashboard statistics."""
        try:
            status = self.app.get_pipeline_status()
            if status:
                doc_count = str(status.get("total_documents", 0))
                chunk_count = str(status.get("total_chunks", 0))
                self.doc_card.update_value(doc_count)
                self.chunk_card.update_value(chunk_count)
        except Exception as e:
            # Log error but don't disrupt UI
            import logging

            logging.debug(f"Could not refresh dashboard stats: {e}")
