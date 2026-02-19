"""GUI Views - Analyze and Literary Views.

Extracted from gui_menu.py for modularity (JPL-004.1).

NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from tkinter import Frame, Label, X, W, BOTH

from ingestforge.cli.interactive.gui_views_base import BaseView
from ingestforge.cli.interactive.gui_theme import Colors, Fonts

if TYPE_CHECKING:
    from ingestforge.cli.interactive.gui_app import IngestForgeGUI


class AnalyzeView(BaseView):
    """Analysis view with topic/entity/similarity tools."""

    def __init__(self, parent: Any, app: "IngestForgeGUI") -> None:
        super().__init__(parent, app)
        self._create_widgets()

    def _create_widgets(self) -> None:
        Label(
            self,
            text="Analysis Tools",
            font=Fonts.H1,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_PRIMARY,
        ).pack(anchor=W, padx=20, pady=20)

        # Analysis options grid
        grid = Frame(self, bg=Colors.BG_PRIMARY)
        grid.pack(fill=BOTH, expand=True, padx=20)

        tools = [
            ("Topics", "📊", "Extract topics from documents using LDA/NMF"),
            ("Entities", "🏷️", "Named entity recognition and extraction"),
            ("Similarity", "🔗", "Find similar documents and clustering"),
            ("Knowledge Graph", "🕸️", "Visualize entity relationships"),
        ]

        for i, (name, icon, desc) in enumerate(tools):
            card = Frame(grid, bg=Colors.BG_SECONDARY)
            card.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky=NSEW)

            Label(
                card,
                text=f"{icon} {name}",
                font=Fonts.H2,
                fg=Colors.TEXT_PRIMARY,
                bg=Colors.BG_SECONDARY,
            ).pack(anchor=W, padx=20, pady=(20, 10))

            Label(
                card,
                text=desc,
                font=Fonts.BODY,
                fg=Colors.TEXT_SECONDARY,
                bg=Colors.BG_SECONDARY,
            ).pack(anchor=W, padx=20)

            StyledButton(
                card,
                text="Run Analysis",
                style="secondary",
            ).pack(anchor=W, padx=20, pady=20)

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)


class LiteraryView(BaseView):
    """Literary analysis view with themes/characters/symbols."""

    def __init__(self, parent: Any, app: "IngestForgeGUI") -> None:
        super().__init__(parent, app)
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create all widgets. Rule #4: Split into helper functions."""
        self._create_header()
        self._create_tab_bar()
        self._create_content_area()

    def _create_header(self) -> None:
        """Create header with work selector."""
        header = Frame(self, bg=Colors.BG_PRIMARY)
        header.pack(fill=X, padx=20, pady=20)

        Label(
            header,
            text="Literary Analysis",
            font=Fonts.H1,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_PRIMARY,
        ).pack(side=LEFT)

        Label(
            header,
            text="Document:",
            font=Fonts.BODY,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_PRIMARY,
        ).pack(side=LEFT, padx=(30, 5))

        self.work_var = StringVar(value="Select a work...")
        ttk.Combobox(
            header,
            textvariable=self.work_var,
            values=["Pride and Prejudice", "Hamlet", "1984", "Great Gatsby"],
            width=25,
        ).pack(side=LEFT)

    def _create_tab_bar(self) -> None:
        """Create tab navigation bar."""
        tab_bar = Frame(self, bg=Colors.BG_SECONDARY)
        tab_bar.pack(fill=X, padx=20, pady=(0, 20))

        self.tabs = {}
        for tab_name in ["Themes", "Characters", "Symbols", "Story Arc"]:
            is_active = tab_name == "Themes"
            btn = Button(
                tab_bar,
                text=tab_name,
                font=Fonts.BODY_BOLD,
                bg=Colors.BG_TERTIARY if is_active else Colors.BG_SECONDARY,
                fg=Colors.ACCENT_SECONDARY if is_active else Colors.TEXT_PRIMARY,
                relief="flat",
                padx=20,
                pady=10,
                cursor="hand2",
            )
            btn.pack(side=LEFT)
            self.tabs[tab_name] = btn

    def _create_content_area(self) -> None:
        """Create main content area with theme cards."""
        content = Frame(self, bg=Colors.BG_SECONDARY)
        content.pack(fill=BOTH, expand=True, padx=20, pady=(0, 20))

        Label(
            content,
            text="Major Themes",
            font=Fonts.H2,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=20, pady=(20, 15))

        # Sample theme cards
        themes = [
            ("Pride and Social Class", 87, 156, 23),
            ("Marriage and Economic Security", 72, 98, 18),
            ("First Impressions vs. Reality", 58, 67, 15),
        ]

        for name, prevalence, mentions, chapters in themes:
            self._create_theme_card(content, name, prevalence, mentions, chapters)

    def _create_theme_card(
        self, parent: Frame, name: str, prevalence: int, mentions: int, chapters: int
    ) -> None:
        """Create a theme card. Rule #4: Split into helpers."""
        card = Frame(parent, bg=Colors.BG_TERTIARY)
        card.pack(fill=X, padx=20, pady=5)

        self._create_theme_header(card, name, prevalence)
        self._create_theme_progress(card, prevalence)
        self._create_theme_details(card, mentions, chapters)
        self._create_theme_actions(card)

    def _create_theme_header(self, card: Frame, name: str, prevalence: int) -> None:
        """Create theme card header."""
        header = Frame(card, bg=Colors.BG_TERTIARY)
        header.pack(fill=X, padx=15, pady=(15, 5))

        Label(
            header,
            text=f"1. {name}",
            font=Fonts.BODY_BOLD,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_TERTIARY,
        ).pack(side=LEFT)

        Label(
            header,
            text=f"{prevalence}%",
            font=Fonts.BADGE,
            fg=Colors.ACCENT_SECONDARY,
            bg=Colors.BG_TERTIARY,
        ).pack(side=RIGHT)

    def _create_theme_progress(self, card: Frame, prevalence: int) -> None:
        """Create theme progress bar."""
        bar_frame = Frame(card, bg=Colors.BG_TERTIARY)
        bar_frame.pack(fill=X, padx=15, pady=5)

        bar = ProgressBar(bar_frame, height=6)
        bar.pack(fill=X)
        bar.set_progress(prevalence / 100)

    def _create_theme_details(self, card: Frame, mentions: int, chapters: int) -> None:
        """Create theme details text."""
        Label(
            card,
            text=f"Prevalence: {mentions} mentions across {chapters} chapters",
            font=Fonts.SMALL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_TERTIARY,
        ).pack(anchor=W, padx=15)

    def _create_theme_actions(self, card: Frame) -> None:
        """Create theme action buttons."""
        btn_frame = Frame(card, bg=Colors.BG_TERTIARY)
        btn_frame.pack(fill=X, padx=15, pady=(5, 15))

        Button(
            btn_frame,
            text="View Passages",
            font=Fonts.SMALL,
            bg=Colors.BG_SECONDARY,
            fg=Colors.TEXT_SECONDARY,
            relief="flat",
            cursor="hand2",
        ).pack(side=LEFT, padx=(0, 10))

        Button(
            btn_frame,
            text="Generate Analysis",
            font=Fonts.SMALL,
            bg=Colors.BG_SECONDARY,
            fg=Colors.TEXT_SECONDARY,
            relief="flat",
            cursor="hand2",
        ).pack(side=LEFT)
