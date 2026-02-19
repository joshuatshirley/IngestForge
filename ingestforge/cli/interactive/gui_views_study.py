"""GUI Views - Study Tools (Glossary, Notes, Flashcards).

Implements US-G9.1 (Visual Glossary Generator) and US-G9.2 (AI Study Notes).
Extracted for modularity (JPL-004.1).

NASA JPL Rule #4: Functions limited to 60 lines.
NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

import threading
import subprocess
import sys
from typing import Any, TYPE_CHECKING, Dict
from tkinter import (
    Frame,
    Label,
    Text,
    Button,
    Scrollbar,
    LEFT,
    RIGHT,
    BOTH,
    X,
    Y,
    W,
    END,
    WORD,
    VERTICAL,
    StringVar,
    IntVar,
)
from tkinter import ttk

from ingestforge.cli.interactive.gui_views_base import BaseView
from ingestforge.cli.interactive.gui_theme import Colors, Fonts
from ingestforge.cli.interactive.gui_widgets import StyledButton, StyledEntry

if TYPE_CHECKING:
    from ingestforge.cli.interactive.gui_app import IngestForgeGUI


class StudyView(BaseView):
    """Study tools view with glossary, notes, and flashcards."""

    def __init__(self, parent: Any, app: "IngestForgeGUI") -> None:
        super().__init__(parent, app)
        self.current_tab = "glossary"
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create all widgets. Rule #4: Split into helper functions."""
        self._create_header()
        self._create_tab_bar()
        self._create_content_area()

    def _create_header(self) -> None:
        """Create header with title and topic input."""
        header = Frame(self, bg=Colors.BG_PRIMARY)
        header.pack(fill=X, padx=20, pady=20)

        Label(
            header,
            text="Study Tools",
            font=Fonts.H1,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_PRIMARY,
        ).pack(side=LEFT)

        # Topic input
        input_frame = Frame(header, bg=Colors.BG_PRIMARY)
        input_frame.pack(side=RIGHT)

        Label(
            input_frame,
            text="Topic:",
            font=Fonts.BODY,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_PRIMARY,
        ).pack(side=LEFT, padx=(0, 10))

        self.topic_var = StringVar(value="")
        self.topic_entry = StyledEntry(
            input_frame, textvariable=self.topic_var, width=30
        )
        self.topic_entry.pack(side=LEFT, padx=(0, 10))

        StyledButton(
            input_frame,
            text="Generate",
            style="primary",
            command=self._on_generate,
        ).pack(side=LEFT)

    def _create_tab_bar(self) -> None:
        """Create tab navigation bar."""
        tab_bar = Frame(self, bg=Colors.BG_SECONDARY)
        tab_bar.pack(fill=X, padx=20, pady=(0, 20))

        self.tabs: Dict[str, Button] = {}
        tab_configs = [
            ("glossary", "Glossary"),
            ("notes", "Study Notes"),
            ("flashcards", "Flashcards"),
            ("quiz", "Quiz"),
        ]

        for tab_id, tab_name in tab_configs:
            is_active = tab_id == self.current_tab
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
                command=lambda t=tab_id: self._switch_tab(t),
            )
            btn.pack(side=LEFT)
            self.tabs[tab_id] = btn

    def _create_content_area(self) -> None:
        """Create main content area."""
        self.content_frame = Frame(self, bg=Colors.BG_SECONDARY)
        self.content_frame.pack(fill=BOTH, expand=True, padx=20, pady=(0, 20))

        # Default: show glossary content
        self._show_glossary_content()

    def _switch_tab(self, tab_id: str) -> None:
        """Switch to specified tab."""
        self.current_tab = tab_id

        # Update tab button styles
        for tid, btn in self.tabs.items():
            is_active = tid == tab_id
            btn.configure(
                bg=Colors.BG_TERTIARY if is_active else Colors.BG_SECONDARY,
                fg=Colors.ACCENT_SECONDARY if is_active else Colors.TEXT_PRIMARY,
            )

        # Clear and repopulate content
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        if tab_id == "glossary":
            self._show_glossary_content()
        elif tab_id == "notes":
            self._show_notes_content()
        elif tab_id == "flashcards":
            self._show_flashcards_content()
        elif tab_id == "quiz":
            self._show_quiz_content()

    def _show_glossary_content(self) -> None:
        """Show glossary generator UI."""
        # Instructions
        Label(
            self.content_frame,
            text="Visual Glossary Generator",
            font=Fonts.H2,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=20, pady=(20, 10))

        Label(
            self.content_frame,
            text="Generate a glossary of key terms from your knowledge base.",
            font=Fonts.BODY,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=20, pady=(0, 20))

        # Options
        options_frame = Frame(self.content_frame, bg=Colors.BG_SECONDARY)
        options_frame.pack(fill=X, padx=20, pady=(0, 20))

        Label(
            options_frame,
            text="Max terms:",
            font=Fonts.BODY,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT)

        self.max_terms_var = IntVar(value=20)
        ttk.Spinbox(
            options_frame, from_=5, to=100, textvariable=self.max_terms_var, width=8
        ).pack(side=LEFT, padx=(5, 20))

        # Results area
        self._create_results_area()

    def _show_notes_content(self) -> None:
        """Show study notes generator UI."""
        Label(
            self.content_frame,
            text="AI Study Notes Generator",
            font=Fonts.H2,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=20, pady=(20, 10))

        Label(
            self.content_frame,
            text="Generate organized study notes from your knowledge base content.",
            font=Fonts.BODY,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=20, pady=(0, 20))

        # Style selector
        style_frame = Frame(self.content_frame, bg=Colors.BG_SECONDARY)
        style_frame.pack(fill=X, padx=20, pady=(0, 20))

        Label(
            style_frame,
            text="Note style:",
            font=Fonts.BODY,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT)

        self.note_style_var = StringVar(value="outline")
        style_combo = ttk.Combobox(
            style_frame,
            textvariable=self.note_style_var,
            width=15,
            values=["outline", "cornell", "bullet", "narrative"],
        )
        style_combo.pack(side=LEFT, padx=(5, 0))

        # Results area
        self._create_results_area()

    def _show_flashcards_content(self) -> None:
        """Show flashcards UI."""
        Label(
            self.content_frame,
            text="Flashcard Generator",
            font=Fonts.H2,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=20, pady=(20, 10))

        Label(
            self.content_frame,
            text="Generate flashcards for spaced repetition study.",
            font=Fonts.BODY,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=20, pady=(0, 20))

        self._create_results_area()

    def _show_quiz_content(self) -> None:
        """Show quiz UI."""
        Label(
            self.content_frame,
            text="Quiz Generator",
            font=Fonts.H2,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=20, pady=(20, 10))

        Label(
            self.content_frame,
            text="Generate quiz questions to test your knowledge.",
            font=Fonts.BODY,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=20, pady=(0, 20))

        self._create_results_area()

    def _create_results_area(self) -> None:
        """Create scrollable results text area."""
        results_frame = Frame(self.content_frame, bg=Colors.BG_TERTIARY)
        results_frame.pack(fill=BOTH, expand=True, padx=20, pady=(0, 20))

        # Scrollbar
        scrollbar = Scrollbar(results_frame, orient=VERTICAL)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Text widget
        self.results_text = Text(
            results_frame,
            font=Fonts.MONO,
            bg=Colors.BG_TERTIARY,
            fg=Colors.TEXT_PRIMARY,
            wrap=WORD,
            yscrollcommand=scrollbar.set,
            padx=15,
            pady=15,
        )
        self.results_text.pack(fill=BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)

        # Placeholder text
        self.results_text.insert(
            END, "Enter a topic and click 'Generate' to create content..."
        )
        self.results_text.config(state="disabled")

    def _on_generate(self) -> None:
        """Handle generate button click."""
        topic = self.topic_var.get().strip()
        if not topic:
            self._show_result("Please enter a topic to generate content for.")
            return

        # Show loading state
        self._show_result(f"Generating {self.current_tab} for '{topic}'...")

        # Run command in background thread
        thread = threading.Thread(target=self._run_generation, args=(topic,))
        thread.daemon = True
        thread.start()

    def _run_generation(self, topic: str) -> None:
        """Run study command in background."""
        cmd_map = {
            "glossary": ["study", "glossary", topic],
            "notes": ["study", "notes", topic],
            "flashcards": ["study", "flashcards", topic],
            "quiz": ["study", "quiz", topic],
        }

        cmd = cmd_map.get(self.current_tab, ["study", "glossary", topic])

        try:
            result = subprocess.run(
                [sys.executable, "-m", "ingestforge"] + cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout if result.returncode == 0 else result.stderr
            self._show_result(output or "No output generated.")
        except subprocess.TimeoutExpired:
            self._show_result("Generation timed out. Try a more specific topic.")
        except Exception as e:
            self._show_result(f"Error: {e}")

    def _show_result(self, text: str) -> None:
        """Display result in text area (thread-safe)."""

        def update():
            self.results_text.config(state="normal")
            self.results_text.delete("1.0", END)
            self.results_text.insert(END, text)
            self.results_text.config(state="disabled")

        self.after(0, update)
