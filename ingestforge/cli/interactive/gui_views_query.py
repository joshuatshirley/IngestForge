"""GUI Views - Query and Research Views.

Extracted from gui_menu.py for modularity (JPL-004.1).

NASA JPL Rule #4: Functions limited to 60 lines.
NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

from typing import Any, List, TYPE_CHECKING
from tkinter import (
    Frame,
    Label,
    Button,
    Text,
    Scrollbar,
    LEFT,
    RIGHT,
    BOTH,
    X,
    Y,
    W,
    END,
    WORD,
    DISABLED,
    NORMAL,
)

from ingestforge.cli.interactive.gui_views_base import BaseView
from ingestforge.cli.interactive.gui_theme import Colors, Fonts
from ingestforge.cli.interactive.gui_widgets import StyledButton, StyledEntry

if TYPE_CHECKING:
    from ingestforge.cli.interactive.gui_app import IngestForgeGUI


class QueryView(BaseView):
    """Query view with search and AI answers."""

    def __init__(self, parent: Any, app: "IngestForgeGUI") -> None:
        super().__init__(parent, app)
        self._create_widgets()

    def _create_search_bar(self, search_frame: Frame) -> None:
        """Create search bar with entry and button. Rule #4: Split helper."""
        search_inner = Frame(search_frame, bg=Colors.BG_SECONDARY)
        search_inner.pack(fill=X, padx=15, pady=15)

        self.search_entry = StyledEntry(
            search_inner, placeholder="Ask a question...", width=60
        )
        self.search_entry.pack(side=LEFT, fill=X, expand=True)
        self.search_entry.bind("<Return>", lambda e: self._execute_search())

        StyledButton(
            search_inner, text="Search", style="primary", command=self._execute_search
        ).pack(side=RIGHT, padx=(10, 0))

    def _create_search_options(self, search_frame: Frame) -> None:
        """Create search options panel. Rule #4: Split helper."""
        options = Frame(search_frame, bg=Colors.BG_SECONDARY)
        options.pack(fill=X, padx=15, pady=(0, 15))

        Label(
            options,
            text="Search Options:",
            font=Fonts.LABEL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT)

        self.search_type_var = StringVar(value="Hybrid")
        for st in ["Hybrid", "Semantic", "BM25"]:
            ttk.Radiobutton(
                options, text=st, variable=self.search_type_var, value=st
            ).pack(side=LEFT, padx=10)

        Label(
            options,
            text="Top K:",
            font=Fonts.LABEL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT, padx=(20, 5))

        self.topk_var = StringVar(value="5")
        ttk.Combobox(
            options,
            textvariable=self.topk_var,
            values=["3", "5", "10", "20"],
            width=5,
            state="readonly",
        ).pack(side=LEFT)

        self.rerank_var = BooleanVar(value=True)
        ttk.Checkbutton(options, text="Rerank", variable=self.rerank_var).pack(
            side=LEFT, padx=20
        )

    def _create_widgets(self) -> None:
        """Create query view widgets. Rule #4: <60 lines via helpers."""
        Label(
            self,
            text="Query Knowledge Base",
            font=Fonts.H1,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_PRIMARY,
        ).pack(anchor=W, padx=20, pady=20)

        # Search panel
        search_frame = Frame(self, bg=Colors.BG_SECONDARY)
        search_frame.pack(fill=X, padx=20, pady=(0, 20))
        self._create_search_bar(search_frame)
        self._create_search_options(search_frame)

        # Results pane
        results_pane = PanedWindow(
            self, orient=HORIZONTAL, bg=Colors.BG_PRIMARY, sashwidth=6
        )
        results_pane.pack(fill=BOTH, expand=True, padx=20, pady=(0, 20))
        self._create_answer_panel(results_pane)
        self._create_sources_panel(results_pane)

    def _create_answer_panel(self, parent: PanedWindow) -> None:
        """Create AI answer panel."""
        panel = Frame(parent, bg=Colors.BG_SECONDARY)
        parent.add(panel, stretch="always")

        Label(
            panel,
            text="AI Answer",
            font=Fonts.H3,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=15, pady=(15, 10))

        # Answer text
        answer_frame = Frame(panel, bg=Colors.BG_SECONDARY)
        answer_frame.pack(fill=BOTH, expand=True, padx=15, pady=(0, 15))

        scrollbar = Scrollbar(answer_frame)
        scrollbar.pack(side=RIGHT, fill=Y)

        self.answer_text = Text(
            answer_frame,
            font=Fonts.BODY,
            bg=Colors.BG_TERTIARY,
            fg=Colors.TEXT_PRIMARY,
            wrap=WORD,
            state=DISABLED,
            padx=10,
            pady=10,
            yscrollcommand=scrollbar.set,
        )
        self.answer_text.pack(fill=BOTH, expand=True)
        scrollbar.config(command=self.answer_text.yview)

        # Configure citation tags
        self.answer_text.tag_configure("citation", foreground=Colors.ACCENT_TERTIARY)

    def _create_sources_panel(self, parent: PanedWindow) -> None:
        """Create sources panel."""
        panel = Frame(parent, bg=Colors.BG_SECONDARY)
        parent.add(panel, stretch="always")

        Label(
            panel,
            text="Sources",
            font=Fonts.H3,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=15, pady=(15, 10))

        # Sources list
        self.sources_frame = Frame(panel, bg=Colors.BG_SECONDARY)
        self.sources_frame.pack(fill=BOTH, expand=True, padx=15, pady=(0, 15))

        # Placeholder
        Label(
            self.sources_frame,
            text="Search results will appear here",
            font=Fonts.BODY,
            fg=Colors.TEXT_DIMMED,
            bg=Colors.BG_SECONDARY,
        ).pack(pady=30)

    def _execute_search(self) -> None:
        """Execute search query."""
        query = self.search_entry.get_value()
        if not query:
            return

        # Clear previous results
        self._clear_results()

        # Show loading state
        self._set_answer("Searching...")

        # Execute search in thread
        thread = threading.Thread(
            target=self._search_thread,
            args=(query,),
            daemon=True,
        )
        thread.start()

    def _search_thread(self, query: str) -> None:
        """Search in background thread."""
        try:
            pipeline = self.app.get_pipeline()
            if not pipeline:
                self.after(
                    0,
                    lambda: self._set_answer(
                        "No pipeline initialized. Please initialize a project first."
                    ),
                )
                return

            # Get parameters
            top_k = int(self.topk_var.get())

            # Execute search
            results = pipeline.query(query, top_k=top_k)

            # Update UI
            self.after(0, lambda q=query, r=results: self._display_results(q, r))

        except Exception as e:
            error_msg = f"Search error: {e}"
            self.after(0, lambda msg=error_msg: self._set_answer(msg))

    def _display_results(self, query: str, results: List[Dict]) -> None:
        """Display search results."""
        if not results:
            self._set_answer(
                "No results found. Try rephrasing your question or ingesting more documents."
            )
            return

        # Display sources
        self._display_sources(results)

        # Generate answer using LLM
        self._generate_answer(query, results)

    def _create_source_card(self, parent: Frame, idx: int, result: Dict) -> None:
        """Create a single source result card. Rule #4: Split helper."""
        frame = Frame(parent, bg=Colors.BG_TERTIARY)
        frame.pack(fill=X, pady=5)

        # Header with source name and score
        header = Frame(frame, bg=Colors.BG_TERTIARY)
        header.pack(fill=X, padx=10, pady=(10, 5))
        Label(
            header,
            text=f"{idx}. {result.get('source_file', 'Unknown')}",
            font=Fonts.BODY_BOLD,
            fg=Colors.ACCENT_TERTIARY,
            bg=Colors.BG_TERTIARY,
        ).pack(side=LEFT)
        Label(
            header,
            text=f"Score: {result.get('score', 0):.3f}",
            font=Fonts.SMALL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_TERTIARY,
        ).pack(side=RIGHT)

        # Content preview
        content = result.get("content", result.get("text", ""))[:200] + "..."
        Label(
            frame,
            text=content,
            font=Fonts.SMALL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_TERTIARY,
            wraplength=300,
            justify=LEFT,
        ).pack(fill=X, padx=10, pady=(0, 10))

        # Action buttons
        btn_frame = Frame(frame, bg=Colors.BG_TERTIARY)
        btn_frame.pack(fill=X, padx=10, pady=(0, 10))
        Button(
            btn_frame,
            text="View",
            font=Fonts.SMALL,
            bg=Colors.BG_SECONDARY,
            fg=Colors.TEXT_SECONDARY,
            relief="flat",
            cursor="hand2",
        ).pack(side=LEFT, padx=(0, 5))
        Button(
            btn_frame,
            text="Copy",
            font=Fonts.SMALL,
            bg=Colors.BG_SECONDARY,
            fg=Colors.TEXT_SECONDARY,
            relief="flat",
            cursor="hand2",
        ).pack(side=LEFT)

    def _display_sources(self, results: List[Dict]) -> None:
        """Display source chunks. Rule #4: <60 lines via helper."""
        # Clear previous
        for widget in self.sources_frame.winfo_children():
            widget.destroy()

        # Create scrollable frame
        canvas = Canvas(
            self.sources_frame, bg=Colors.BG_SECONDARY, highlightthickness=0
        )
        scrollbar = Scrollbar(self.sources_frame, orient=VERTICAL, command=canvas.yview)
        scrollable = Frame(canvas, bg=Colors.BG_SECONDARY)

        canvas.create_window((0, 0), window=scrollable, anchor=NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Create source cards
        for i, result in enumerate(results, 1):
            self._create_source_card(scrollable, i, result)

        # Update scroll region
        scrollable.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _generate_answer(self, query: str, results: List[Dict]) -> None:
        """Generate AI answer."""
        try:
            llm_client = self.app.get_llm_client()
            if not llm_client:
                self._set_answer(
                    "No LLM available. Configure an LLM provider in Settings."
                )
                return

            # Build context
            context_parts = []
            for i, result in enumerate(results, 1):
                text = result.get("content", result.get("text", ""))
                source = result.get("source_file", f"Source {i}")
                context_parts.append(f"[{i}] {source}:\n{text}")

            context = "\n\n".join(context_parts)

            # Create prompt
            prompt = (
                f"Answer the following question based on the provided context. "
                f"Cite sources using [number] notation.\n\n"
                f"Question: {query}\n\n"
                f"Context:\n{context}\n\n"
                f"Answer:"
            )

            # Generate answer
            answer = llm_client.generate(prompt)

            # Display answer with tools used
            tools_info = (
                f"\n\n---\nTools Used:\n"
                f"- ChromaDB (vector search)\n"
                f"- {self.search_type_var.get()} retrieval\n"
                f"- {llm_client.__class__.__name__} (answer generation)"
            )

            self._set_answer(answer + tools_info)

        except Exception as e:
            self._set_answer(
                f"Answer generation failed: {e}\n\nResults are shown in the Sources panel."
            )

    def _set_answer(self, text: str) -> None:
        """Set answer text."""
        self.answer_text.config(state=NORMAL)
        self.answer_text.delete(1.0, END)
        self.answer_text.insert(END, text)
        self.answer_text.config(state=DISABLED)

    def _clear_results(self) -> None:
        """Clear all results."""
        self._set_answer("")
        for widget in self.sources_frame.winfo_children():
            widget.destroy()
