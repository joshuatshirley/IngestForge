"""GUI Views - Import View.

Extracted from gui_menu.py for modularity (JPL-004.1).

NASA JPL Rule #4: Functions limited to 60 lines.
NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, TYPE_CHECKING
from tkinter import (
    Frame,
    Label,
    Button,
    Entry,
    Event,
    StringVar,
    BooleanVar,
    LEFT,
    RIGHT,
    X,
    W,
    filedialog,
    messagebox,
    ttk,
)

from ingestforge.cli.interactive.gui_views_base import BaseView
from ingestforge.cli.interactive.gui_theme import Colors, Fonts
from ingestforge.cli.interactive.gui_widgets import StyledButton
from ingestforge.cli.interactive.gui_state import ViewType, DocumentInfo

if TYPE_CHECKING:
    from ingestforge.cli.interactive.gui_app import IngestForgeGUI


class ImportView(BaseView):
    """Document import view with drag-and-drop."""

    def __init__(self, parent: Any, app: IngestForgeGUI) -> None:
        super().__init__(parent, app)
        self.import_queue: List[DocumentInfo] = []
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create all import view widgets."""
        Label(
            self,
            text="Import Documents",
            font=Fonts.H1,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_PRIMARY,
        ).pack(anchor=W, padx=20, pady=20)

        self._create_drop_zone()
        self._create_queue_panel()
        self._create_options_panel()
        self._create_action_buttons()

    def _create_drop_zone(self) -> None:
        """Create drag-and-drop file zone."""
        self.drop_frame = Frame(
            self,
            bg=Colors.BG_SECONDARY,
            highlightbackground=Colors.ACCENT_TERTIARY,
            highlightthickness=2,
        )
        self.drop_frame.pack(fill=X, padx=20, pady=(0, 20))

        inner = Frame(self.drop_frame, bg=Colors.BG_SECONDARY)
        inner.pack(fill=X, pady=40)

        labels_data = [
            ("📁", ("Segoe UI", 48), Colors.ACCENT_TERTIARY, (0, 0, 0, 0)),
            ("DROP FILES HERE", Fonts.H2, Colors.TEXT_PRIMARY, (10, 5, 0, 0)),
            ("or click to browse", Fonts.BODY, Colors.TEXT_SECONDARY, (0, 0, 0, 0)),
            (
                "Supported: PDF, DOCX, TXT, MD, HTML, EPUB, LaTeX, Jupyter",
                Fonts.SMALL,
                Colors.TEXT_DIMMED,
                (15, 0, 0, 0),
            ),
        ]

        for text, font, fg, pady in labels_data:
            Label(inner, text=text, font=font, fg=fg, bg=Colors.BG_SECONDARY).pack(
                pady=pady
            )

        # Bind click to open file dialog
        self._bind_browse_click(self.drop_frame)

    def _bind_browse_click(self, widget: Any) -> None:
        """Recursively bind browse click to widget and children."""
        widget.bind("<Button-1>", self._on_browse_click)
        for child in widget.winfo_children():
            self._bind_browse_click(child)

    def _create_queue_panel(self) -> None:
        """Create import queue panel."""
        panel = Frame(self, bg=Colors.BG_SECONDARY)
        panel.pack(fill=X, padx=20, pady=(0, 20))

        header = Frame(panel, bg=Colors.BG_SECONDARY)
        header.pack(fill=X, padx=15, pady=(15, 10))

        self.queue_label = Label(
            header,
            text="Import Queue (0 files)",
            font=Fonts.H3,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        )
        self.queue_label.pack(side=LEFT)

        Button(
            header,
            text="Clear All",
            font=Fonts.SMALL,
            bg=Colors.BG_TERTIARY,
            fg=Colors.TEXT_SECONDARY,
            relief="flat",
            cursor="hand2",
            command=self._clear_queue,
        ).pack(side=RIGHT)

        # Queue list frame
        self.queue_frame = Frame(panel, bg=Colors.BG_SECONDARY)
        self.queue_frame.pack(fill=X, padx=15, pady=(0, 15))

        # Empty state
        self.empty_label = Label(
            self.queue_frame,
            text="No files in queue",
            font=Fonts.BODY,
            fg=Colors.TEXT_DIMMED,
            bg=Colors.BG_SECONDARY,
        )
        self.empty_label.pack(pady=10)

    def _create_options_panel(self) -> None:
        """Create processing options panel."""
        panel = Frame(self, bg=Colors.BG_SECONDARY)
        panel.pack(fill=X, padx=20, pady=(0, 20))

        Label(
            panel,
            text="Processing Options",
            font=Fonts.H3,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=15, pady=(15, 10))

        options_frame = Frame(panel, bg=Colors.BG_SECONDARY)
        options_frame.pack(fill=X, padx=15, pady=(0, 15))

        # Left options (chunking)
        left = Frame(options_frame, bg=Colors.BG_SECONDARY)
        left.pack(side=LEFT, fill=X, expand=True)
        self._create_chunking_options(left)

        # Right options (OCR)
        right = Frame(options_frame, bg=Colors.BG_SECONDARY)
        right.pack(side=RIGHT)
        self._create_ocr_options(right)

    def _create_chunking_options(self, parent: Frame) -> None:
        """Create chunking options (left side)."""
        Label(
            parent,
            text="Chunking:",
            font=Fonts.LABEL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT)

        self.chunking_var = StringVar(value="Semantic")
        ttk.Combobox(
            parent,
            textvariable=self.chunking_var,
            values=["Semantic", "Fixed", "Paragraph", "Legal", "Code"],
            width=12,
            state="readonly",
        ).pack(side=LEFT, padx=(5, 20))

        Label(
            parent,
            text="Overlap:",
            font=Fonts.LABEL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT)

        self.overlap_var = StringVar(value="50")
        Entry(
            parent,
            textvariable=self.overlap_var,
            width=6,
            font=Fonts.BODY,
            bg=Colors.BG_TERTIARY,
            fg=Colors.TEXT_PRIMARY,
            relief="flat",
        ).pack(side=LEFT, padx=5)

        Label(
            parent,
            text="words",
            font=Fonts.LABEL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT)

    def _create_ocr_options(self, parent: Frame) -> None:
        """Create OCR options (right side)."""
        self.ocr_var = BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Enable OCR", variable=self.ocr_var).pack(
            side=LEFT, padx=(0, 20)
        )

        Label(
            parent,
            text="OCR Engine:",
            font=Fonts.LABEL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT)

        self.ocr_engine_var = StringVar(value="Tesseract")
        ttk.Combobox(
            parent,
            textvariable=self.ocr_engine_var,
            values=["Tesseract", "EasyOCR"],
            width=10,
            state="readonly",
        ).pack(side=LEFT, padx=5)

    def _create_action_buttons(self) -> None:
        """Create action buttons."""
        btn_frame = Frame(self, bg=Colors.BG_PRIMARY)
        btn_frame.pack(fill=X, padx=20, pady=10)

        StyledButton(
            btn_frame,
            text="▶ Start Processing",
            style="primary",
            command=self._start_processing,
        ).pack(side=RIGHT)

        StyledButton(
            btn_frame,
            text="Cancel",
            style="secondary",
            command=lambda: self.app.show_view(ViewType.DASHBOARD),
        ).pack(side=RIGHT, padx=10)

    def _on_browse_click(self, event: Event) -> None:
        """Handle browse click."""
        paths = filedialog.askopenfilenames(
            title="Select Documents",
            filetypes=[
                (
                    "All Supported",
                    "*.pdf *.docx *.txt *.md *.html *.epub *.tex *.ipynb",
                ),
                ("PDF", "*.pdf"),
                ("Word", "*.docx"),
                ("Text", "*.txt *.md"),
                ("HTML", "*.html *.htm"),
                ("EPUB", "*.epub"),
                ("LaTeX", "*.tex *.latex"),
                ("Jupyter", "*.ipynb"),
            ],
        )

        for path in paths:
            self._add_file_to_queue(Path(path))

    def _add_file_to_queue(self, path: Path) -> None:
        """Add file to import queue."""
        if not path.exists():
            return

        # Determine format
        format_map = {
            ".pdf": "PDF",
            ".docx": "DOCX",
            ".txt": "Text",
            ".md": "Markdown",
            ".html": "HTML",
            ".htm": "HTML",
            ".epub": "EPUB",
            ".tex": "LaTeX",
            ".latex": "LaTeX",
            ".ipynb": "Jupyter",
        }

        doc_info = DocumentInfo(
            path=path,
            name=path.name,
            size_bytes=path.stat().st_size,
            format_type=format_map.get(path.suffix.lower(), "Unknown"),
        )

        self.import_queue.append(doc_info)
        self._refresh_queue_display()

    def _refresh_queue_display(self) -> None:
        """Refresh queue display."""
        for widget in self.queue_frame.winfo_children():
            widget.destroy()

        count = len(self.import_queue)
        self.queue_label.config(text=f"Import Queue ({count} files)")

        if not self.import_queue:
            Label(
                self.queue_frame,
                text="No files in queue",
                font=Fonts.BODY,
                fg=Colors.TEXT_DIMMED,
                bg=Colors.BG_SECONDARY,
            ).pack(pady=10)
            return

        for doc in self.import_queue:
            self._create_queue_item(doc)

    def _create_queue_item(self, doc: DocumentInfo) -> None:
        """Create a single queue item row."""
        row = Frame(self.queue_frame, bg=Colors.BG_SECONDARY)
        row.pack(fill=X, pady=2)

        Label(
            row,
            text="📄",
            font=Fonts.BODY,
            fg=Colors.ACCENT_TERTIARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT)

        Label(
            row,
            text=doc.name,
            font=Fonts.BODY,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT, padx=5)

        Label(
            row,
            text=self._format_size(doc.size_bytes),
            font=Fonts.SMALL,
            fg=Colors.TEXT_DIMMED,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT, padx=5)

        Label(
            row,
            text=doc.format_type,
            font=Fonts.BADGE,
            fg=Colors.ACCENT_TERTIARY,
            bg=Colors.BG_TERTIARY,
            padx=5,
            pady=1,
        ).pack(side=LEFT, padx=5)

        Button(
            row,
            text="×",
            font=Fonts.BODY,
            fg=Colors.TEXT_DIMMED,
            bg=Colors.BG_SECONDARY,
            relief="flat",
            cursor="hand2",
            command=lambda: self._remove_from_queue(doc),
        ).pack(side=RIGHT)

    def _format_size(self, size_bytes: int) -> str:
        """Format file size."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"

    def _remove_from_queue(self, doc: DocumentInfo) -> None:
        """Remove file from queue."""
        self.import_queue.remove(doc)
        self._refresh_queue_display()

    def _clear_queue(self) -> None:
        """Clear import queue."""
        self.import_queue.clear()
        self._refresh_queue_display()

    def _start_processing(self) -> None:
        """Start processing queued documents."""
        if not self.import_queue:
            messagebox.showwarning("No Files", "Add files to the queue first.")
            return

        # Validate overlap value
        try:
            overlap_value = int(self.overlap_var.get())
            if not 0 <= overlap_value <= 500:
                raise ValueError("Overlap must be between 0 and 500")
        except ValueError as e:
            messagebox.showerror(
                "Invalid Input", f"Overlap must be a valid number (0-500): {e}"
            )
            return

        # Get options with validated values
        options = {
            "chunking": self.chunking_var.get().lower(),
            "overlap": overlap_value,
            "ocr_enabled": self.ocr_var.get(),
            "ocr_engine": self.ocr_engine_var.get().lower(),
        }

        # Store files and switch to processing view
        files = [doc.path for doc in self.import_queue]
        self.app.start_processing(files, options)
