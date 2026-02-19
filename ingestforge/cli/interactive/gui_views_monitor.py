"""GUI Views - Live Engine Log Monitor.

Implements US-G10.1 (Live Engine Log Monitor).
Provides real-time log viewing in a TUI-style browser interface.

NASA JPL Rule #4: Functions limited to 60 lines.
NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

import logging
import queue
from datetime import datetime
from typing import Any, TYPE_CHECKING, Optional
from tkinter import (
    Frame,
    Label,
    Text,
    Scrollbar,
    Checkbutton,
    LEFT,
    RIGHT,
    BOTH,
    X,
    Y,
    END,
    WORD,
    VERTICAL,
    StringVar,
    BooleanVar,
)
from tkinter import ttk

from ingestforge.cli.interactive.gui_views_base import BaseView
from ingestforge.cli.interactive.gui_theme import Colors, Fonts
from ingestforge.cli.interactive.gui_widgets import StyledButton

if TYPE_CHECKING:
    from ingestforge.cli.interactive.gui_app import IngestForgeGUI


class GUILogHandler(logging.Handler):
    """Custom log handler that sends logs to a queue for GUI display."""

    def __init__(self, log_queue: queue.Queue) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the queue."""
        try:
            msg = self.format(record)
            self.log_queue.put((record.levelno, msg))
        except Exception:
            self.handleError(record)


class MonitorView(BaseView):
    """Live engine log monitor view."""

    # Max log lines to keep (Rule #2: Fixed upper bound)
    MAX_LOG_LINES = 1000

    def __init__(self, parent: Any, app: "IngestForgeGUI") -> None:
        super().__init__(parent, app)
        self.log_queue: queue.Queue = queue.Queue()
        self.log_handler: Optional[GUILogHandler] = None
        self.auto_scroll = True
        self.show_debug = False
        self._create_widgets()
        self._setup_logging()

    def _create_widgets(self) -> None:
        """Create all widgets."""
        self._create_header()
        self._create_toolbar()
        self._create_log_area()
        self._create_status_bar()

    def _create_header(self) -> None:
        """Create header with title."""
        header = Frame(self, bg=Colors.BG_PRIMARY)
        header.pack(fill=X, padx=20, pady=20)

        Label(
            header,
            text="Engine Log Monitor",
            font=Fonts.H1,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_PRIMARY,
        ).pack(side=LEFT)

        # Live indicator
        self.live_label = Label(
            header,
            text="LIVE",
            font=Fonts.BODY_BOLD,
            fg=Colors.ACCENT_SUCCESS,
            bg=Colors.BG_PRIMARY,
        )
        self.live_label.pack(side=LEFT, padx=(20, 0))

    def _create_toolbar(self) -> None:
        """Create toolbar with filter controls."""
        toolbar = Frame(self, bg=Colors.BG_SECONDARY)
        toolbar.pack(fill=X, padx=20, pady=(0, 10))

        # Log level filter
        Label(
            toolbar,
            text="Level:",
            font=Fonts.BODY,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT, padx=(10, 5), pady=10)

        self.level_var = StringVar(value="INFO")
        level_combo = ttk.Combobox(
            toolbar,
            textvariable=self.level_var,
            width=10,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
        )
        level_combo.pack(side=LEFT, padx=(0, 20))
        level_combo.bind("<<ComboboxSelected>>", self._on_level_change)

        # Auto-scroll checkbox
        self.auto_scroll_var = BooleanVar(value=True)
        Checkbutton(
            toolbar,
            text="Auto-scroll",
            variable=self.auto_scroll_var,
            font=Fonts.BODY,
            bg=Colors.BG_SECONDARY,
            fg=Colors.TEXT_PRIMARY,
            selectcolor=Colors.BG_TERTIARY,
            activebackground=Colors.BG_SECONDARY,
            command=self._on_auto_scroll_change,
        ).pack(side=LEFT, padx=(0, 20))

        # Clear button
        StyledButton(
            toolbar,
            text="Clear",
            style="secondary",
            command=self._clear_logs,
        ).pack(side=RIGHT, padx=10, pady=5)

        # Pause/Resume button
        self.paused = False
        self.pause_btn = StyledButton(
            toolbar,
            text="Pause",
            style="secondary",
            command=self._toggle_pause,
        )
        self.pause_btn.pack(side=RIGHT, padx=10, pady=5)

    def _create_log_area(self) -> None:
        """Create scrollable log text area."""
        log_frame = Frame(self, bg=Colors.BG_TERTIARY)
        log_frame.pack(fill=BOTH, expand=True, padx=20, pady=(0, 10))

        # Scrollbar
        scrollbar = Scrollbar(log_frame, orient=VERTICAL)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Log text widget with monospace font
        self.log_text = Text(
            log_frame,
            font=Fonts.MONO,
            bg="#1a1a2e",  # Dark terminal-like background
            fg="#e0e0e0",
            wrap=WORD,
            yscrollcommand=scrollbar.set,
            padx=10,
            pady=10,
            state="disabled",
        )
        self.log_text.pack(fill=BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)

        # Configure log level tags
        self.log_text.tag_configure("DEBUG", foreground="#888888")
        self.log_text.tag_configure("INFO", foreground="#00d4aa")
        self.log_text.tag_configure("WARNING", foreground="#ffa500")
        self.log_text.tag_configure("ERROR", foreground="#ff4444")
        self.log_text.tag_configure(
            "CRITICAL", foreground="#ff0000", font=Fonts.MONO_BOLD
        )
        self.log_text.tag_configure("timestamp", foreground="#666666")

    def _create_status_bar(self) -> None:
        """Create status bar."""
        status = Frame(self, bg=Colors.BG_TERTIARY, height=30)
        status.pack(fill=X, padx=20, pady=(0, 20))

        self.line_count_label = Label(
            status,
            text="Lines: 0",
            font=Fonts.SMALL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_TERTIARY,
        )
        self.line_count_label.pack(side=LEFT, padx=10)

        self.status_label = Label(
            status,
            text="Monitoring ingestforge logs",
            font=Fonts.SMALL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_TERTIARY,
        )
        self.status_label.pack(side=RIGHT, padx=10)

    def _setup_logging(self) -> None:
        """Set up log handler to capture ingestforge logs."""
        self.log_handler = GUILogHandler(self.log_queue)
        self.log_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
        self.log_handler.setFormatter(formatter)

        # Add handler to ingestforge root logger
        logger = logging.getLogger("ingestforge")
        logger.addHandler(self.log_handler)

        # Start polling for log messages
        self._poll_logs()

    def _poll_logs(self) -> None:
        """Poll log queue and update display."""
        if not self.paused:
            # Process all pending log messages (Rule #2: Fixed upper bound)
            max_process = 50
            processed = 0

            while processed < max_process:
                try:
                    level, msg = self.log_queue.get_nowait()
                    self._append_log(level, msg)
                    processed += 1
                except queue.Empty:
                    break

        # Schedule next poll
        self.after(100, self._poll_logs)

    def _append_log(self, level: int, msg: str) -> None:
        """Append a log message to the display."""
        if level < getattr(logging, self.level_var.get(), logging.INFO):
            return

        tag = self._get_log_tag(level)
        self._insert_log_text(msg, tag)
        self._trim_log_buffer()

        if self.auto_scroll_var.get():
            self.log_text.see(END)

        # Update line count status
        line_count = int(self.log_text.index("end-1c").split(".")[0]) - 1
        self.line_count_label.config(text=f"Lines: {line_count}")

    def _get_log_tag(self, level: int) -> str:
        """Map logging level to UI tag name."""
        if level >= logging.CRITICAL:
            return "CRITICAL"
        if level >= logging.ERROR:
            return "ERROR"
        if level >= logging.WARNING:
            return "WARNING"
        if level >= logging.INFO:
            return "INFO"
        return "DEBUG"

    def _insert_log_text(self, msg: str, tag: str) -> None:
        """Insert message into the log text widget."""
        self.log_text.config(state="normal")
        self.log_text.insert(END, msg + "\n", tag)
        self.log_text.config(state="disabled")

    def _trim_log_buffer(self) -> None:
        """Enforce Rule #2: Bound the log display buffer."""
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > self.MAX_LOG_LINES:
            self.log_text.config(state="normal")
            self.log_text.delete("1.0", f"{line_count - self.MAX_LOG_LINES}.0")
            self.log_text.config(state="disabled")

    def _on_level_change(self, event: Any) -> None:
        """Handle log level filter change."""
        pass  # Filter is applied during _append_log

    def _on_auto_scroll_change(self) -> None:
        """Handle auto-scroll toggle."""
        self.auto_scroll = self.auto_scroll_var.get()

    def _clear_logs(self) -> None:
        """Clear all logs from display."""
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", END)
        self.log_text.config(state="disabled")
        self.line_count_label.config(text="Lines: 0")

    def _toggle_pause(self) -> None:
        """Toggle pause/resume log capture."""
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")
        self.live_label.config(
            text="PAUSED" if self.paused else "LIVE",
            fg=Colors.TEXT_SECONDARY if self.paused else Colors.ACCENT_SUCCESS,
        )

    def on_hide(self) -> None:
        """Called when view is hidden."""
        pass

    def on_show(self) -> None:
        """Called when view is shown."""
        # Add a startup message
        self._append_log(
            logging.INFO,
            f"{datetime.now().strftime('%H:%M:%S')} [INFO] monitor: Log monitor active",
        )
