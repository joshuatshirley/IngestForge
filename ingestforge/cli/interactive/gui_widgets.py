"""GUI Widgets - Reusable UI Components.

Custom Tkinter widgets for IngestForge GUI application.
Extracted from gui_menu.py for modularity (JPL-004).

NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from tkinter import (
    Button,
    Canvas,
    Entry,
    Frame,
    Label,
    Text,
    Scrollbar,
    Toplevel,
    Event,
    END,
    WORD,
    DISABLED,
    NORMAL,
    LEFT,
    RIGHT,
    BOTH,
    X,
    Y,
    W,
)

from ingestforge.cli.interactive.gui_theme import Colors, Fonts


class StyledButton(Button):
    """Styled button with hover effects."""

    def __init__(
        self,
        parent: Any,
        text: str,
        command: Optional[Callable[[], None]] = None,
        style: str = "primary",
        width: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # Determine colors based on style
        style_map = {
            "primary": (Colors.BUTTON_PRIMARY, Colors.BUTTON_TEXT, "#3eb893"),
            "secondary": (Colors.BUTTON_SECONDARY, Colors.TEXT_PRIMARY, "#1a4a7a"),
            "danger": (Colors.BUTTON_DANGER, Colors.BUTTON_TEXT, "#c73b52"),
        }
        bg, fg, hover_bg = style_map.get(
            style, (Colors.BG_TERTIARY, Colors.TEXT_PRIMARY, "#1a4a7a")
        )

        btn_kwargs = {
            "text": text,
            "font": Fonts.BODY_BOLD,
            "bg": bg,
            "fg": fg,
            "activebackground": hover_bg,
            "activeforeground": fg,
            "relief": "flat",
            "cursor": "hand2",
            "padx": 16,
            "pady": 8,
            **kwargs,
        }
        if command is not None:
            btn_kwargs["command"] = command

        super().__init__(parent, **btn_kwargs)

        if width:
            self.config(width=width)

        self._bg = bg
        self._hover_bg = hover_bg

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, e: "Event[Any]") -> None:
        """Handle mouse enter event."""
        self.config(bg=self._hover_bg)

    def _on_leave(self, e: "Event[Any]") -> None:
        """Handle mouse leave event."""
        self.config(bg=self._bg)


class StyledEntry(Entry):
    """Styled entry field with placeholder."""

    def __init__(self, parent: Any, placeholder: str = "", **kwargs: Any) -> None:
        super().__init__(
            parent,
            font=Fonts.BODY,
            bg=Colors.BG_TERTIARY,
            fg=Colors.TEXT_PRIMARY,
            insertbackground=Colors.TEXT_PRIMARY,
            relief="flat",
            **kwargs,
        )

        self._placeholder = placeholder
        self._placeholder_color = Colors.TEXT_DIMMED
        self._normal_color = Colors.TEXT_PRIMARY

        if placeholder:
            self._show_placeholder()
            self.bind("<FocusIn>", self._on_focus_in)
            self.bind("<FocusOut>", self._on_focus_out)

    def _show_placeholder(self) -> None:
        """Display placeholder text."""
        self.delete(0, END)
        self.insert(0, self._placeholder)
        self.config(fg=self._placeholder_color)

    def _on_focus_in(self, e: "Event[Any]") -> None:
        """Handle focus in event."""
        if self.get() == self._placeholder:
            self.delete(0, END)
            self.config(fg=self._normal_color)

    def _on_focus_out(self, e: "Event[Any]") -> None:
        """Handle focus out event."""
        if not self.get():
            self._show_placeholder()

    def get_value(self) -> str:
        """Get value, excluding placeholder."""
        val = self.get()
        return "" if val == self._placeholder else val


class StatCard(Frame):
    """Statistics display card."""

    def __init__(
        self,
        parent: Any,
        icon: str,
        value: str,
        label: str,
        detail: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, bg=Colors.BG_SECONDARY, **kwargs)

        # Icon and value
        top = Frame(self, bg=Colors.BG_SECONDARY)
        top.pack(fill=X, padx=15, pady=(15, 5))

        Label(
            top,
            text=icon,
            font=("Segoe UI", 20),
            fg=Colors.ACCENT_TERTIARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT)

        self.value_label = Label(
            top,
            text=value,
            font=Fonts.H2,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        )
        self.value_label.pack(side=RIGHT)

        # Label
        Label(
            self,
            text=label,
            font=Fonts.LABEL,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_SECONDARY,
        ).pack(fill=X, padx=15)

        # Detail
        if detail:
            Label(
                self,
                text=detail,
                font=Fonts.SMALL,
                fg=Colors.TEXT_DIMMED,
                bg=Colors.BG_SECONDARY,
            ).pack(fill=X, padx=15, pady=(0, 10))

    def update_value(self, value: str) -> None:
        """Update the displayed value."""
        self.value_label.config(text=value)


class ProgressBar(Canvas):
    """Custom progress bar with gradient."""

    def __init__(self, parent: Any, height: int = 8, **kwargs: Any) -> None:
        super().__init__(
            parent, height=height, bg=Colors.BG_TERTIARY, highlightthickness=0, **kwargs
        )
        self._height = height
        self._progress = 0.0

        self.bind("<Configure>", self._draw)

    def set_progress(self, value: float) -> None:
        """Set progress (0.0 to 1.0)."""
        self._progress = max(0.0, min(1.0, value))
        self._draw()

    def _draw(self, event: "Optional[Event[Any]]" = None) -> None:
        self.delete("all")
        width = self.winfo_width()

        if width <= 0:
            return

        # Background
        self.create_rectangle(
            0, 0, width, self._height, fill=Colors.BG_TERTIARY, outline=""
        )

        # Progress fill
        fill_width = int(width * self._progress)
        if fill_width > 0:
            self.create_rectangle(
                0, 0, fill_width, self._height, fill=Colors.ACCENT_SECONDARY, outline=""
            )


class ToolBadge(Label):
    """Tool usage badge."""

    def __init__(
        self, parent: Any, text: str, active: bool = True, **kwargs: Any
    ) -> None:
        color = Colors.ACCENT_TERTIARY if active else Colors.TEXT_DIMMED
        super().__init__(
            parent,
            text=text,
            font=Fonts.CODE,
            fg=color,
            bg=Colors.BG_SECONDARY,
            padx=8,
            pady=2,
            **kwargs,
        )


class NavButton(Button):
    """Navigation sidebar button."""

    def __init__(
        self,
        parent: Any,
        text: str,
        icon: str = "",
        command: Optional[Callable[[], Any]] = None,
        active: bool = False,
        **kwargs: Any,
    ) -> None:
        bg = Colors.BG_TERTIARY if active else Colors.BG_SECONDARY
        fg = Colors.ACCENT_SECONDARY if active else Colors.TEXT_PRIMARY

        display_text = f"{icon}  {text}" if icon else text

        nav_kwargs = {
            "text": display_text,
            "font": Fonts.BODY,
            "bg": bg,
            "fg": fg,
            "activebackground": Colors.BG_TERTIARY,
            "activeforeground": Colors.ACCENT_SECONDARY,
            "relief": "flat",
            "anchor": "w",
            "padx": 15,
            "pady": 10,
            "cursor": "hand2",
            **kwargs,
        }
        if command is not None:
            nav_kwargs["command"] = command

        super().__init__(parent, **nav_kwargs)

        self._active = active
        self._normal_bg = Colors.BG_SECONDARY
        self._hover_bg = Colors.BG_TERTIARY
        self._active_bg = Colors.BG_TERTIARY

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self.config(bg=Colors.BG_TERTIARY, fg=Colors.ACCENT_SECONDARY)
        else:
            self.config(bg=Colors.BG_SECONDARY, fg=Colors.TEXT_PRIMARY)

    def _on_enter(self, e: "Event[Any]") -> None:
        if not self._active:
            self.config(bg=self._hover_bg)

    def _on_leave(self, e: "Event[Any]") -> None:
        if not self._active:
            self.config(bg=self._normal_bg)


class LogViewer(Frame):
    """Real-time log viewer widget with bounded buffer."""

    MAX_LOG_ENTRIES = 1000  # Prevent unbounded memory growth

    def __init__(self, parent: Any, height: int = 10, **kwargs: Any) -> None:
        super().__init__(parent, bg=Colors.BG_SECONDARY, **kwargs)

        # Header
        header = Frame(self, bg=Colors.BG_SECONDARY)
        header.pack(fill=X, padx=10, pady=(10, 5))

        Label(
            header,
            text="Live Log",
            font=Fonts.BODY_BOLD,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(side=LEFT)

        Button(
            header,
            text="Show Full",
            font=Fonts.SMALL,
            bg=Colors.BG_TERTIARY,
            fg=Colors.TEXT_SECONDARY,
            relief="flat",
            cursor="hand2",
            command=self._show_full_log,
        ).pack(side=RIGHT)

        # Log text area
        log_frame = Frame(self, bg=Colors.BG_TERTIARY)
        log_frame.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

        scrollbar = Scrollbar(log_frame)
        scrollbar.pack(side=RIGHT, fill=Y)

        self.log_text = Text(
            log_frame,
            font=Fonts.CODE,
            bg=Colors.BG_TERTIARY,
            fg=Colors.TEXT_SECONDARY,
            height=height,
            wrap=WORD,
            state=DISABLED,
            yscrollcommand=scrollbar.set,
        )
        self.log_text.pack(fill=BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)

        # Configure tags
        self.log_text.tag_configure("timestamp", foreground=Colors.TEXT_DIMMED)
        self.log_text.tag_configure("info", foreground=Colors.INFO)
        self.log_text.tag_configure("success", foreground=Colors.SUCCESS)
        self.log_text.tag_configure("warning", foreground=Colors.WARNING)
        self.log_text.tag_configure("error", foreground=Colors.ERROR)

        # Bounded log buffer (circular)
        self._full_log: List[Tuple[str, str, str]] = []

    def log(self, message: str, level: str = "info") -> None:
        """Add log entry with bounded buffer."""
        timestamp = datetime.now().strftime("[%H:%M:%S]")

        # Enforce max log entries (circular buffer behavior)
        if len(self._full_log) >= self.MAX_LOG_ENTRIES:
            self._full_log.pop(0)  # Remove oldest entry

        self._full_log.append((timestamp, message, level))

        self.log_text.config(state=NORMAL)
        self.log_text.insert(END, f"{timestamp} ", "timestamp")
        self.log_text.insert(END, f"{message}\n", level)
        self.log_text.see(END)
        self.log_text.config(state=DISABLED)

    def clear(self) -> None:
        """Clear log display."""
        self.log_text.config(state=NORMAL)
        self.log_text.delete(1.0, END)
        self.log_text.config(state=DISABLED)
        self._full_log.clear()

    def _show_full_log(self) -> None:
        """Show full log in popup."""
        popup = Toplevel(self)
        popup.title("Full Log")
        popup.geometry("800x600")
        popup.configure(bg=Colors.BG_PRIMARY)

        text = Text(
            popup,
            font=Fonts.CODE,
            bg=Colors.BG_PRIMARY,
            fg=Colors.TEXT_PRIMARY,
            wrap=WORD,
        )
        text.pack(fill=BOTH, expand=True, padx=10, pady=10)

        for timestamp, message, level in self._full_log:
            text.insert(END, f"{timestamp} {message}\n")


class PipelineVisualizer(Frame):
    """Visual pipeline stage indicator."""

    STAGES = [
        ("Extract", ""),
        ("Chunk", ""),
        ("Enrich", ""),
        ("Embed", ""),
        ("Store", ""),
    ]

    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(parent, bg=Colors.BG_SECONDARY, **kwargs)

        Label(
            self,
            text="Pipeline Stage",
            font=Fonts.BODY_BOLD,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=15, pady=(15, 10))

        # Stage indicators
        stages_frame = Frame(self, bg=Colors.BG_SECONDARY)
        stages_frame.pack(fill=X, padx=15, pady=(0, 15))

        self.stage_labels: Dict[str, Label] = {}
        self.stage_indicators: Dict[str, Label] = {}

        for i, (name, icon) in enumerate(self.STAGES):
            stage_frame = Frame(stages_frame, bg=Colors.BG_SECONDARY)
            stage_frame.pack(side=LEFT, padx=5)

            # Stage box
            indicator = Label(
                stage_frame,
                text=name[:3],
                font=Fonts.BADGE,
                fg=Colors.TEXT_DIMMED,
                bg=Colors.BG_TERTIARY,
                width=6,
                height=2,
            )
            indicator.pack()
            self.stage_indicators[name] = indicator

            # Arrow between stages
            if i < len(self.STAGES) - 1:
                Label(
                    stages_frame,
                    text="→",
                    font=Fonts.BODY,
                    fg=Colors.TEXT_DIMMED,
                    bg=Colors.BG_SECONDARY,
                ).pack(side=LEFT, padx=2)

    def set_stage(self, stage_name: str, status: str = "active") -> None:
        """Set stage status. Rule #1: Dictionary dispatch eliminates nesting."""
        status_styles = {
            "active": (Colors.TEXT_PRIMARY, Colors.ACCENT_TERTIARY),
            "complete": (Colors.SUCCESS, Colors.BG_TERTIARY),
            "error": (Colors.ERROR, Colors.BG_TERTIARY),
        }
        default_style = (Colors.TEXT_DIMMED, Colors.BG_TERTIARY)

        indicator = self.stage_indicators.get(stage_name)
        if indicator:
            fg, bg = status_styles.get(status, default_style)
            indicator.config(fg=fg, bg=bg)

    def reset(self) -> None:
        """Reset all stages to pending."""
        for indicator in self.stage_indicators.values():
            indicator.config(fg=Colors.TEXT_DIMMED, bg=Colors.BG_TERTIARY)
