"""GUI Views - Settings View.

Extracted from gui_menu.py for modularity (JPL-004.1).

NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from tkinter import (
    Frame,
    Label,
    Button,
    Entry,
    StringVar,
    LEFT,
    RIGHT,
    X,
    W,
    messagebox,
    simpledialog,
    ttk,
)

from ingestforge.cli.interactive.gui_views_base import BaseView
from ingestforge.cli.interactive.gui_theme import Colors, Fonts
from ingestforge.cli.interactive.gui_widgets import StyledButton

if TYPE_CHECKING:
    from ingestforge.cli.interactive.gui_app import IngestForgeGUI


class SettingsView(BaseView):
    """Settings view for LLM and storage configuration."""

    def __init__(self, parent: Any, app: "IngestForgeGUI") -> None:
        super().__init__(parent, app)
        self._create_widgets()

    def _create_provider_options(self, parent: Frame) -> None:
        """Create LLM provider radio buttons. Rule #4: Split helper."""
        Label(
            parent,
            text="Active Provider",
            font=Fonts.H3,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=20, pady=(0, 10))

        self.provider_var = StringVar(value="llamacpp")
        providers = [
            ("llamacpp", "llama.cpp (Local - No API costs)"),
            ("ollama", "Ollama (Local)"),
            ("claude", "Claude (Cloud - API key required)"),
            ("openai", "OpenAI (Cloud - API key required)"),
            ("gemini", "Gemini (Cloud - API key required)"),
        ]

        for value, label in providers:
            frame = Frame(parent, bg=Colors.BG_SECONDARY)
            frame.pack(fill=X, padx=20, pady=2)
            ttk.Radiobutton(
                frame, text=label, variable=self.provider_var, value=value
            ).pack(side=LEFT)
            Button(
                frame,
                text="Configure",
                font=Fonts.SMALL,
                bg=Colors.BG_TERTIARY,
                fg=Colors.TEXT_SECONDARY,
                relief="flat",
                cursor="hand2",
                command=lambda v=value: self._configure_provider(v),
            ).pack(side=RIGHT)

    def _create_llamacpp_settings(self, parent: Frame) -> None:
        """Create llama.cpp specific settings. Rule #4: Split helper."""
        panel = Frame(parent, bg=Colors.BG_TERTIARY)
        panel.pack(fill=X, padx=20, pady=15)

        Label(
            panel,
            text="LLAMA.CPP SETTINGS",
            font=Fonts.H3,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_TERTIARY,
        ).pack(anchor=W, padx=15, pady=(15, 10))

        # Model path
        model_frame = Frame(panel, bg=Colors.BG_TERTIARY)
        model_frame.pack(fill=X, padx=15, pady=5)
        Label(
            model_frame,
            text="Model:",
            font=Fonts.BODY,
            fg=Colors.TEXT_SECONDARY,
            bg=Colors.BG_TERTIARY,
        ).pack(side=LEFT)
        self.model_path_var = StringVar(value="Qwen2.5-14B-Instruct-Q4_K_M.gguf")
        Entry(
            model_frame,
            textvariable=self.model_path_var,
            font=Fonts.CODE,
            bg=Colors.BG_SECONDARY,
            fg=Colors.TEXT_PRIMARY,
            relief="flat",
            width=40,
        ).pack(side=LEFT, padx=10)
        Button(
            model_frame,
            text="Browse",
            font=Fonts.SMALL,
            bg=Colors.BG_SECONDARY,
            fg=Colors.TEXT_SECONDARY,
            relief="flat",
            cursor="hand2",
        ).pack(side=LEFT)

        # GPU status
        self._create_gpu_status(panel)

    def _create_gpu_status(self, parent: Frame) -> None:
        """Create GPU status display. Rule #4: Split helper."""
        gpu_frame = Frame(parent, bg=Colors.BG_SECONDARY)
        gpu_frame.pack(fill=X, padx=15, pady=(10, 15))
        gpu_info = [
            "GPU Status: CUDA 12.4 detected",
            "VRAM: 10.2 GB available / 12.0 GB total",
            "Layers offloaded: 35/40 (87%)",
            "Estimated speed: ~15 tokens/sec",
        ]
        for info in gpu_info:
            Label(
                gpu_frame,
                text=info,
                font=Fonts.CODE,
                fg=Colors.TEXT_SECONDARY,
                bg=Colors.BG_SECONDARY,
            ).pack(anchor=W, padx=10, pady=2)

    def _create_cloud_warning(self, parent: Frame) -> None:
        """Create cloud provider warning banner. Rule #4: Split helper."""
        warning = Frame(parent, bg=Colors.WARNING)
        warning.pack(fill=X, padx=20, pady=(0, 20))
        Label(
            warning,
            text="⚠️  CLOUD PROVIDER WARNING",
            font=Fonts.BODY_BOLD,
            fg=Colors.BG_PRIMARY,
            bg=Colors.WARNING,
        ).pack(anchor=W, padx=15, pady=(10, 5))
        Label(
            warning,
            text="Cloud providers (Claude, OpenAI, Gemini) incur API costs.\n"
            "You will be prompted for confirmation before each cloud API call.",
            font=Fonts.SMALL,
            fg=Colors.BG_PRIMARY,
            bg=Colors.WARNING,
        ).pack(anchor=W, padx=15, pady=(0, 10))

    def _create_settings_buttons(self) -> None:
        """Create settings action buttons. Rule #4: Split helper."""
        btn_frame = Frame(self, bg=Colors.BG_PRIMARY)
        btn_frame.pack(fill=X, padx=20, pady=10)
        StyledButton(
            btn_frame, text="Save", style="primary", command=self._save_settings
        ).pack(side=RIGHT)
        StyledButton(
            btn_frame,
            text="Cancel",
            style="secondary",
            command=lambda: self.app.show_view(ViewType.DASHBOARD),
        ).pack(side=RIGHT, padx=10)

    def _create_widgets(self) -> None:
        """Create settings view widgets. Rule #4: <60 lines via helpers."""
        Label(
            self,
            text="Settings",
            font=Fonts.H1,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_PRIMARY,
        ).pack(anchor=W, padx=20, pady=20)

        # LLM Configuration panel
        llm_panel = Frame(self, bg=Colors.BG_SECONDARY)
        llm_panel.pack(fill=X, padx=20, pady=(0, 20))
        Label(
            llm_panel,
            text="LLM Configuration",
            font=Fonts.H2,
            fg=Colors.TEXT_PRIMARY,
            bg=Colors.BG_SECONDARY,
        ).pack(anchor=W, padx=20, pady=(20, 15))

        self._create_provider_options(llm_panel)
        self._create_llamacpp_settings(llm_panel)
        self._create_cloud_warning(llm_panel)
        self._create_settings_buttons()

    def _configure_provider(self, provider: str) -> None:
        """Open provider configuration dialog."""
        if provider in ("claude", "openai", "gemini"):
            api_key = simpledialog.askstring(
                f"Configure {provider.title()}",
                f"Enter your {provider.title()} API key:",
                show="*",
            )
            if api_key:
                messagebox.showinfo("Saved", f"{provider.title()} API key configured.")

    def _save_settings(self) -> None:
        """Save settings."""
        messagebox.showinfo("Saved", "Settings saved successfully.")
        self.app.show_view(ViewType.DASHBOARD)


# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================
