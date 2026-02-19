"""IngestForge Professional GUI Application (Facade Module).

This module provides backwards compatibility by re-exporting all GUI components
from their modular locations. The actual implementation has been split across
multiple modules to comply with NASA JPL Rule #4 (function length < 200 lines).

Modular Structure (JPL-004.1):
- gui_theme.py: Colors and Fonts
- gui_state.py: Data models and enums
- gui_widgets.py: Reusable UI components
- gui_views_base.py: BaseView class
- gui_views_dashboard.py: DashboardView
- gui_views_import.py: ImportView
- gui_views_processing.py: ProcessingView
- gui_views_query.py: QueryView
- gui_views_analyze.py: AnalyzeView, LiteraryView
- gui_views_settings.py: SettingsView
- gui_app.py: IngestForgeGUI main class

NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

# Re-export theme components
from ingestforge.cli.interactive.gui_theme import Colors, Fonts

# Re-export state models
from ingestforge.cli.interactive.gui_state import (
    ViewType,
    ProcessingResult,
    DocumentInfo,
    SearchResult,
    PipelineStage,
)

# Re-export widgets
from ingestforge.cli.interactive.gui_widgets import (
    StyledButton,
    StyledEntry,
    StatCard,
    ProgressBar,
    ToolBadge,
    NavButton,
    LogViewer,
    PipelineVisualizer,
)

# Re-export views
from ingestforge.cli.interactive.gui_views_base import BaseView
from ingestforge.cli.interactive.gui_views_dashboard import DashboardView
from ingestforge.cli.interactive.gui_views_import import ImportView
from ingestforge.cli.interactive.gui_views_processing import ProcessingView
from ingestforge.cli.interactive.gui_views_query import QueryView
from ingestforge.cli.interactive.gui_views_analyze import AnalyzeView, LiteraryView
from ingestforge.cli.interactive.gui_views_study import StudyView
from ingestforge.cli.interactive.gui_views_monitor import MonitorView
from ingestforge.cli.interactive.gui_views_settings import SettingsView

# Re-export main app
from ingestforge.cli.interactive.gui_app import IngestForgeGUI, main

__all__ = [
    # Theme
    "Colors",
    "Fonts",
    # State
    "ViewType",
    "ProcessingResult",
    "DocumentInfo",
    "SearchResult",
    "PipelineStage",
    # Widgets
    "StyledButton",
    "StyledEntry",
    "StatCard",
    "ProgressBar",
    "ToolBadge",
    "NavButton",
    "LogViewer",
    "PipelineVisualizer",
    # Views
    "BaseView",
    "DashboardView",
    "ImportView",
    "ProcessingView",
    "QueryView",
    "AnalyzeView",
    "LiteraryView",
    "StudyView",
    "MonitorView",
    "SettingsView",
    # App
    "IngestForgeGUI",
    "main",
]
