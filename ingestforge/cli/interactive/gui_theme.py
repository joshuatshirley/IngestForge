"""GUI Theme - Color Palette and Typography.

Design system for IngestForge GUI application.
Extracted from gui_menu.py for modularity (JPL-004).

NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations


class Colors:
    """Design system color palette (Dark Mode)."""

    # Background colors
    BG_PRIMARY = "#1a1a2e"  # Main content background
    BG_SECONDARY = "#16213e"  # Cards, panels, sidebar
    BG_TERTIARY = "#0f3460"  # Hover states, emphasis
    BG_HEADER = "#16213e"  # Header background

    # Accent colors
    ACCENT_PRIMARY = "#e94560"  # Brand accent, CTAs, errors
    ACCENT_SECONDARY = "#4ecca3"  # Success, confirmations
    ACCENT_TERTIARY = "#4fc3f7"  # Links, info, tool badges

    # Text colors
    TEXT_PRIMARY = "#e0e0e0"  # Main text
    TEXT_SECONDARY = "#a0a0a0"  # Muted text, labels
    TEXT_DIMMED = "#666666"  # Disabled, placeholder

    # Semantic colors
    SUCCESS = "#4ecca3"
    WARNING = "#f9a825"
    ERROR = "#e94560"
    INFO = "#64b5f6"

    # Component colors
    BUTTON_PRIMARY = "#4ecca3"
    BUTTON_SECONDARY = "#0f3460"
    BUTTON_DANGER = "#e94560"
    BUTTON_TEXT = "#ffffff"

    # Progress bar gradient (left to right)
    PROGRESS_START = "#4ecca3"
    PROGRESS_END = "#4fc3f7"


class Fonts:
    """Design system typography."""

    FAMILY_UI = "Segoe UI"
    FAMILY_CODE = "Consolas"

    H1 = (FAMILY_UI, 24, "bold")
    H2 = (FAMILY_UI, 18, "bold")
    H3 = (FAMILY_UI, 14, "bold")
    BODY = (FAMILY_UI, 13)
    BODY_BOLD = (FAMILY_UI, 13, "bold")
    CODE = (FAMILY_CODE, 12)
    LABEL = (FAMILY_UI, 11)
    BADGE = (FAMILY_UI, 10, "bold")
    SMALL = (FAMILY_UI, 10)
