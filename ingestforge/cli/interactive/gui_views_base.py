"""GUI Views Base - Base class for all views.

Extracted from gui_menu.py for modularity (JPL-004.1).

NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from tkinter import Frame

from ingestforge.cli.interactive.gui_theme import Colors

if TYPE_CHECKING:
    from ingestforge.cli.interactive.gui_app import IngestForgeGUI


class BaseView(Frame):
    """Base class for all views."""

    def __init__(self, parent: Any, app: IngestForgeGUI, **kwargs: Any) -> None:
        super().__init__(parent, bg=Colors.BG_PRIMARY, **kwargs)
        self.app = app

    def on_show(self) -> None:
        """Called when view is shown."""
        pass

    def on_hide(self) -> None:
        """Called when view is hidden."""
        pass
