"""Comprehension commands - Understand and explain concepts."""

from ingestforge.cli.comprehension.base import ComprehensionCommand
from ingestforge.cli.comprehension.main import app as comprehension_app

__all__ = ["ComprehensionCommand", "comprehension_app"]
