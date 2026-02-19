"""Study commands package.

Provides study tools for learning from ingested content:
- quiz: Generate quiz questions from knowledge base
- flashcards: Create flashcard sets for memorization

Usage:
    from ingestforge.cli.study import study_app
"""

from __future__ import annotations

from ingestforge.cli.study.main import app as study_app

__all__ = ["study_app"]
