"""Transform command group - Data transformation operations.

Provides commands for transforming and processing documents:
- split: Split documents into chunks / partition collections
- merge: Merge and deduplicate chunks
- filter: Filter chunks by criteria
- clean: Clean and normalize text
- enrich: Enrich with metadata

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations


from ingestforge.cli.transform.main import app as transform_app

__all__ = ["transform_app"]
