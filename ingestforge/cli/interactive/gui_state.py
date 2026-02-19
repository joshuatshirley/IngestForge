"""GUI State - Data Models and Enums.

State management data classes for IngestForge GUI application.
Extracted from gui_menu.py for modularity (JPL-004).

NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List


class ViewType(Enum):
    """Navigation view types."""

    DASHBOARD = auto()
    DOCUMENTS = auto()
    IMPORT = auto()
    PROCESSING = auto()
    QUERY = auto()
    ANALYZE = auto()
    LITERARY = auto()
    STUDY = auto()
    EXPORT = auto()
    MONITOR = auto()
    SETTINGS = auto()


@dataclass
class ProcessingResult:
    """Result from a processing operation."""

    success: bool
    files_processed: int = 0
    chunks_created: int = 0
    tools_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_sec: float = 0.0


@dataclass
class DocumentInfo:
    """Information about a queued document."""

    path: Path
    name: str
    size_bytes: int
    format_type: str
    status: str = "pending"  # pending, processing, complete, error


@dataclass
class SearchResult:
    """Search result with source."""

    content: str
    source: str
    score: float
    chunk_id: str = ""


@dataclass
class PipelineStage:
    """Pipeline processing stage."""

    name: str
    icon: str
    status: str = "pending"  # pending, active, complete, error
    progress: float = 0.0
