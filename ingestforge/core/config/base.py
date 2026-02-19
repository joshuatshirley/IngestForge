"""
Base configuration classes for project, ingest, and split settings.

Provides fundamental configuration dataclasses that define project structure,
document ingestion behavior, and PDF/document splitting strategies.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProjectConfig:
    """Project-level configuration."""

    name: str = "my-knowledge-base"
    version: str = "1.0.0"
    data_dir: str = ".data"
    ingest_dir: str = ".ingest"
    mobile_mode: bool = False


@dataclass
class IngestConfig:
    """Document ingestion configuration."""

    watch_enabled: bool = True
    watch_interval_sec: int = 5
    supported_formats: List[str] = field(
        default_factory=lambda: [".pdf", ".epub", ".txt", ".docx", ".md"]
    )
    move_completed: bool = True
    max_inline_size_mb: float = 10.0  # Max file size (MB) for inline processing
    pending_path_override: Optional[str] = None  # Override default pending folder path


@dataclass
class SplitConfig:
    """PDF/document splitting configuration."""

    use_toc: bool = True
    deep_split: bool = False
    min_chapter_size_kb: int = 5
    fallback_single_file: bool = True
