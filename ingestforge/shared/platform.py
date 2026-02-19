"""
Platform detection utilities for mobile/Termux environments.

Detects the runtime environment and provides configuration hints
for optimal performance on constrained devices.
"""

import os
import platform
from dataclasses import dataclass
from typing import List

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PlatformInfo:
    """Detected platform information."""

    is_termux: bool
    is_android: bool
    arch: str
    has_pymupdf: bool
    has_chromadb: bool
    has_sentence_transformers: bool
    has_torch: bool
    has_llama_cpp: bool

    @property
    def is_mobile(self) -> bool:
        """True if running on a mobile/constrained platform."""
        return self.is_termux or self.is_android

    @property
    def recommended_storage(self) -> str:
        """Recommend storage backend based on available dependencies."""
        if self.has_chromadb and self.has_sentence_transformers:
            return "chromadb"
        return "jsonl"

    @property
    def recommended_retrieval(self) -> str:
        """Recommend retrieval strategy based on available dependencies."""
        if self.has_sentence_transformers:
            return "hybrid"
        return "bm25"

    @property
    def supported_formats(self) -> List[str]:
        """List file formats that can be processed."""
        formats = [".txt", ".md"]
        if self.has_pymupdf:
            formats.append(".pdf")
        # These use optional deps with lazy imports
        try:
            import docx  # noqa: F401

            formats.append(".docx")
        except ImportError as e:
            logger.debug(f"docx library not available: {e}")
        try:
            import ebooklib  # noqa: F401

            formats.append(".epub")
        except ImportError as e:
            logger.debug(f"ebooklib library not available: {e}")
        return formats

    @property
    def missing_for_full(self) -> List[str]:
        """List packages needed for full functionality."""
        missing = []
        if not self.has_pymupdf:
            missing.append("pymupdf (PDF processing)")
        if not self.has_chromadb:
            missing.append("chromadb (vector search)")
        if not self.has_sentence_transformers:
            missing.append("sentence-transformers (embeddings)")
        if not self.has_torch:
            missing.append("torch (ML backend)")
        return missing


def detect_platform() -> PlatformInfo:
    """
    Detect the current runtime platform and available dependencies.

    Returns:
        PlatformInfo with environment details.
    """
    is_termux = (
        "TERMUX_VERSION" in os.environ
        or "com.termux" in os.environ.get("PREFIX", "")
        or os.path.isdir("/data/data/com.termux")
    )

    is_android = (
        is_termux
        or "ANDROID_ROOT" in os.environ
        or os.path.isfile("/system/build.prop")
    )

    arch = platform.machine().lower()

    # Check available native dependencies
    has_pymupdf = _check_import("fitz")
    has_chromadb = _check_import("chromadb")
    has_sentence_transformers = _check_import("sentence_transformers")
    has_torch = _check_import("torch")
    has_llama_cpp = _check_import("llama_cpp")

    return PlatformInfo(
        is_termux=is_termux,
        is_android=is_android,
        arch=arch,
        has_pymupdf=has_pymupdf,
        has_chromadb=has_chromadb,
        has_sentence_transformers=has_sentence_transformers,
        has_torch=has_torch,
        has_llama_cpp=has_llama_cpp,
    )


def _check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def is_termux() -> bool:
    """Quick check: are we running in Termux?"""
    return "TERMUX_VERSION" in os.environ or "com.termux" in os.environ.get(
        "PREFIX", ""
    )
