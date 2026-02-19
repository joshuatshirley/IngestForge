"""
HTML Extractor Processor.

IFProcessor implementation for HTML document extraction.
NASA JPL Power of Ten compliant.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFFileArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)
from ingestforge.core.pipeline.registry import register_processor

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_HTML_SIZE_BYTES = 50 * 1024 * 1024  # 50MB


@register_processor(
    processor_id="html-extractor",
    capabilities=["html-extraction", "web-content"],
    mime_types=["text/html", "application/xhtml+xml"],
)
class HTMLExtractor(IFProcessor):
    """
    IFProcessor for HTML document extraction.

    Replaces PipelineSplittersMixin._split_html_document().
    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize HTML extractor.

        Args:
            config: Optional pipeline configuration.
        """
        self._config = config
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "html-extractor"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Capabilities provided by this processor."""
        return ["html-extraction", "web-content"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement."""
        return 200

    def is_available(self) -> bool:
        """Check if HTML processing is available."""
        try:
            from ingestforge.ingest.html_processor import HTMLProcessor

            return True
        except ImportError:
            return False

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process an HTML file artifact.

        Rule #7: Check return values.

        Args:
            artifact: Input IFFileArtifact pointing to HTML file.

        Returns:
            IFTextArtifact with extracted content or IFFailureArtifact on error.
        """
        if not isinstance(artifact, IFFileArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-html-error",
                error_message=f"Expected IFFileArtifact, got {type(artifact).__name__}",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
            )

        try:
            file_path = artifact.file_path
            return self._process_html(file_path, artifact)
        except Exception as e:
            logger.exception(f"HTML processing failed: {e}")
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-html-error",
                error_message=str(e),
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
            )

    def _process_html(self, file_path: Path, artifact: IFFileArtifact) -> IFArtifact:
        """
        Internal HTML processing logic.

        Rule #4: Function < 60 lines.
        """
        from ingestforge.ingest.html_processor import HTMLProcessor

        # JPL Rule #2: Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_HTML_SIZE_BYTES:
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-html-error",
                error_message=f"HTML file too large: {file_size} bytes (max {MAX_HTML_SIZE_BYTES})",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
            )

        html_processor = HTMLProcessor()
        html_result = html_processor.process(file_path)

        source_loc = html_result.source_location
        source_title = source_loc.title if source_loc else file_path.stem

        return IFTextArtifact(
            artifact_id=f"{artifact.artifact_id}-text",
            content=html_result.text,
            parent_id=artifact.artifact_id,
            provenance=artifact.provenance + [self.processor_id],
            metadata={
                "source_type": "html",
                "title": source_title,
                "word_count": len(html_result.text.split()),
                "char_count": len(html_result.text),
                "source_location": str(source_loc) if source_loc else None,
            },
        )

    def teardown(self) -> bool:
        """Clean up resources."""
        return True
