"""
Markdown Processor.

IFProcessor implementation for markdown file extraction.
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
from ingestforge.core.provenance import SourceLocation, SourceType
from ingestforge.core.pipeline.registry import register_processor

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_MARKDOWN_SIZE_BYTES = 50 * 1024 * 1024  # 50MB
ADO_PREVIEW_SIZE = 500  # Characters to check for ADO markers


@register_processor(
    processor_id="markdown-processor",
    capabilities=["markdown-extraction", "ado-detection"],
    mime_types=["text/markdown", "text/x-markdown"],
)
class MarkdownProcessor(IFProcessor):
    """
    IFProcessor for markdown file extraction.

    Replaces PipelineSplittersMixin._split_markdown_document().
    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize markdown processor.

        Args:
            config: Optional pipeline configuration.
        """
        self._config = config
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "markdown-processor"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Capabilities provided by this processor."""
        return ["markdown-extraction", "ado-detection"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement."""
        return 100

    def is_available(self) -> bool:
        """Check if markdown processing is available."""
        return True

    def is_ado_markdown(self, file_path: Path) -> bool:
        """
        Check if markdown file is an ADO work item export.

        Rule #1: Extract detection logic to reduce nesting.
        Rule #4: Function < 60 lines.
        Rule #9: Full type hints.
        """
        try:
            preview = file_path.read_text(encoding="utf-8")[:ADO_PREVIEW_SIZE]
            return "| ID |" in preview and (
                "| Type |" in preview or "| Work Item Type |" in preview
            )
        except Exception as e:
            logger.debug(f"ADO detection failed: {e}")
            return False

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process a markdown file artifact.

        Rule #7: Check return values.

        Args:
            artifact: Input IFFileArtifact pointing to markdown file.

        Returns:
            IFTextArtifact with extracted content or IFFailureArtifact on error.
        """
        if not isinstance(artifact, IFFileArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-md-error",
                error_message=f"Expected IFFileArtifact, got {type(artifact).__name__}",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
            )

        try:
            file_path = artifact.file_path
            return self._process_markdown(file_path, artifact)
        except Exception as e:
            logger.exception(f"Markdown processing failed: {e}")
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-md-error",
                error_message=str(e),
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
            )

    def _process_markdown(
        self, file_path: Path, artifact: IFFileArtifact
    ) -> IFArtifact:
        """
        Internal markdown processing logic.

        Rule #4: Function < 60 lines.
        """
        # JPL Rule #2: Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_MARKDOWN_SIZE_BYTES:
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-md-error",
                error_message=f"Markdown file too large: {file_size} bytes (max {MAX_MARKDOWN_SIZE_BYTES})",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
            )

        # Check for ADO work item
        if self.is_ado_markdown(file_path):
            return self._process_ado_markdown(file_path, artifact)

        # Regular markdown
        content = file_path.read_text(encoding="utf-8")
        return IFTextArtifact(
            artifact_id=f"{artifact.artifact_id}-text",
            content=content,
            parent_id=artifact.artifact_id,
            provenance=artifact.provenance + [self.processor_id],
            metadata={
                "source_type": "markdown",
                "title": file_path.stem,
                "word_count": len(content.split()),
                "char_count": len(content),
            },
        )

    def _process_ado_markdown(
        self, file_path: Path, artifact: IFFileArtifact
    ) -> IFTextArtifact:
        """
        Process Azure DevOps work item markdown.

        Rule #4: Function < 60 lines.
        """
        from ingestforge.ingest.ado_processor import ADOProcessor

        ado_processor = ADOProcessor()
        ado_result = ado_processor.process(file_path)

        ado_id = ado_result.metadata.get("ado_id", 0)
        title = ado_result.metadata.get("title", file_path.stem)

        source_loc = SourceLocation(
            source_type=SourceType.ADO_WORK_ITEM,
            title=f"#{ado_id}: {title}" if ado_id else title,
            file_path=str(file_path),
        )

        return IFTextArtifact(
            artifact_id=f"{artifact.artifact_id}-ado",
            content=ado_result.text,
            parent_id=artifact.artifact_id,
            provenance=artifact.provenance + [self.processor_id],
            metadata={
                "source_type": "ado_work_item",
                "ado_id": ado_id,
                "title": title,
                "source_location": str(source_loc),
                **ado_result.metadata,
            },
        )

    def teardown(self) -> bool:
        """Clean up resources."""
        return True
