"""
Code Processor.

IFProcessor implementation for code file extraction (Apex, LWC).
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
MAX_CODE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


@register_processor(
    processor_id="code-processor",
    capabilities=["code-extraction", "apex", "lwc"],
    mime_types=["text/x-apex", "text/javascript", "application/javascript"],
)
class CodeProcessor(IFProcessor):
    """
    IFProcessor for code file extraction (Apex and LWC).

    Replaces PipelineSplittersMixin._split_code_document().
    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize code processor.

        Args:
            config: Optional pipeline configuration.
        """
        self._config = config
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "code-processor"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Capabilities provided by this processor."""
        return ["code-extraction", "apex", "lwc"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement."""
        return 100

    def is_available(self) -> bool:
        """Check if code processing is available."""
        return True  # No external dependencies for basic code processing

    def is_code_file(self, file_path: Path) -> bool:
        """
        Check if file is Apex or LWC code.

        Rule #1: Extract boolean logic to reduce nesting.
        Rule #4: Function < 60 lines.
        """
        suffix = file_path.suffix.lower()
        if suffix in (".cls", ".trigger"):
            return True
        if suffix == ".js" and "lwc" in str(file_path).lower():
            return True
        return False

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process a code file artifact.

        Rule #7: Check return values.

        Args:
            artifact: Input IFFileArtifact pointing to code file.

        Returns:
            IFTextArtifact with extracted content or IFFailureArtifact on error.
        """
        if not isinstance(artifact, IFFileArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-code-error",
                error_message=f"Expected IFFileArtifact, got {type(artifact).__name__}",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
            )

        try:
            file_path = artifact.file_path
            return self._process_code(file_path, artifact)
        except Exception as e:
            logger.exception(f"Code processing failed: {e}")
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-code-error",
                error_message=str(e),
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
            )

    def _process_code(self, file_path: Path, artifact: IFFileArtifact) -> IFArtifact:
        """
        Internal code processing logic.

        Rule #4: Function < 60 lines.
        """
        # JPL Rule #2: Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_CODE_SIZE_BYTES:
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-code-error",
                error_message=f"Code file too large: {file_size} bytes (max {MAX_CODE_SIZE_BYTES})",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
            )

        suffix = file_path.suffix.lower()
        code_result = self._dispatch_processor(file_path, suffix)

        source_loc = SourceLocation(
            source_type=SourceType.CODE,
            title=code_result.metadata.get("name", file_path.stem),
            file_path=str(file_path),
        )

        return IFTextArtifact(
            artifact_id=f"{artifact.artifact_id}-text",
            content=code_result.text,
            parent_id=artifact.artifact_id,
            provenance=artifact.provenance + [self.processor_id],
            metadata={
                "source_type": "code",
                "language": code_result.metadata.get("language", suffix[1:]),
                "title": source_loc.title,
                "source_location": str(source_loc),
                **code_result.metadata,
            },
        )

    def _dispatch_processor(self, file_path: Path, suffix: str) -> Any:
        """
        Dispatch to appropriate code processor based on file type.

        Rule #4: Function < 60 lines.
        """
        if suffix in (".cls", ".trigger"):
            from ingestforge.ingest.apex_processor import ApexProcessor

            processor = ApexProcessor()
            return processor.process(file_path)
        else:  # .js with lwc
            from ingestforge.ingest.lwc_processor import LWCProcessor

            processor = LWCProcessor()
            return processor.process(file_path)

    def teardown(self) -> bool:
        """Clean up resources."""
        return True
