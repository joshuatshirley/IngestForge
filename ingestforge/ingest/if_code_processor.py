"""
IF-Protocol Code and Structure Processors.

Migration - Code & Structure Parity
Provides IF-Protocol-compliant processors for source code and
structured document formats (JSON, YAML, XML).

Follows NASA JPL Power of Ten rules.
"""

import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ingestforge.core.logging import get_logger
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFFileArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_FILE_SIZE = 10_000_000  # 10MB
MAX_LINES = 100_000
MAX_XML_DEPTH = 100

# Supported file extensions
CODE_EXTENSIONS: Set[str] = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".cs",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".sql",
    ".r",
    ".m",
    ".lua",
    ".pl",
}

STRUCTURE_EXTENSIONS: Dict[str, str] = {
    ".json": "application/json",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".xml": "application/xml",
    ".toml": "application/toml",
    ".ini": "text/plain",
    ".cfg": "text/plain",
}


class IFCodeProcessor(IFProcessor):
    """
    IF-Protocol processor for source code extraction.

    Extracts text from source code files while preserving
    syntax structure and providing proper artifact tracking.

    Rule #9: Complete type hints.
    """

    def __init__(self, processor_id: Optional[str] = None, version: str = "1.0.0"):
        """
        Initialize the code processor.

        Args:
            processor_id: Optional custom ID. Defaults to 'if-code-extractor'.
            version: SemVer version string.
        """
        self._processor_id = processor_id or "if-code-extractor"
        self._version = version

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return self._processor_id

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Functional capabilities provided by this processor."""
        return ["ingest.code", "text-extraction"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 64

    def is_available(self) -> bool:
        """
        Check if processor is available.

        Rule #7: Check return values.
        """
        return True  # No external dependencies

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Extract text from source code file.

        Rule #4: Method should be < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact (must be IFFileArtifact).

        Returns:
            IFTextArtifact on success, IFFailureArtifact on error.
        """
        if not isinstance(artifact, IFFileArtifact):
            return self._create_failure(
                artifact,
                f"IFCodeProcessor requires IFFileArtifact, got {type(artifact).__name__}",
            )

        file_path = artifact.file_path
        suffix = file_path.suffix.lower()

        if suffix not in CODE_EXTENSIONS:
            return self._create_failure(artifact, f"Unsupported code format: {suffix}")

        if not file_path.exists():
            return self._create_failure(artifact, f"File not found: {file_path}")

        try:
            content = self._extract_code(file_path)
            return self._create_text_artifact(artifact, content, suffix)
        except Exception as e:
            logger.error(f"Code extraction failed: {e}")
            return self._create_failure(artifact, str(e))

    def _extract_code(self, file_path: Path) -> str:
        """
        Extract code content from file.

        Rule #2: Enforce size limits.
        Rule #4: Function < 60 lines.
        """
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File exceeds {MAX_FILE_SIZE} bytes limit")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= MAX_LINES:
                    lines.append(f"\n... [Truncated at {MAX_LINES} lines]")
                    break
                lines.append(line)

        return "".join(lines)

    def _create_text_artifact(
        self, source: IFFileArtifact, content: str, file_type: str
    ) -> IFTextArtifact:
        """Create IFTextArtifact from extracted code."""
        line_count = content.count("\n") + 1
        return IFTextArtifact(
            artifact_id=f"{source.artifact_id}-code-{uuid.uuid4().hex[:8]}",
            content=content,
            parent_id=source.artifact_id,
            root_artifact_id=source.effective_root_id,
            lineage_depth=source.lineage_depth + 1,
            provenance=source.provenance + [self._processor_id],
            metadata={
                "source_file": str(source.file_path),
                "file_type": file_type,
                "extractor": self._processor_id,
                "line_count": line_count,
                "char_count": len(content),
            },
        )

    def _create_failure(
        self, artifact: IFArtifact, error_message: str
    ) -> IFFailureArtifact:
        """Create IFFailureArtifact for error cases."""
        return IFFailureArtifact(
            artifact_id=f"{artifact.artifact_id}-code-failed",
            error_message=error_message,
            failed_processor_id=self._processor_id,
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.effective_root_id,
            lineage_depth=artifact.lineage_depth + 1,
            provenance=artifact.provenance + [self._processor_id],
        )

    def teardown(self) -> bool:
        """Perform resource cleanup."""
        return True


class IFStructureProcessor(IFProcessor):
    """
    IF-Protocol processor for structured document extraction.

    Handles JSON, YAML, XML, TOML, and INI files.
    Extracts content while validating structure.

    Rule #9: Complete type hints.
    """

    def __init__(self, processor_id: Optional[str] = None, version: str = "1.0.0"):
        """
        Initialize the structure processor.

        Args:
            processor_id: Optional custom ID. Defaults to 'if-structure-extractor'.
            version: SemVer version string.
        """
        self._processor_id = processor_id or "if-structure-extractor"
        self._version = version
        self._yaml_module: Any = None

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return self._processor_id

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Functional capabilities provided by this processor."""
        return ["ingest.json", "ingest.yaml", "ingest.xml", "text-extraction"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 128

    def is_available(self) -> bool:
        """Check if processor is available."""
        return True

    def _load_yaml(self) -> Any:
        """Lazy-load PyYAML module."""
        if self._yaml_module is None:
            try:
                import yaml

                self._yaml_module = yaml
            except ImportError:
                return None
        return self._yaml_module

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Extract and validate structured document.

        Rule #4: Method should be < 60 lines.
        Rule #7: Check return values.
        """
        if not isinstance(artifact, IFFileArtifact):
            return self._create_failure(
                artifact,
                f"IFStructureProcessor requires IFFileArtifact, got {type(artifact).__name__}",
            )

        file_path = artifact.file_path
        suffix = file_path.suffix.lower()

        if suffix not in STRUCTURE_EXTENSIONS:
            return self._create_failure(
                artifact, f"Unsupported structure format: {suffix}"
            )

        if not file_path.exists():
            return self._create_failure(artifact, f"File not found: {file_path}")

        try:
            content, parsed = self._extract_structure(file_path, suffix)
            return self._create_text_artifact(artifact, content, suffix, parsed)
        except Exception as e:
            logger.error(f"Structure extraction failed: {e}")
            return self._create_failure(artifact, str(e))

    def _extract_structure(self, file_path: Path, suffix: str) -> tuple[str, bool]:
        """
        Extract structured content from file.

        Returns:
            Tuple of (content_string, was_parsed_successfully).
        """
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File exceeds {MAX_FILE_SIZE} bytes limit")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Validate structure based on type
        parsed = False
        if suffix == ".json":
            parsed = self._validate_json(content)
        elif suffix in {".yaml", ".yml"}:
            parsed = self._validate_yaml(content)
        elif suffix == ".xml":
            parsed = self._validate_xml(content)
        else:
            parsed = True  # TOML, INI - just read as text

        return content, parsed

    def _validate_json(self, content: str) -> bool:
        """Validate JSON content."""
        try:
            json.loads(content)
            return True
        except json.JSONDecodeError:
            return False

    def _validate_yaml(self, content: str) -> bool:
        """Validate YAML content."""
        yaml = self._load_yaml()
        if yaml is None:
            return True  # Can't validate without PyYAML
        try:
            yaml.safe_load(content)
            return True
        except yaml.YAMLError:
            return False

    def _validate_xml(self, content: str) -> bool:
        """
        Validate XML content.

        Rule #2: Limit parsing depth.
        """
        try:
            # Check depth limit
            depth = self._estimate_xml_depth(content)
            if depth > MAX_XML_DEPTH:
                return False
            ET.fromstring(content)
            return True
        except ET.ParseError:
            return False

    def _estimate_xml_depth(self, content: str) -> int:
        """Estimate XML nesting depth without full parse."""
        max_depth = 0
        current_depth = 0
        for char in content:
            if char == "<":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ">":
                current_depth = max(0, current_depth - 1)
        return max_depth

    def _create_text_artifact(
        self, source: IFFileArtifact, content: str, file_type: str, parsed: bool
    ) -> IFTextArtifact:
        """Create IFTextArtifact from structured content."""
        return IFTextArtifact(
            artifact_id=f"{source.artifact_id}-struct-{uuid.uuid4().hex[:8]}",
            content=content,
            parent_id=source.artifact_id,
            root_artifact_id=source.effective_root_id,
            lineage_depth=source.lineage_depth + 1,
            provenance=source.provenance + [self._processor_id],
            metadata={
                "source_file": str(source.file_path),
                "file_type": file_type,
                "extractor": self._processor_id,
                "parsed_successfully": parsed,
                "char_count": len(content),
            },
        )

    def _create_failure(
        self, artifact: IFArtifact, error_message: str
    ) -> IFFailureArtifact:
        """Create IFFailureArtifact for error cases."""
        return IFFailureArtifact(
            artifact_id=f"{artifact.artifact_id}-struct-failed",
            error_message=error_message,
            failed_processor_id=self._processor_id,
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.effective_root_id,
            lineage_depth=artifact.lineage_depth + 1,
            provenance=artifact.provenance + [self._processor_id],
        )

    def teardown(self) -> bool:
        """Perform resource cleanup."""
        self._yaml_module = None
        return True
