"""
Core Interfaces for the IngestForge (IF) Modular Architecture.

Defines the base contracts for Artifacts and Processors.
Follows NASA JPL Power of Ten rules.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field, field_validator

# JPL Rule #2: All loops and data structures must have fixed upper bounds
MAX_METADATA_KEYS = 128
MAX_METADATA_VALUE_SIZE = 65536  # 64KB per value when serialized


class IFArtifact(BaseModel, ABC):
    """
    Base class for all IngestForge Artifacts.

    Artifacts are immutable data containers that flow through the pipeline.

    Rule #9: Complete type hints.
    """

    model_config = {"frozen": True, "extra": "forbid", "arbitrary_types_allowed": True}

    artifact_id: str = Field(..., description="Unique identifier for the artifact")
    schema_version: str = Field(
        "1.0.0", description="SemVer version of the artifact schema"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=f"Arbitrary metadata (Max {MAX_METADATA_KEYS} keys, JSON-serializable values)",
    )
    provenance: List[str] = Field(
        default_factory=list,
        description="Chain of processor IDs that modified this data",
    )
    parent_id: Optional[str] = Field(
        None, description="ID of the artifact this was derived from"
    )
    root_artifact_id: Optional[str] = Field(
        None, description="ID of the original root artifact in the lineage chain"
    )
    lineage_depth: int = Field(
        0, ge=0, description="Number of derivation steps from root (0 = root artifact)"
    )

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate metadata dictionary constraints.

        Rule #2: Bounded data structures.
        Rule #7: Check all inputs.
        """
        # Check key count limit
        if len(v) > MAX_METADATA_KEYS:
            raise ValueError(
                f"Metadata exceeds maximum of {MAX_METADATA_KEYS} keys (got {len(v)})"
            )

        # Validate each value is JSON-serializable and within size limit
        for key, value in v.items():
            try:
                serialized = json.dumps(value)
                if len(serialized) > MAX_METADATA_VALUE_SIZE:
                    raise ValueError(
                        f"Metadata value for '{key}' exceeds {MAX_METADATA_VALUE_SIZE} bytes"
                    )
            except (TypeError, ValueError) as e:
                if "exceeds" in str(e):
                    raise
                raise ValueError(
                    f"Metadata value for '{key}' is not JSON-serializable: {type(value).__name__}"
                ) from e

        return v

    @abstractmethod
    def derive(self, processor_id: str, **kwargs: Any) -> "IFArtifact":
        """
        Create a new artifact derived from this one.

        Automatically sets parent_id, root_artifact_id, lineage_depth,
        and appends to provenance.
        """
        pass

    @property
    def is_root(self) -> bool:
        """Check if this artifact is a root (no parent)."""
        return self.parent_id is None

    @property
    def effective_root_id(self) -> str:
        """Get the root artifact ID (self if root, otherwise root_artifact_id)."""
        return self.root_artifact_id if self.root_artifact_id else self.artifact_id

    def validate_lineage_consistency(self) -> bool:
        """
        Validate that lineage fields are internally consistent.

        Rule #7: Check invariants.

        Returns:
            True if lineage is consistent.
        """
        # Root artifacts should have depth 0 and no parent
        if self.parent_id is None:
            return self.lineage_depth == 0 and self.root_artifact_id is None
        # Derived artifacts must have parent, root, and depth > 0
        return (
            self.lineage_depth > 0
            and self.root_artifact_id is not None
            and len(self.provenance) == self.lineage_depth
        )

    def metadata_to_json(self) -> str:
        """
        Serialize metadata dictionary to JSON string.

        Returns:
            JSON string representation of metadata.
        """
        return json.dumps(self.metadata, sort_keys=True)

    @property
    def metadata_key_count(self) -> int:
        """Get the current number of metadata keys."""
        return len(self.metadata)

    @property
    def can_add_metadata_keys(self) -> int:
        """Get the number of metadata keys that can still be added."""
        return MAX_METADATA_KEYS - len(self.metadata)


class IFProcessor(ABC):
    """
    Interface for all IngestForge Processors.

    Processors perform atomic operations on Artifacts.
    """

    @abstractmethod
    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process an artifact and return a new derived artifact.

        Rule #7: Check return values (Implicitly enforced by type hints).
        Rule #4: Method should be < 60 lines.

        Args:
            artifact: The input artifact.

        Returns:
            A new IFArtifact (or subclass).
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the processor and its dependencies are available.

        Returns:
            True if available, False otherwise.
        """
        pass

    def teardown(self) -> bool:
        """
        Perform resource cleanup.

        Rule #7: Check return values.

        Returns:
            True if cleanup successful.
        """
        return True

    @property
    @abstractmethod
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """SemVer version of this processor."""
        pass

    @property
    def capabilities(self) -> List[str]:
        """
        Functional capabilities provided by this processor.

        Examples: ["ocr", "table-extraction", "embedding", "summarization"]

        Override in subclasses to declare specific capabilities.
        Default returns empty list for backward compatibility.

        Rule #9: Complete type hints.
        """
        return []

    @property
    def memory_mb(self) -> int:
        """
        Estimated memory requirement in megabytes.

        Memory-Aware Selection.
        Rule #2: Fixed upper bound (default 100MB).
        Rule #9: Complete type hints.

        Override in subclasses to declare actual memory requirements.
        Default returns 100MB for backward compatibility.

        Returns:
            Estimated memory usage in MB.
        """
        return 100


class IFStage(ABC):
    """
    Interface for a Pipeline Stage.

    A stage orchestrates one or more processors to perform a high-level task.
    """

    @abstractmethod
    def execute(self, artifact: IFArtifact) -> IFArtifact:
        """
        Execute the stage logic.

        Args:
            artifact: Input artifact.

        Returns:
            Processed artifact or IFFailureArtifact.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the stage."""
        pass

    @property
    @abstractmethod
    def input_type(self) -> Type[IFArtifact]:
        """Expected input artifact type."""
        pass

    @property
    @abstractmethod
    def output_type(self) -> Type[IFArtifact]:
        """Produced output artifact type."""
        pass


# JPL Rule #2: Maximum interceptors per pipeline
MAX_INTERCEPTORS = 16


class IFInterceptor(ABC):
    """
    Interface for pipeline execution interceptors.

    Monitoring - Non-Blocking Interceptors.
    Interceptors observe pipeline execution without affecting the main flow.

    Rule #9: Complete type hints.
    """

    def pre_stage(
        self, stage_name: str, artifact: IFArtifact, document_id: str
    ) -> None:
        """
        Called before a stage executes.

        Monitoring - Non-Blocking Interceptors.
        Rule #7: Exceptions are isolated by the runner.

        Args:
            stage_name: Name of the stage about to execute.
            artifact: Input artifact for the stage.
            document_id: Document being processed.
        """
        pass

    def post_stage(
        self,
        stage_name: str,
        artifact: IFArtifact,
        document_id: str,
        duration_ms: float,
    ) -> None:
        """
        Called after a stage completes successfully.

        Monitoring - Non-Blocking Interceptors.
        Rule #7: Exceptions are isolated by the runner.

        Args:
            stage_name: Name of the completed stage.
            artifact: Output artifact from the stage.
            document_id: Document being processed.
            duration_ms: Execution time in milliseconds.
        """
        pass

    def on_error(
        self, stage_name: str, artifact: IFArtifact, document_id: str, error: Exception
    ) -> None:
        """
        Called when a stage fails.

        Monitoring - Non-Blocking Interceptors.
        Rule #7: Exceptions are isolated by the runner.

        Args:
            stage_name: Name of the failed stage.
            artifact: Last known artifact state.
            document_id: Document being processed.
            error: The exception that caused the failure.
        """
        pass

    def on_pipeline_start(self, document_id: str, stage_count: int) -> None:
        """
        Called when pipeline execution begins.

        Args:
            document_id: Document being processed.
            stage_count: Total number of stages to execute.
        """
        pass

    def on_pipeline_end(
        self, document_id: str, success: bool, total_duration_ms: float
    ) -> None:
        """
        Called when pipeline execution completes.

        Args:
            document_id: Document processed.
            success: True if pipeline completed without failure.
            total_duration_ms: Total execution time in milliseconds.
        """
        pass
