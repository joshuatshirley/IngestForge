"""
IFStage Implementations for Document Ingestion Pipeline.

Main Pipeline Runner Adoption.
Provides IFStage adapters that wrap the existing stage logic from PipelineStagesMixin.

NASA JPL Power of Ten compliant.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from ingestforge.core.pipeline.interfaces import IFArtifact, IFStage
from ingestforge.core.pipeline.artifacts import (
    IFFailureArtifact,
)
from pydantic import Field

if TYPE_CHECKING:
    from ingestforge.core.state import DocumentState
    from ingestforge.core.logging import PipelineLogger

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_CHAPTERS_PER_DOCUMENT = 1000
MAX_EXTRACTED_TEXTS = 1000
MAX_CHUNKS_PER_DOCUMENT = 10000


class IFPipelineContextArtifact(IFArtifact):
    """
    Artifact carrying document processing context through the pipeline.

    Envelope artifact for pipeline runner adoption.
    Carries all intermediate state between stages.

    Rule #9: Complete type hints.
    """

    # Source document info
    file_path: str = Field(..., description="Path to source document")
    document_id: str = Field(..., description="Unique document identifier")

    # Stage 1 outputs: Split
    chapters: List[str] = Field(
        default_factory=list, description="List of chapter file paths after splitting"
    )
    source_location: Optional[Dict[str, Any]] = Field(
        None, description="Source location metadata from splitting"
    )
    split_context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context from split stage"
    )

    # Stage 2 outputs: Extract
    extracted_texts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted text sections with metadata"
    )

    # Stage 3 outputs: Chunk
    chunk_records: List[Dict[str, Any]] = Field(
        default_factory=list, description="Serialized chunk records"
    )
    chunk_artifacts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Serialized chunk artifacts"
    )

    # Stage 4 outputs: Enrich
    enriched_chunks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Enriched chunk records"
    )

    # Stage 5 outputs: Index
    indexed_count: int = Field(0, description="Number of chunks indexed")

    # Processing metadata
    library: Optional[str] = Field(None, description="Target library name")
    processing_time_sec: float = Field(0.0, description="Total processing time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    success: bool = Field(True, description="Whether processing succeeded")

    def derive(self, processor_id: str, **kwargs: Any) -> "IFPipelineContextArtifact":
        """Create a derived context artifact."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "artifact_id": str(uuid.uuid4()),
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )


class IFSplitStage(IFStage):
    """
    Stage 1: Split document based on file type.

    Adapter wrapping _stage_split_document logic.
    Rule #4: Function < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        pipeline: Any,
        doc_state: "DocumentState",
        plog: "PipelineLogger",
    ) -> None:
        """
        Initialize split stage.

        Args:
            pipeline: Pipeline instance with splitter component.
            doc_state: Document state tracker.
            plog: Pipeline logger for progress.
        """
        self._pipeline = pipeline
        self._doc_state = doc_state
        self._plog = plog

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        """
        Execute split stage.

        Rule #7: Check return values.
        """
        if not isinstance(artifact, IFPipelineContextArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-split-error",
                error_message=f"Expected IFPipelineContextArtifact, got {type(artifact).__name__}",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["split-type-error"],
            )

        try:
            file_path = Path(artifact.file_path)
            chapters, source_loc, context = self._pipeline._stage_split_document(
                file_path,
                artifact.document_id,
                self._doc_state,
                self._plog,
            )

            # JPL Rule #2: Bound chapters
            if len(chapters) > MAX_CHAPTERS_PER_DOCUMENT:
                chapters = chapters[:MAX_CHAPTERS_PER_DOCUMENT]
                logger.warning(f"Truncated chapters to {MAX_CHAPTERS_PER_DOCUMENT}")

            # Convert to serializable format
            chapter_paths = [str(c) for c in chapters]
            source_loc_dict = None
            if source_loc:
                source_loc_dict = {
                    "source_id": getattr(source_loc, "source_id", None),
                    "source_type": str(getattr(source_loc, "source_type", "")),
                    "title": getattr(source_loc, "title", None),
                }

            return artifact.derive(
                "split-stage",
                chapters=chapter_paths,
                source_location=source_loc_dict,
                split_context=context,
            )

        except Exception as e:
            logger.exception(f"Split stage failed: {e}")
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-split-error",
                error_message=str(e),
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["split-crash"],
            )

    @property
    def name(self) -> str:
        """Name of this stage."""
        return "split"

    @property
    def input_type(self) -> Type[IFArtifact]:
        """Expected input artifact type."""
        return IFPipelineContextArtifact

    @property
    def output_type(self) -> Type[IFArtifact]:
        """Produced output artifact type."""
        return IFPipelineContextArtifact


class IFExtractStage(IFStage):
    """
    Stage 2: Extract text from document chapters.

    Adapter wrapping _stage_extract_text logic.
    Rule #4: Function < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        pipeline: Any,
        plog: "PipelineLogger",
    ) -> None:
        """
        Initialize extract stage.

        Args:
            pipeline: Pipeline instance with extractor component.
            plog: Pipeline logger for progress.
        """
        self._pipeline = pipeline
        self._plog = plog

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        """Execute extract stage."""
        if not isinstance(artifact, IFPipelineContextArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-extract-error",
                error_message=f"Expected IFPipelineContextArtifact, got {type(artifact).__name__}",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["extract-type-error"],
            )

        try:
            file_path = Path(artifact.file_path)
            chapters = [Path(c) for c in artifact.chapters]
            context = dict(artifact.split_context)

            extracted_texts = self._pipeline._stage_extract_text(
                chapters, file_path, context, self._plog
            )

            # JPL Rule #2: Bound extracted texts
            if len(extracted_texts) > MAX_EXTRACTED_TEXTS:
                extracted_texts = extracted_texts[:MAX_EXTRACTED_TEXTS]
                logger.warning(f"Truncated extracted texts to {MAX_EXTRACTED_TEXTS}")

            # Serialize for artifact (remove non-serializable _artifact key)
            serialized = []
            for text in extracted_texts:
                entry = {k: v for k, v in text.items() if k != "_artifact"}
                serialized.append(entry)

            return artifact.derive(
                "extract-stage",
                extracted_texts=serialized,
                split_context=context,  # May have been updated
            )

        except Exception as e:
            logger.exception(f"Extract stage failed: {e}")
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-extract-error",
                error_message=str(e),
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["extract-crash"],
            )

    @property
    def name(self) -> str:
        return "extract"

    @property
    def input_type(self) -> Type[IFArtifact]:
        return IFPipelineContextArtifact

    @property
    def output_type(self) -> Type[IFArtifact]:
        return IFPipelineContextArtifact


class IFRefineStage(IFStage):
    """
    Stage 2.5: Refine extracted text.

    Adapter wrapping _stage_refine_text logic.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        pipeline: Any,
        plog: "PipelineLogger",
    ) -> None:
        """Initialize refine stage."""
        self._pipeline = pipeline
        self._plog = plog

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        """Execute refine stage."""
        if not isinstance(artifact, IFPipelineContextArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-refine-error",
                error_message=f"Expected IFPipelineContextArtifact, got {type(artifact).__name__}",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["refine-type-error"],
            )

        try:
            file_path = Path(artifact.file_path)
            # Reconstruct extracted_texts from serialized form
            extracted_texts = list(artifact.extracted_texts)

            refined_texts = self._pipeline._stage_refine_text(
                extracted_texts, file_path, self._plog
            )

            return artifact.derive(
                "refine-stage",
                extracted_texts=refined_texts,
            )

        except Exception as e:
            logger.exception(f"Refine stage failed: {e}")
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-refine-error",
                error_message=str(e),
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["refine-crash"],
            )

    @property
    def name(self) -> str:
        return "refine"

    @property
    def input_type(self) -> Type[IFArtifact]:
        return IFPipelineContextArtifact

    @property
    def output_type(self) -> Type[IFArtifact]:
        return IFPipelineContextArtifact


class IFChunkStage(IFStage):
    """
    Stage 3: Chunk text into semantic units.

    Adapter wrapping _stage_chunk_text logic.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        pipeline: Any,
        doc_state: "DocumentState",
        plog: "PipelineLogger",
    ) -> None:
        """Initialize chunk stage."""
        self._pipeline = pipeline
        self._doc_state = doc_state
        self._plog = plog

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        """Execute chunk stage."""
        if not isinstance(artifact, IFPipelineContextArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-chunk-error",
                error_message=f"Expected IFPipelineContextArtifact, got {type(artifact).__name__}",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["chunk-type-error"],
            )

        try:
            file_path = Path(artifact.file_path)
            extracted_texts = list(artifact.extracted_texts)
            context = dict(artifact.split_context)

            # Reconstruct source_location if available
            source_location = None
            if artifact.source_location:
                from ingestforge.core.provenance import SourceLocation

                source_location = SourceLocation(
                    source_id=artifact.source_location.get("source_id"),
                    source_type=artifact.source_location.get("source_type", "file"),
                    title=artifact.source_location.get("title"),
                )

            chunks = self._pipeline._stage_chunk_text(
                extracted_texts,
                artifact.document_id,
                file_path,
                artifact.library,
                source_location,
                self._doc_state,
                context,
                self._plog,
            )

            # JPL Rule #2: Bound chunks
            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                chunks = chunks[:MAX_CHUNKS_PER_DOCUMENT]
                logger.warning(f"Truncated chunks to {MAX_CHUNKS_PER_DOCUMENT}")

            # Serialize chunks for artifact
            serialized_chunks = []
            for chunk in chunks:
                serialized_chunks.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "section_title": chunk.section_title,
                        "chunk_index": chunk.chunk_index,
                        "library": chunk.library,
                    }
                )

            return artifact.derive(
                "chunk-stage",
                chunk_records=serialized_chunks,
                split_context=context,
            )

        except Exception as e:
            logger.exception(f"Chunk stage failed: {e}")
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-chunk-error",
                error_message=str(e),
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["chunk-crash"],
            )

    @property
    def name(self) -> str:
        return "chunk"

    @property
    def input_type(self) -> Type[IFArtifact]:
        return IFPipelineContextArtifact

    @property
    def output_type(self) -> Type[IFArtifact]:
        return IFPipelineContextArtifact


class IFEnrichStageAdapter(IFStage):
    """
    Stage 4: Enrich chunks with embeddings and metadata.

    Adapter wrapping _stage_enrich_chunks logic.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        pipeline: Any,
        plog: "PipelineLogger",
    ) -> None:
        """Initialize enrich stage adapter."""
        self._pipeline = pipeline
        self._plog = plog

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        """Execute enrich stage."""
        if not isinstance(artifact, IFPipelineContextArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-enrich-error",
                error_message=f"Expected IFPipelineContextArtifact, got {type(artifact).__name__}",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["enrich-type-error"],
            )

        try:
            # Reconstruct chunks from serialized form
            from ingestforge.chunking.semantic_chunker import ChunkRecord

            chunks = []
            for chunk_data in artifact.chunk_records:
                chunk = ChunkRecord(
                    chunk_id=chunk_data.get("chunk_id", ""),
                    document_id=chunk_data.get("document_id", ""),
                    content=chunk_data.get("content", ""),
                    section_title=chunk_data.get("section_title", ""),
                    chunk_index=chunk_data.get("chunk_index", 0),
                    library=chunk_data.get("library", "default"),
                )
                chunks.append(chunk)

            context = dict(artifact.split_context)
            enriched_chunks = self._pipeline._stage_enrich_chunks(
                chunks, context, self._plog
            )

            # Serialize enriched chunks
            serialized = []
            for chunk in enriched_chunks:
                serialized.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "section_title": chunk.section_title,
                        "chunk_index": chunk.chunk_index,
                        "library": chunk.library,
                        "embedding": getattr(chunk, "embedding", None),
                        "entities": getattr(chunk, "entities", []),
                        "quality_score": getattr(chunk, "quality_score", 0.0),
                    }
                )

            return artifact.derive(
                "enrich-stage",
                enriched_chunks=serialized,
            )

        except Exception as e:
            logger.exception(f"Enrich stage failed: {e}")
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-enrich-error",
                error_message=str(e),
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["enrich-crash"],
            )

    @property
    def name(self) -> str:
        return "enrich"

    @property
    def input_type(self) -> Type[IFArtifact]:
        return IFPipelineContextArtifact

    @property
    def output_type(self) -> Type[IFArtifact]:
        return IFPipelineContextArtifact


class IFIndexStage(IFStage):
    """
    Stage 5: Index chunks into storage.

    Adapter wrapping _stage_index_chunks logic.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        pipeline: Any,
        doc_state: "DocumentState",
        plog: "PipelineLogger",
    ) -> None:
        """Initialize index stage."""
        self._pipeline = pipeline
        self._doc_state = doc_state
        self._plog = plog

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        """Execute index stage."""
        if not isinstance(artifact, IFPipelineContextArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-index-error",
                error_message=f"Expected IFPipelineContextArtifact, got {type(artifact).__name__}",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["index-type-error"],
            )

        try:
            # Reconstruct enriched chunks
            from ingestforge.chunking.semantic_chunker import ChunkRecord

            chunks = []
            for chunk_data in artifact.enriched_chunks:
                chunk = ChunkRecord(
                    chunk_id=chunk_data.get("chunk_id", ""),
                    document_id=chunk_data.get("document_id", ""),
                    content=chunk_data.get("content", ""),
                    section_title=chunk_data.get("section_title", ""),
                    chunk_index=chunk_data.get("chunk_index", 0),
                    library=chunk_data.get("library", "default"),
                    embedding=chunk_data.get("embedding"),
                    entities=chunk_data.get("entities", []),
                    quality_score=chunk_data.get("quality_score", 0.0),
                )
                chunks.append(chunk)

            file_path = Path(artifact.file_path)
            indexed_count = self._pipeline._stage_index_chunks(
                chunks, file_path, artifact.document_id, self._doc_state, self._plog
            )

            return artifact.derive(
                "index-stage",
                indexed_count=indexed_count,
                success=True,
            )

        except Exception as e:
            logger.exception(f"Index stage failed: {e}")
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-index-error",
                error_message=str(e),
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["index-crash"],
            )

    @property
    def name(self) -> str:
        return "index"

    @property
    def input_type(self) -> Type[IFArtifact]:
        return IFPipelineContextArtifact

    @property
    def output_type(self) -> Type[IFArtifact]:
        return IFPipelineContextArtifact


def create_initial_artifact(
    file_path: Path,
    document_id: str,
    library: Optional[str] = None,
) -> IFPipelineContextArtifact:
    """
    Create the initial context artifact for pipeline processing.

    Factory function for pipeline runner adoption.
    Rule #9: Complete type hints.

    Args:
        file_path: Path to the source document.
        document_id: Unique document identifier.
        library: Optional target library name.

    Returns:
        Initial IFPipelineContextArtifact ready for processing.
    """
    return IFPipelineContextArtifact(
        artifact_id=str(uuid.uuid4()),
        file_path=str(file_path.absolute()),
        document_id=document_id,
        library=library,
    )
