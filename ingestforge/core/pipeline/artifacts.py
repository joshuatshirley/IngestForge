"""
Concrete Artifact Implementations for IngestForge.

This module provides the standard data containers used in the modular pipeline.
Follows NASA JPL Power of Ten rules.

b: Added from_chunk_record/to_chunk_record for bidirectional conversion.
Added routing metadata to IFFileArtifact for content-based routing provenance.
"""

import hashlib
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional, List, Any, Dict, TYPE_CHECKING
from pydantic import Field, BaseModel
from ingestforge.core.pipeline.interfaces import IFArtifact

if TYPE_CHECKING:
    from ingestforge.chunking.semantic_chunker import ChunkRecord


def calculate_sha256(data: str) -> str:
    """Helper to calculate SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class IFFileArtifact(IFArtifact):
    """
    Artifact representing a source file.

    Enhanced with routing_metadata for content-based routing provenance.
    """

    file_path: Path = Field(..., description="Absolute path to the source file")
    mime_type: str = Field("application/octet-stream", description="IANA MIME type")
    content_hash: Optional[str] = Field(
        None, description="SHA-256 hash of the file content"
    )
    routing_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Routing decision metadata (confidence, method, detected_type)",
    )

    def model_post_init(self, __context: Any) -> None:
        """Auto-calculate hash if file exists and hash missing."""
        if not self.content_hash and self.file_path.exists():
            with open(self.file_path, "rb") as f:
                file_hash = hashlib.sha256()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
                object.__setattr__(self, "content_hash", file_hash.hexdigest())

    def derive(self, processor_id: str, **kwargs: Any) -> "IFArtifact":
        """Implementation of derive for FileArtifact."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        # Usually derive from File produces Text or Chapters,
        # but here we provide a generic way to clone with new metadata if needed.
        # Specific processors will call specialized constructors.
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )


class IFTextArtifact(IFArtifact):
    """
    Artifact representing extracted text content.
    """

    content: str = Field(..., description="Extracted text content")
    content_hash: Optional[str] = Field(
        None, description="SHA-256 hash of the text content"
    )

    def model_post_init(self, __context: Any) -> None:
        """Auto-calculate hash for text content."""
        if not self.content_hash:
            object.__setattr__(self, "content_hash", calculate_sha256(self.content))

    def derive(self, processor_id: str, **kwargs: Any) -> "IFTextArtifact":
        """Create a new TextArtifact derived from this one."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )


class IFChunkArtifact(IFArtifact):
    """
    Artifact representing a single chunk of content.
    """

    document_id: str = Field(..., description="ID of the source document")
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(0, description="Index of the chunk in the document")
    total_chunks: int = Field(1, description="Total number of chunks in the document")
    content_hash: Optional[str] = Field(
        None, description="SHA-256 hash of the chunk content"
    )

    def model_post_init(self, __context: Any) -> None:
        """Auto-calculate hash for chunk content."""
        if not self.content_hash:
            object.__setattr__(self, "content_hash", calculate_sha256(self.content))

    def derive(self, processor_id: str, **kwargs: Any) -> "IFChunkArtifact":
        """Create a new ChunkArtifact derived from this one."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )

    @classmethod
    def from_chunk_record(
        cls, record: Any, parent: Optional[IFArtifact] = None
    ) -> "IFChunkArtifact":
        """
        Create an IFChunkArtifact from a legacy ChunkRecord.

        b: Bidirectional conversion for migration.
        Supports both object-based records and raw dictionaries.
        Rule #7: Explicit return type.
        """

        def _get(attr: str, default: Any = None) -> Any:
            if isinstance(record, dict):
                return record.get(attr, default)
            return getattr(record, attr, default)

        # Build metadata from ChunkRecord fields
        metadata: Dict[str, Any] = {
            "section_title": _get("section_title"),
            "chunk_type": _get("chunk_type"),
            "source_file": _get("source_file"),
            "word_count": _get("word_count", 0),
            "char_count": _get("char_count", 0),
            "library": _get("library"),
            "is_read": _get("is_read", False),
            "element_type": _get("element_type"),
        }

        # Mapping for optional fields
        opt_fields = [
            "section_hierarchy",
            "page_start",
            "page_end",
            "source_location",
            "ingested_at",
            "tags",
            "author_id",
            "author_name",
            "embedding",
            "entities",
            "concepts",
            "quality_score",
            "visual_description",
            "bbox",
            "table_html",
        ]
        for field in opt_fields:
            val = _get(field)
            if val is not None:
                metadata[field] = val

        # Handle nested metadata
        if isinstance(record, dict) and "metadata" in record:
            metadata["chunk_metadata"] = record["metadata"]
        elif hasattr(record, "metadata"):
            metadata["chunk_metadata"] = record.metadata

        # Build lineage from parent if provided
        parent_id: Optional[str] = None
        root_artifact_id: Optional[str] = None
        lineage_depth: int = 0
        provenance: List[str] = []

        if parent:
            parent_id = parent.artifact_id
            root_artifact_id = parent.effective_root_id
            lineage_depth = parent.lineage_depth + 1
            provenance = list(parent.provenance) + ["from-chunk-record"]

        return cls(
            artifact_id=record.chunk_id or str(uuid.uuid4()),
            document_id=record.document_id,
            content=record.content,
            chunk_index=record.chunk_index,
            total_chunks=record.total_chunks,
            metadata=metadata,
            parent_id=parent_id,
            root_artifact_id=root_artifact_id,
            lineage_depth=lineage_depth,
            provenance=provenance,
        )

    def to_chunk_record(self) -> "ChunkRecord":
        """
        Convert this IFChunkArtifact to a legacy ChunkRecord.

        b: Bidirectional conversion for migration.
        Rule #7: Explicit return type.

        Returns:
            ChunkRecord with all data preserved from the artifact.
        """
        from ingestforge.chunking.semantic_chunker import ChunkRecord

        meta = self.metadata

        # Build ChunkRecord from artifact data
        return ChunkRecord(
            chunk_id=self.artifact_id,
            document_id=self.document_id,
            content=self.content,
            chunk_index=self.chunk_index,
            total_chunks=self.total_chunks,
            section_title=str(meta.get("section_title", "")),
            chunk_type=str(meta.get("chunk_type", "content")),
            source_file=str(meta.get("source_file", "")),
            word_count=int(meta.get("word_count", 0)),
            char_count=int(meta.get("char_count", 0)),
            library=str(meta.get("library", "default")),
            is_read=bool(meta.get("is_read", False)),
            element_type=str(meta.get("element_type", "NarrativeText")),
            section_hierarchy=meta.get("section_hierarchy"),
            page_start=meta.get("page_start"),
            page_end=meta.get("page_end"),
            source_location=meta.get("source_location"),
            ingested_at=meta.get("ingested_at"),
            tags=meta.get("tags", []),
            author_id=meta.get("author_id"),
            author_name=meta.get("author_name"),
            embedding=meta.get("embedding"),
            entities=meta.get("entities", []),
            concepts=meta.get("concepts", []),
            quality_score=float(meta.get("quality_score", 0.0)),
            visual_description=meta.get("visual_description"),
            bbox=meta.get("bbox"),
            table_html=meta.get("table_html"),
            metadata=_build_round_trip_metadata(meta, self),
        )


def _build_round_trip_metadata(
    meta: Dict[str, Any], artifact: "IFChunkArtifact"
) -> Dict[str, Any]:
    """
    Build metadata dict for ChunkRecord preserving lineage info.

    Rule #4: Function under 60 lines.

    Args:
        meta: Original artifact metadata.
        artifact: The source artifact.

    Returns:
        Dictionary with preserved lineage and custom metadata.
    """
    # Preserve any nested chunk_metadata
    result: Dict[str, Any] = dict(meta.get("chunk_metadata", {}))

    # Add lineage tracking for round-trip fidelity
    if artifact.parent_id:
        result["_lineage_parent_id"] = artifact.parent_id
    if artifact.root_artifact_id:
        result["_lineage_root_id"] = artifact.root_artifact_id
    if artifact.lineage_depth > 0:
        result["_lineage_depth"] = artifact.lineage_depth
    if artifact.provenance:
        result["_lineage_provenance"] = artifact.provenance
    if artifact.content_hash:
        result["_content_hash"] = artifact.content_hash

    return result


class SourceProvenance(BaseModel):
    """
    Provenance link for entity extraction source.

    Entity Artifact Model.
    Tracks exactly where an entity was extracted from.

    Rule #9: Complete type hints.
    """

    model_config = {"frozen": True}

    source_artifact_id: str = Field(
        ..., description="ID of the artifact where entity was extracted"
    )
    char_offset_start: int = Field(
        ..., description="Character offset where entity starts in source content"
    )
    char_offset_end: int = Field(
        ..., description="Character offset where entity ends in source content"
    )
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Extraction confidence score (0.0 to 1.0)"
    )
    extraction_method: str = Field(
        "unknown",
        description="Method used for extraction (e.g., 'spacy', 'llm', 'regex')",
    )


class EntityNode(BaseModel):
    """
    Node representing a single extracted entity.

    Entity Artifact Model.
    Confidence-Aware-Extraction - Every node contains confidence.
    Chain-of-Custody-Integrity - parent_hash for verification.
    Rule #9: Complete type hints.

    Attributes:
        entity_id: Unique identifier for this entity.
        entity_type: Type classification (e.g., PERSON, ORG, GPE).
        name: Canonical entity name.
        aliases: Alternative names for this entity.
        confidence: Extraction confidence score (0.0 to 1.0).
        parent_hash: SHA-256 hash of parent artifact content for integrity.
        source_provenance: Where this entity was extracted from.
    """

    model_config = {"frozen": True}

    entity_id: str = Field(..., description="Unique identifier for this entity")
    entity_type: str = Field(
        ..., description="Entity type (PERSON, ORG, GPE, DATE, etc.)"
    )
    name: str = Field(..., description="Canonical name of the entity")
    aliases: List[str] = Field(
        default_factory=list, description="Alternative names or variations"
    )
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Extraction confidence score (0.0 to 1.0). AC."
    )
    parent_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of parent artifact content for chain-of-custody. AC.",
    )
    source_provenance: SourceProvenance = Field(
        ..., description="Link to source artifact and position"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional entity properties (domain-specific)",
    )

    def verify_parent_integrity(self, artifact: "IFArtifact") -> bool:
        """
        Verify that this entity's parent hash matches the artifact.

        Chain-of-Custody-Integrity.
        Rule #7: Explicit return value.

        Args:
            artifact: The artifact to verify against.

        Returns:
            True if hash matches or no parent_hash set, False otherwise.
        """
        if self.parent_hash is None:
            return True  # No hash to verify
        artifact_hash = getattr(artifact, "content_hash", None)
        if artifact_hash is None:
            return False  # Artifact has no hash to compare
        return self.parent_hash == artifact_hash

    def __repr__(self) -> str:
        """Debug representation showing type and name."""
        return f"EntityNode({self.entity_type}: {self.name})"


class RelationshipEdge(BaseModel):
    """
    Edge representing a relationship between two entities.

    Entity Artifact Model.
    Rule #9: Complete type hints.

    Attributes:
        source_entity_id: ID of the source entity.
        target_entity_id: ID of the target entity.
        predicate: Relationship type (e.g., 'works_for', 'located_in').
        confidence: Extraction confidence score.
        source_provenance: Where this relationship was extracted from.
    """

    model_config = {"frozen": True}

    source_entity_id: str = Field(
        ..., description="ID of the source entity in the relationship"
    )
    target_entity_id: str = Field(
        ..., description="ID of the target entity in the relationship"
    )
    predicate: str = Field(
        ...,
        description="Relationship type (e.g., 'works_for', 'located_in', 'part_of')",
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Relationship extraction confidence (0.0 to 1.0)",
    )
    source_provenance: Optional[SourceProvenance] = Field(
        None, description="Where this relationship was extracted from (if available)"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional relationship properties (e.g., temporal bounds)",
    )

    def __repr__(self) -> str:
        """Debug representation showing relationship triple."""
        return f"Edge({self.source_entity_id} --{self.predicate}--> {self.target_entity_id})"


class IFEntityArtifact(IFArtifact):
    """
    Artifact representing extracted entities and their relationships.

    Entity Artifact Model.
    Provides a knowledge graph fragment with full provenance.

    Rule #4: Function < 60 lines.
    Rule #9: Complete type hints.

    Attributes:
        nodes: List of extracted entity nodes.
        edges: List of relationship edges between nodes.
        extraction_model: Model or method used for extraction.
        source_document_id: ID of the source document.
    """

    nodes: List[EntityNode] = Field(
        default_factory=list, description="List of extracted entity nodes"
    )
    edges: List[RelationshipEdge] = Field(
        default_factory=list, description="List of relationship edges between entities"
    )
    extraction_model: str = Field(
        "unknown", description="Model/method used for entity extraction"
    )
    source_document_id: Optional[str] = Field(
        None, description="ID of the source document for this extraction"
    )

    def derive(self, processor_id: str, **kwargs: Any) -> "IFEntityArtifact":
        """Create a new EntityArtifact derived from this one."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )

    @property
    def node_count(self) -> int:
        """Return the number of entity nodes."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Return the number of relationship edges."""
        return len(self.edges)

    def get_node_by_id(self, entity_id: str) -> Optional[EntityNode]:
        """
        Find a node by its entity ID.

        Rule #1: Linear search (no recursion).
        Rule #7: Explicit return value.
        """
        for node in self.nodes:
            if node.entity_id == entity_id:
                return node
        return None

    def get_nodes_by_type(self, entity_type: str) -> List[EntityNode]:
        """
        Find all nodes of a specific type.

        Rule #1: Linear search (no recursion).
        """
        return [n for n in self.nodes if n.entity_type == entity_type]

    def get_edges_for_node(self, entity_id: str) -> List[RelationshipEdge]:
        """
        Find all edges involving a specific entity (as source or target).

        Rule #1: Linear search (no recursion).
        """
        return [
            e
            for e in self.edges
            if e.source_entity_id == entity_id or e.target_entity_id == entity_id
        ]

    def __repr__(self) -> str:
        """Debug representation showing node and edge counts."""
        return (
            f"IFEntityArtifact(id={self.artifact_id!r}, "
            f"nodes={self.node_count}, edges={self.edge_count})"
        )


class IFFailureArtifact(IFArtifact):
    """
    Artifact representing a failure in the pipeline.
    """

    error_message: str = Field(..., description="Descriptive error message")
    stack_trace: Optional[str] = Field(
        None, description="Detailed stack trace if available"
    )
    failed_processor_id: Optional[str] = Field(
        None, description="ID of the processor that failed"
    )

    def derive(self, processor_id: str, **kwargs: Any) -> "IFFailureArtifact":
        """Failures don't usually produce children, but we satisfy the interface."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )


# =============================================================================
# IMAGE ARTIFACT ()
# =============================================================================


class BoundingBox(BaseModel):
    """
    Bounding box coordinates for visual elements.

    Hard-linking data points to bounding box coordinates.
    Rule #9: Complete type hints.
    """

    x1: float = Field(..., description="Left x coordinate (0-1 normalized)")
    y1: float = Field(..., description="Top y coordinate (0-1 normalized)")
    x2: float = Field(..., description="Right x coordinate (0-1 normalized)")
    y2: float = Field(..., description="Bottom y coordinate (0-1 normalized)")
    label: Optional[str] = Field(None, description="Label for the region")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Detection confidence")


class DataPoint(BaseModel):
    """
    Extracted data point from a chart or diagram.

    Logic to extract CSV data from chart images.
    Rule #9: Complete type hints.
    """

    label: str = Field(..., description="Data point label/category")
    value: float = Field(..., description="Numeric value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    bbox: Optional[BoundingBox] = Field(
        None, description="Bounding box in source image"
    )


class IFImageArtifact(IFArtifact):
    """
    Artifact representing an image with extracted visual data.

    IFImageArtifact Schema with pixel hash verification.
    VLM Vision Processor.
    Stores image metadata, bounding boxes, and extracted data points.

    Rule #9: Complete type hints.
    Rule #10: Verify SHA-256 hash of raw pixels.
    """

    image_path: Optional[str] = Field(None, description="Path to source image file")
    mime_type: str = Field("image/png", description="Image MIME type")
    width: int = Field(0, ge=0, description="Image width in pixels")
    height: int = Field(0, ge=0, description="Image height in pixels")

    # SHA-256 hash of raw pixel data (Rule #10)
    content_hash: Optional[str] = Field(
        None, description="SHA-256 hash of raw pixel data"
    )

    # Visual description from VLM (visual_summary)
    visual_summary: str = Field("", description="VLM-generated summary of the image")
    description: str = Field("", description="VLM-generated description of the image")

    # Extracted data (CSV data from charts)
    data_points: List[DataPoint] = Field(
        default_factory=list, description="Extracted data points"
    )
    chart_type: Optional[str] = Field(
        None, description="Detected chart type (bar, line, pie, etc.)"
    )

    # Bounding boxes for visual elements (hard-linking)
    regions: List[BoundingBox] = Field(
        default_factory=list, description="Detected regions/elements"
    )

    # Processing metadata
    vlm_model: Optional[str] = Field(None, description="VLM model used for extraction")
    extraction_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Overall extraction confidence"
    )

    def derive(self, processor_id: str, **kwargs: Any) -> "IFImageArtifact":
        """Create a derived image artifact."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )

    def to_csv(self) -> str:
        """
        Export extracted data points as CSV.

        Logic to extract CSV data from chart images.

        Returns:
            CSV string with headers: label,value,unit
        """
        if not self.data_points:
            return "label,value,unit\n"

        lines = ["label,value,unit"]
        for dp in self.data_points:
            unit = dp.unit or ""
            lines.append(f"{dp.label},{dp.value},{unit}")
        return "\n".join(lines)

    @property
    def has_data(self) -> bool:
        """Check if any data points were extracted."""
        return len(self.data_points) > 0

    def compute_pixel_hash(self, pixel_data: bytes) -> str:
        """
        Compute SHA-256 hash of raw pixel data.

        JPL Rule #10 verification of raw pixels.

        Args:
            pixel_data: Raw pixel bytes (RGB/RGBA format).

        Returns:
            SHA-256 hex digest of the pixel data.
        """
        return hashlib.sha256(pixel_data).hexdigest()

    def verify_pixel_hash(self, pixel_data: bytes) -> bool:
        """
        Verify pixel data matches stored content_hash.

        JPL Rule #10 integrity verification.

        Args:
            pixel_data: Raw pixel bytes to verify.

        Returns:
            True if hash matches, False otherwise.
        """
        if not self.content_hash:
            return False
        computed = self.compute_pixel_hash(pixel_data)
        return computed == self.content_hash

    @classmethod
    def from_image_file(
        cls,
        image_path: Path,
        visual_summary: str = "",
        **kwargs: Any,
    ) -> "IFImageArtifact":
        """
        Create IFImageArtifact from an image file with pixel hash.

        Computes SHA-256 of raw pixel data.
        Rule #7: Returns artifact with verified hash.

        Args:
            image_path: Path to the image file.
            visual_summary: Optional VLM-generated summary.
            **kwargs: Additional artifact fields.

        Returns:
            IFImageArtifact with computed content_hash.
        """
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size
                # Convert to RGB for consistent hashing
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pixel_data = img.tobytes()
                content_hash = hashlib.sha256(pixel_data).hexdigest()

            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
            }
            suffix = image_path.suffix.lower()
            mime_type = mime_map.get(suffix, "image/png")

            return cls(
                image_path=str(image_path),
                mime_type=mime_type,
                width=width,
                height=height,
                content_hash=content_hash,
                visual_summary=visual_summary,
                **kwargs,
            )
        except ImportError:
            # Pillow not installed, create without hash
            return cls(
                image_path=str(image_path),
                visual_summary=visual_summary,
                **kwargs,
            )
        except Exception:
            # File read error, create minimal artifact
            return cls(
                image_path=str(image_path),
                visual_summary=visual_summary,
                **kwargs,
            )


# =============================================================================
# AUDIO ARTIFACT ()
# =============================================================================


class AudioSegment(BaseModel):
    """
    A segment of audio with speaker and timestamp information.

    Fields for speaker_id, start_ms, end_ms.
    Rule #9: Complete type hints for timestamp ranges.
    """

    speaker_id: str = Field(..., description="Identifier for the speaker")
    start_ms: int = Field(..., ge=0, description="Start time in milliseconds")
    end_ms: int = Field(..., ge=0, description="End time in milliseconds")
    text: str = Field("", description="Transcribed text for this segment")
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Transcription confidence"
    )
    language: Optional[str] = Field(
        None, description="Detected language code (ISO 639-1)"
    )

    @property
    def duration_ms(self) -> int:
        """Duration of the segment in milliseconds."""
        return max(0, self.end_ms - self.start_ms)

    @property
    def duration_seconds(self) -> float:
        """Duration of the segment in seconds."""
        return self.duration_ms / 1000.0


class IFAudioArtifact(IFArtifact):
    """
    Artifact representing audio content with transcription and diarization.

    Data model for audio transcripts and playable segments.
    Rule #9: Complete type hints for timestamp ranges.
    """

    # Source file information
    audio_path: Optional[str] = Field(None, description="Path to source audio file")
    mime_type: str = Field("audio/wav", description="Audio MIME type")
    duration_ms: int = Field(0, ge=0, description="Total duration in milliseconds")
    sample_rate: int = Field(16000, gt=0, description="Sample rate in Hz")
    channels: int = Field(1, ge=1, le=8, description="Number of audio channels")

    # Content hash for integrity verification
    content_hash: Optional[str] = Field(None, description="SHA-256 hash of audio data")

    # Transcription data
    full_transcript: str = Field("", description="Complete transcription text")
    segments: List[AudioSegment] = Field(
        default_factory=list,
        description="Time-aligned transcript segments with speaker IDs",
    )

    # Processing metadata
    transcription_model: Optional[str] = Field(
        None, description="Model used for transcription"
    )
    language: Optional[str] = Field(None, description="Primary detected language")
    word_count: int = Field(0, ge=0, description="Total word count in transcript")

    def derive(self, processor_id: str, **kwargs: Any) -> "IFAudioArtifact":
        """Create a derived audio artifact."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )

    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        return self.duration_ms / 1000.0

    @property
    def speaker_count(self) -> int:
        """Number of unique speakers detected."""
        return len(set(seg.speaker_id for seg in self.segments))

    @property
    def speakers(self) -> List[str]:
        """List of unique speaker IDs in order of first appearance."""
        seen: Dict[str, bool] = {}
        result: List[str] = []
        for seg in self.segments:
            if seg.speaker_id not in seen:
                seen[seg.speaker_id] = True
                result.append(seg.speaker_id)
        return result

    def get_speaker_segments(self, speaker_id: str) -> List[AudioSegment]:
        """
        Get all segments for a specific speaker.

        Rule #1: Linear search, no recursion.
        """
        return [seg for seg in self.segments if seg.speaker_id == speaker_id]

    def get_segment_at(self, time_ms: int) -> Optional[AudioSegment]:
        """
        Find the segment containing the given timestamp.

        Rule #7: Returns None if no segment contains the timestamp.
        """
        for seg in self.segments:
            if seg.start_ms <= time_ms < seg.end_ms:
                return seg
        return None

    def to_srt(self) -> str:
        """
        Export segments as SRT subtitle format.

        Returns:
            SRT-formatted string with numbered entries.
        """
        if not self.segments:
            return ""

        lines: List[str] = []
        for i, seg in enumerate(self.segments, 1):
            start = self._ms_to_srt_time(seg.start_ms)
            end = self._ms_to_srt_time(seg.end_ms)
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(f"[{seg.speaker_id}] {seg.text}")
            lines.append("")

        return "\n".join(lines)

    def _ms_to_srt_time(self, ms: int) -> str:
        """
        Convert milliseconds to SRT timestamp format.

        Rule #4: Isolated helper function.
        """
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        millis = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    @classmethod
    def from_audio_file(
        cls,
        audio_path: Path,
        **kwargs: Any,
    ) -> "IFAudioArtifact":
        """
        Create IFAudioArtifact from an audio file.

        Rule #7: Handles missing dependencies gracefully.

        Args:
            audio_path: Path to the audio file.
            **kwargs: Additional artifact fields.

        Returns:
            IFAudioArtifact with basic metadata.
        """
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".webm": "audio/webm",
        }
        suffix = audio_path.suffix.lower()
        mime_type = mime_map.get(suffix, "audio/wav")

        # Try to compute content hash
        content_hash = None
        if audio_path.exists():
            try:
                with open(audio_path, "rb") as f:
                    content_hash = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                pass

        return cls(
            audio_path=str(audio_path),
            mime_type=mime_type,
            content_hash=content_hash,
            **kwargs,
        )


# =============================================================================
# CODE ARTIFACT ()
# =============================================================================


class SymbolKind(str, Enum):
    """
    Kind of code symbol (AST node type).

    Strict typing for AST nodes.
    Rule #9: Enumerated types for symbol classification.
    """

    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    INTERFACE = "interface"
    ENUM = "enum"
    MODULE = "module"
    PROPERTY = "property"
    PARAMETER = "parameter"
    TYPE_ALIAS = "type_alias"
    NAMESPACE = "namespace"


class CodeSymbol(BaseModel):
    """
    A symbol (class, method, variable) in source code.

    Strict typing for AST nodes.
    Rule #9: Complete type hints.
    """

    name: str = Field(..., description="Symbol name/identifier")
    kind: SymbolKind = Field(..., description="Type of symbol")
    line_start: int = Field(..., ge=1, description="Starting line number (1-indexed)")
    line_end: int = Field(..., ge=1, description="Ending line number (1-indexed)")
    column_start: int = Field(0, ge=0, description="Starting column (0-indexed)")
    column_end: int = Field(0, ge=0, description="Ending column (0-indexed)")
    signature: Optional[str] = Field(None, description="Function/method signature")
    docstring: Optional[str] = Field(None, description="Documentation string")
    parent_symbol: Optional[str] = Field(
        None, description="Parent symbol name (e.g., class for method)"
    )
    visibility: str = Field(
        "public", description="Visibility: public, private, protected"
    )
    is_async: bool = Field(False, description="Whether function/method is async")
    decorators: List[str] = Field(
        default_factory=list, description="Applied decorators"
    )

    @property
    def span(self) -> int:
        """Number of lines the symbol spans."""
        return max(1, self.line_end - self.line_start + 1)


class ImportInfo(BaseModel):
    """
    Information about an import statement.

    Fields for imports.
    Rule #9: Complete type hints.
    """

    module: str = Field(..., description="Module being imported")
    names: List[str] = Field(
        default_factory=list, description="Specific names imported"
    )
    alias: Optional[str] = Field(None, description="Import alias (as ...)")
    is_relative: bool = Field(False, description="Whether import is relative")
    line_number: int = Field(0, ge=0, description="Line number of import")


class ExportInfo(BaseModel):
    """
    Information about an exported symbol.

    Fields for exports.
    Rule #9: Complete type hints.
    """

    name: str = Field(..., description="Exported symbol name")
    kind: SymbolKind = Field(..., description="Type of exported symbol")
    is_default: bool = Field(False, description="Whether this is a default export")
    is_reexport: bool = Field(
        False, description="Whether this is a re-export from another module"
    )


class IFCodeArtifact(IFArtifact):
    """
    Artifact representing source code with structural analysis.

    Data structure for software architectural units.
    Rule #9: Strict typing for AST nodes.
    """

    # Source file information
    file_path: Optional[str] = Field(None, description="Path to source file")
    language: str = Field(
        ..., description="Programming language (python, javascript, apex, etc.)"
    )
    content: str = Field("", description="Source code content")
    content_hash: Optional[str] = Field(None, description="SHA-256 hash of source code")

    # Structural analysis (AC)
    imports: List[ImportInfo] = Field(
        default_factory=list, description="Import statements"
    )
    exports: List[ExportInfo] = Field(
        default_factory=list, description="Exported symbols"
    )
    symbols: List[CodeSymbol] = Field(
        default_factory=list, description="All code symbols"
    )

    # Code metrics
    line_count: int = Field(0, ge=0, description="Total lines of code")
    blank_lines: int = Field(0, ge=0, description="Number of blank lines")
    comment_lines: int = Field(0, ge=0, description="Number of comment lines")
    complexity: Optional[int] = Field(None, ge=0, description="Cyclomatic complexity")

    # Dependencies
    dependencies: List[str] = Field(
        default_factory=list, description="External dependencies"
    )

    def model_post_init(self, __context: Any) -> None:
        """Auto-calculate hash for source content."""
        if not self.content_hash and self.content:
            object.__setattr__(self, "content_hash", calculate_sha256(self.content))

    def derive(self, processor_id: str, **kwargs: Any) -> "IFCodeArtifact":
        """Create a derived code artifact."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )

    @property
    def classes(self) -> List[CodeSymbol]:
        """Get all class symbols."""
        return [s for s in self.symbols if s.kind == SymbolKind.CLASS]

    @property
    def functions(self) -> List[CodeSymbol]:
        """Get all function symbols (excluding methods)."""
        return [s for s in self.symbols if s.kind == SymbolKind.FUNCTION]

    @property
    def methods(self) -> List[CodeSymbol]:
        """Get all method symbols."""
        return [s for s in self.symbols if s.kind == SymbolKind.METHOD]

    def get_symbol(self, name: str) -> Optional[CodeSymbol]:
        """
        Find a symbol by name.

        Rule #7: Returns None if not found.
        """
        for symbol in self.symbols:
            if symbol.name == name:
                return symbol
        return None

    def get_symbols_at_line(self, line: int) -> List[CodeSymbol]:
        """
        Get all symbols that span the given line.

        Rule #1: Linear search, no recursion.
        """
        return [s for s in self.symbols if s.line_start <= line <= s.line_end]

    def get_class_methods(self, class_name: str) -> List[CodeSymbol]:
        """
        Get all methods belonging to a class.

        Rule #1: Linear search.
        """
        return [
            s
            for s in self.symbols
            if s.kind == SymbolKind.METHOD and s.parent_symbol == class_name
        ]

    @classmethod
    def from_source_file(
        cls,
        file_path: Path,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> "IFCodeArtifact":
        """
        Create IFCodeArtifact from a source file.

        Rule #7: Handles missing file gracefully.

        Args:
            file_path: Path to source file.
            language: Programming language (auto-detected if None).
            **kwargs: Additional artifact fields.

        Returns:
            IFCodeArtifact with source content.
        """
        # Language detection by extension
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cls": "apex",
            ".trigger": "apex",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
        }

        detected_lang = language
        if not detected_lang:
            suffix = file_path.suffix.lower()
            detected_lang = lang_map.get(suffix, "unknown")

        content = ""
        line_count = 0
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8")
                line_count = len(content.splitlines())
            except Exception:
                pass

        return cls(
            file_path=str(file_path),
            language=detected_lang,
            content=content,
            line_count=line_count,
            **kwargs,
        )


class IFDiscoveryIntentArtifact(IFTextArtifact):
    """
    Artifact representing a discovery suggestion for knowledge gaps.

    Proactive Scout
    Generated when the scout identifies missing connections or underexplored entities.

    NASA JPL Power of Ten compliant.
    Rule #9: Complete type hints.
    """

    target_entity: str = Field(..., description="Entity that needs more exploration")
    entity_type: str = Field(..., description="Type of the target entity")
    missing_link_type: str = Field(..., description="Type of connection that's missing")
    rationale: str = Field(
        ..., description="Explanation of why this discovery is recommended"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the recommendation"
    )
    priority_score: float = Field(
        ..., ge=0.0, le=1.0, description="Priority ranking (0-1)"
    )
    current_reference_count: int = Field(
        0, ge=0, description="Current number of references"
    )
    suggested_search_terms: List[str] = Field(
        default_factory=list, description="Recommended search queries"
    )

    def derive(self, processor_id: str, **kwargs: Any) -> "IFDiscoveryIntentArtifact":
        """
        Create a derived discovery intent artifact.

        Rule #4: Under 60 lines.
        Rule #7: Explicit return type.
        """
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )
