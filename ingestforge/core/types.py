"""
Type definitions for IngestForge.

Provides TypedDicts, Protocols, and Literal types to replace
overly permissive `Any` type hints throughout the codebase.

Architecture Context
--------------------
These types serve as contracts between components:

    - ChunkMetadata: Structure for chunk-level metadata
    - SearchResultDict: Standard search result format
    - LLMKwargs: Type-safe LLM generation parameters
    - IEnricher: Protocol for enrichment plugins

Usage Pattern
-------------
Import specific types as needed:

    from ingestforge.core.types import ChunkMetadata, SearchResultDict

    def process_chunk(metadata: ChunkMetadata) -> None:
        if metadata.get("embedding"):
            # Process embedding
            pass

For Protocols:

    from ingestforge.core.types import IEnricher

    def run_enrichers(enrichers: List[IEnricher]) -> None:
        for enricher in enrichers:
            if enricher.is_available():
                enricher.enrich_chunk(chunk)
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)


# =============================================================================
# Configuration Literal Types
# =============================================================================

StorageBackendType = Literal["chromadb", "jsonl", "postgres", "sqlite"]
"""Supported storage backends."""

ChunkingStrategyType = Literal["semantic", "fixed", "paragraph", "header"]
"""Available chunking strategies."""

LLMProviderType = Literal["gemini", "openai", "anthropic", "ollama", "llamacpp"]
"""Supported LLM providers."""

OCREngineType = Literal["tesseract", "easyocr", "paddleocr", "none"]
"""Available OCR engines."""

PerformanceModeType = Literal["quality", "balanced", "speed", "mobile"]
"""Performance mode presets."""

RetrievalStrategyType = Literal["hybrid", "semantic", "bm25"]
"""Retrieval strategies for search."""


# =============================================================================
# Chunk-Related Types
# =============================================================================


class ChunkMetadata(TypedDict, total=False):
    """Metadata associated with a document chunk.

    This is the standard metadata structure attached to chunks
    during ingestion and enrichment.

    Attributes:
        embedding: Whether embedding has been computed.
        entities: Extracted named entities.
        concepts: Extracted concepts/topics.
        quality_score: Quality score from QA analysis.
        questions: Auto-generated questions for RAG.
        source_file: Original source document path.
        page_number: Page number in source document.
        chunk_index: Index of this chunk in document.
        header_path: Hierarchical header path (for header chunking).
        word_count: Number of words in chunk.
        char_count: Number of characters in chunk.
        author_id: Contributor identifier (TICKET-301).
        author_name: Contributor display name (TICKET-301).
    """

    embedding: bool
    entities: List[str]
    concepts: List[str]
    quality_score: float
    questions: List[str]
    source_file: str
    page_number: int
    chunk_index: int
    header_path: List[str]
    word_count: int
    char_count: int
    author_id: str
    author_name: str
    # HR Vertical
    resume_skills: List[str]
    resume_email: str
    resume_phone: str
    resume_education: List[str]
    # Real Estate Vertical
    property_address: str
    property_price: float
    property_sqft: float
    # Grant Vertical
    grant_id: str
    grant_amount: float
    grant_deadline: str
    # Cyber Vertical
    cyber_cve_id: str
    cyber_cvss_score: float
    cyber_affected_sw: List[str]
    # Educational Vertical
    edu_grade_level: str
    edu_subject: str
    edu_standards: List[str]
    # Manufacturing Vertical
    mfg_part_number: str
    mfg_maintenance_cycle: str
    mfg_error_codes: List[str]
    # Disaster Response Vertical
    disaster_incident_type: str
    disaster_coordinates: str
    disaster_urgency: str
    # Political Vertical
    political_candidate: str
    political_vote: str
    political_donors: List[str]
    # Wellness Vertical
    wellness_calories: float
    wellness_macros: Dict[str, float]
    wellness_allergens: List[str]
    # Spiritual Vertical
    spiritual_citation: str
    spiritual_book: str
    spiritual_themes: List[str]
    # Museum Vertical
    museum_accession_id: str
    museum_artist: str
    museum_era: str
    museum_medium: str
    # Bio/Lab Vertical
    bio_exp_id: str
    bio_formulas: List[str]
    bio_protocol: str
    # Automotive Vertical
    auto_vin: str
    auto_part_numbers: List[str]
    auto_restoration_status: str
    # Gaming Vertical
    gaming_patch_version: str
    gaming_characters: List[str]
    gaming_stat_changes: List[str]
    # AI Safety Vertical
    ai_model_name: str
    ai_param_count: str
    ai_safety_benchmarks: Dict[str, float]
    # Urban Planning Vertical
    urban_zoning_code: str
    urban_far_ratio: float
    urban_density_target: str
    # Unstructured-style spatial and element metadata
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    table_html: str
    element_type: str


class ChunkDict(TypedDict, total=False):
    """Full chunk representation including content and metadata.

    Attributes:
        id: Unique chunk identifier.
        content: Text content of the chunk.
        document_id: Parent document identifier.
        metadata: Chunk metadata (see ChunkMetadata).
        embedding: Optional embedding vector.
    """

    id: str
    content: str
    document_id: str
    metadata: ChunkMetadata
    embedding: List[float]


# =============================================================================
# Search-Related Types
# =============================================================================


class SearchResultDict(TypedDict):
    """Standard search result format.

    This is the common format returned by all search operations.

    Attributes:
        chunk_id: Unique identifier of the matched chunk.
        content: Text content of the chunk.
        score: Relevance score (higher is better).
        document_id: Parent document identifier.
        metadata: Chunk metadata.
    """

    chunk_id: str
    content: str
    score: float
    document_id: str
    metadata: ChunkMetadata


class QueryResultDict(TypedDict, total=False):
    """Complete query result with answer and sources.

    Attributes:
        answer: Generated answer text.
        sources: List of source chunks used.
        confidence: Confidence score for the answer.
        model: Model used for generation.
        tokens_used: Token count for the request.
    """

    answer: str
    sources: List[SearchResultDict]
    confidence: float
    model: str
    tokens_used: int


# =============================================================================
# LLM-Related Types
# =============================================================================


class LLMKwargs(TypedDict, total=False):
    """Type-safe LLM generation parameters.

    Use instead of **kwargs: Any in LLM methods.

    Attributes:
        stop: Stop sequences for generation.
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling parameter.
        presence_penalty: Penalty for new topics.
        frequency_penalty: Penalty for repetition.
    """

    stop: List[str]
    temperature: float
    max_tokens: int
    top_p: float
    stream: bool
    presence_penalty: float
    frequency_penalty: float


class LLMResponseDict(TypedDict, total=False):
    """LLM response structure.

    Attributes:
        content: Generated text content.
        model: Model that generated the response.
        tokens_input: Input token count.
        tokens_output: Output token count.
        finish_reason: Why generation stopped.
    """

    content: str
    model: str
    tokens_input: int
    tokens_output: int
    finish_reason: str


# =============================================================================
# Document-Related Types
# =============================================================================


class DocumentMetadata(TypedDict, total=False):
    """Document-level metadata.

    Attributes:
        title: Document title.
        author: Document author.
        created_at: Creation timestamp (ISO format).
        modified_at: Last modification timestamp.
        file_type: File extension/type.
        file_size: File size in bytes.
        page_count: Number of pages.
        word_count: Total word count.
        chunk_count: Number of chunks after processing.
        hash: Content hash for deduplication.
    """

    title: str
    author: str
    created_at: str
    modified_at: str
    file_type: str
    file_size: int
    page_count: int
    word_count: int
    chunk_count: int
    hash: str


class DocumentDict(TypedDict, total=False):
    """Full document representation.

    Attributes:
        id: Unique document identifier.
        name: Document filename.
        path: Original file path.
        content: Full text content.
        metadata: Document metadata.
        chunks: List of chunk IDs.
    """

    id: str
    name: str
    path: str
    content: str
    metadata: DocumentMetadata
    chunks: List[str]


# =============================================================================
# Analysis-Related Types
# =============================================================================


class EntityDict(TypedDict):
    """Extracted named entity.

    Attributes:
        text: Entity text.
        type: Entity type (PERSON, ORG, LOC, etc.).
        start: Start character offset.
        end: End character offset.
        confidence: Extraction confidence score.
    """

    text: str
    type: str
    start: int
    end: int
    confidence: float


class TopicDict(TypedDict, total=False):
    """Extracted topic/concept.

    Attributes:
        name: Topic name.
        score: Relevance score.
        keywords: Associated keywords.
        document_count: Documents containing this topic.
    """

    name: str
    score: float
    keywords: List[str]
    document_count: int


# =============================================================================
# Database-Related Types
# =============================================================================


class DatabaseRowDict(TypedDict, total=False):
    """Generic database row type.

    Used for SELECT results where column names vary.
    Provides basic structure validation.
    """

    id: Union[int, str]


# =============================================================================
# Protocol Definitions
# =============================================================================


@runtime_checkable
class IEnricher(Protocol):
    """Protocol for enrichment plugins.

    Enrichers add metadata to chunks during processing.

    Example:
        class EntityEnricher:
            def enrich_chunk(self, chunk: ChunkDict) -> ChunkDict:
                entities = self._extract_entities(chunk["content"])
                chunk["metadata"]["entities"] = entities
                return chunk

            def is_available(self) -> bool:
                return self._nlp_model is not None
    """

    def enrich_chunk(self, chunk: Any) -> Any:
        """Enrich a chunk with additional metadata."""
        ...

    def is_available(self) -> bool:
        """Check if enricher is available/configured."""
        ...


@runtime_checkable
class IChunker(Protocol):
    """Protocol for chunking implementations.

    Chunkers split documents into smaller pieces for processing.
    """

    def chunk(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[ChunkDict]:
        """Split content into chunks."""
        ...

    @property
    def strategy(self) -> ChunkingStrategyType:
        """Return the chunking strategy name."""
        ...


@runtime_checkable
class IStorage(Protocol):
    """Protocol for storage backends.

    Storage implementations handle persistence of documents and chunks.
    """

    def store_document(self, document: DocumentDict) -> str:
        """Store a document, return its ID."""
        ...

    def get_document(self, document_id: str) -> Optional[DocumentDict]:
        """Retrieve a document by ID."""
        ...

    def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[SearchResultDict]:
        """Search for relevant chunks."""
        ...


@runtime_checkable
class ILLMProvider(Protocol):
    """Protocol for LLM provider implementations.

    Providers handle communication with LLM APIs.
    """

    def generate(
        self,
        prompt: str,
        **kwargs: Any,  # LLMKwargs
    ) -> str:
        """Generate text from prompt."""
        ...

    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...
