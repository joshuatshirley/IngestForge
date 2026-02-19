"""
IF-Protocol Enrichment Processors.

Migration - Enrichment Parity
Provides native IF-Protocol processors for entity extraction,
embedding generation, and summary generation.

Follows NASA JPL Power of Ten rules.
"""

import uuid
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFChunkArtifact,
    IFFailureArtifact,
)

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_BATCH_SIZE = 100
MAX_ENTITIES_PER_CHUNK = 100
MAX_EMBEDDING_DIM = 1536
MAX_SUMMARY_LENGTH = 500
MAX_CONTENT_SIZE = 100_000  # 100KB


class IFEntityProcessor(IFProcessor):
    """
    IF-Protocol processor for entity extraction.

    Extracts named entities from chunk content using spaCy
    with regex fallback.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        processor_id: Optional[str] = None,
        version: str = "1.0.0",
        use_spacy: bool = True,
        model_name: str = "en_core_web_lg",
        min_confidence: float = 0.7,
    ):
        """
        Initialize the entity processor.

        Args:
            processor_id: Optional custom ID. Defaults to 'if-entity-extractor'.
            version: SemVer version string.
            use_spacy: Whether to use spaCy (fallback to regex if False).
            model_name: spaCy model name.
            min_confidence: Minimum confidence threshold.
        """
        self._processor_id = processor_id or "if-entity-extractor"
        self._version = version
        self._use_spacy = use_spacy
        self._model_name = model_name
        self._min_confidence = min_confidence
        self._extractor: Any = None

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
        return ["enrich.entities", "ner"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 256  # spaCy model

    def _get_extractor(self) -> Any:
        """Lazy-load entity extractor."""
        if self._extractor is None:
            from ingestforge.enrichment.entities import EntityExtractor

            self._extractor = EntityExtractor(
                use_spacy=self._use_spacy,
                model_name=self._model_name,
                min_confidence=self._min_confidence,
            )
        return self._extractor

    def is_available(self) -> bool:
        """
        Check if processor is available.

        Rule #7: Check return values.
        """
        try:
            extractor = self._get_extractor()
            return extractor.is_available()
        except ImportError:
            return False

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Extract entities from chunk content.

        Rule #4: Method should be < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact (must be IFChunkArtifact).

        Returns:
            IFChunkArtifact with entities in metadata, or IFFailureArtifact.
        """
        if not isinstance(artifact, IFChunkArtifact):
            return self._create_failure(
                artifact,
                f"IFEntityProcessor requires IFChunkArtifact, got {type(artifact).__name__}",
            )

        content = artifact.content
        if len(content) > MAX_CONTENT_SIZE:
            content = content[:MAX_CONTENT_SIZE]

        try:
            extractor = self._get_extractor()
            entities = extractor.extract_structured(content)

            # Limit entities per JPL Rule #2
            entities = entities[:MAX_ENTITIES_PER_CHUNK]

            # Convert to serializable format
            entities_data = [
                {
                    "text": e.text,
                    "label": e.label,
                    "start": e.start_char,
                    "end": e.end_char,
                    "confidence": e.confidence,
                }
                for e in entities
            ]

            return self._create_enriched_artifact(
                artifact,
                {"entities": entities_data, "entity_count": len(entities_data)},
            )

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return self._create_failure(artifact, str(e))

    def _create_enriched_artifact(
        self, source: IFChunkArtifact, enrichment: Dict[str, Any]
    ) -> IFChunkArtifact:
        """Create enriched IFChunkArtifact with new metadata."""
        new_metadata = dict(source.metadata)
        new_metadata.update(enrichment)
        new_metadata["enriched_by"] = self._processor_id

        return IFChunkArtifact(
            artifact_id=f"{source.artifact_id}-entities-{uuid.uuid4().hex[:8]}",
            document_id=source.document_id,
            content=source.content,
            chunk_index=source.chunk_index,
            total_chunks=source.total_chunks,
            parent_id=source.artifact_id,
            root_artifact_id=source.effective_root_id,
            lineage_depth=source.lineage_depth + 1,
            provenance=source.provenance + [self._processor_id],
            metadata=new_metadata,
        )

    def _create_failure(
        self, artifact: IFArtifact, error_message: str
    ) -> IFFailureArtifact:
        """Create IFFailureArtifact for error cases."""
        return IFFailureArtifact(
            artifact_id=f"{artifact.artifact_id}-entities-failed",
            error_message=error_message,
            failed_processor_id=self._processor_id,
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.effective_root_id,
            lineage_depth=artifact.lineage_depth + 1,
            provenance=artifact.provenance + [self._processor_id],
        )

    def teardown(self) -> bool:
        """Perform resource cleanup."""
        if self._extractor is not None:
            if hasattr(self._extractor, "clear_cache"):
                self._extractor.clear_cache()
            self._extractor = None
        return True


class IFEmbeddingProcessor(IFProcessor):
    """
    IF-Protocol processor for embedding generation.

    Generates vector embeddings for chunk content using
    sentence-transformers.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        processor_id: Optional[str] = None,
        version: str = "1.0.0",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the embedding processor.

        Args:
            processor_id: Optional custom ID. Defaults to 'if-embedding-generator'.
            version: SemVer version string.
            model_name: Sentence transformer model name.
        """
        self._processor_id = processor_id or "if-embedding-generator"
        self._version = version
        self._model_name = model_name
        self._generator: Any = None

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
        return ["enrich.embeddings", "vectorization"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 512  # sentence-transformers model

    def _get_generator(self) -> Any:
        """Lazy-load embedding generator."""
        if self._generator is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._generator = SentenceTransformer(self._model_name)
            except ImportError:
                self._generator = None
        return self._generator

    def is_available(self) -> bool:
        """
        Check if processor is available.

        Rule #7: Check return values.
        """
        try:
            generator = self._get_generator()
            return generator is not None
        except Exception:
            return False

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Generate embeddings for chunk content.

        Rule #4: Method should be < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact (must be IFChunkArtifact).

        Returns:
            IFChunkArtifact with embedding in metadata, or IFFailureArtifact.
        """
        if not isinstance(artifact, IFChunkArtifact):
            return self._create_failure(
                artifact,
                f"IFEmbeddingProcessor requires IFChunkArtifact, got {type(artifact).__name__}",
            )

        content = artifact.content
        if len(content) > MAX_CONTENT_SIZE:
            content = content[:MAX_CONTENT_SIZE]

        try:
            generator = self._get_generator()
            if generator is None:
                return self._create_failure(
                    artifact,
                    "Embedding generator not available (sentence-transformers not installed)",
                )

            # Generate embedding
            embedding = generator.encode(content, convert_to_numpy=True)
            embedding_list = embedding.tolist()

            # Validate dimension limit
            if len(embedding_list) > MAX_EMBEDDING_DIM:
                embedding_list = embedding_list[:MAX_EMBEDDING_DIM]

            return self._create_enriched_artifact(
                artifact,
                {
                    "embedding": embedding_list,
                    "embedding_model": self._model_name,
                    "embedding_dim": len(embedding_list),
                },
            )

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._create_failure(artifact, str(e))

    def _create_enriched_artifact(
        self, source: IFChunkArtifact, enrichment: Dict[str, Any]
    ) -> IFChunkArtifact:
        """Create enriched IFChunkArtifact with new metadata."""
        new_metadata = dict(source.metadata)
        new_metadata.update(enrichment)
        new_metadata["enriched_by"] = self._processor_id

        return IFChunkArtifact(
            artifact_id=f"{source.artifact_id}-embed-{uuid.uuid4().hex[:8]}",
            document_id=source.document_id,
            content=source.content,
            chunk_index=source.chunk_index,
            total_chunks=source.total_chunks,
            parent_id=source.artifact_id,
            root_artifact_id=source.effective_root_id,
            lineage_depth=source.lineage_depth + 1,
            provenance=source.provenance + [self._processor_id],
            metadata=new_metadata,
        )

    def _create_failure(
        self, artifact: IFArtifact, error_message: str
    ) -> IFFailureArtifact:
        """Create IFFailureArtifact for error cases."""
        return IFFailureArtifact(
            artifact_id=f"{artifact.artifact_id}-embed-failed",
            error_message=error_message,
            failed_processor_id=self._processor_id,
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.effective_root_id,
            lineage_depth=artifact.lineage_depth + 1,
            provenance=artifact.provenance + [self._processor_id],
        )

    def teardown(self) -> bool:
        """Perform resource cleanup."""
        self._generator = None
        return True


class IFSummaryProcessor(IFProcessor):
    """
    IF-Protocol processor for summary generation.

    Generates concise summaries for chunk content.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        processor_id: Optional[str] = None,
        version: str = "1.0.0",
        max_length: int = MAX_SUMMARY_LENGTH,
    ):
        """
        Initialize the summary processor.

        Args:
            processor_id: Optional custom ID. Defaults to 'if-summary-generator'.
            version: SemVer version string.
            max_length: Maximum summary length in characters.
        """
        self._processor_id = processor_id or "if-summary-generator"
        self._version = version
        self._max_length = min(max_length, MAX_SUMMARY_LENGTH)
        self._generator: Any = None

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
        return ["enrich.summary", "summarization"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 128

    def _get_generator(self) -> Any:
        """Lazy-load summary generator."""
        if self._generator is None:
            try:
                from ingestforge.enrichment.summary import SummaryGenerator
                from ingestforge.core.config import Config

                # Create minimal config for summary generator
                config = Config()
                self._generator = SummaryGenerator(config)
            except (ImportError, Exception):
                self._generator = None
        return self._generator

    def is_available(self) -> bool:
        """
        Check if processor is available.

        Rule #7: Check return values.
        """
        # Extractive fallback is always available
        return True

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Generate summary for chunk content.

        Rule #4: Method should be < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact (must be IFChunkArtifact).

        Returns:
            IFChunkArtifact with summary in metadata, or IFFailureArtifact.
        """
        if not isinstance(artifact, IFChunkArtifact):
            return self._create_failure(
                artifact,
                f"IFSummaryProcessor requires IFChunkArtifact, got {type(artifact).__name__}",
            )

        content = artifact.content
        if len(content) > MAX_CONTENT_SIZE:
            content = content[:MAX_CONTENT_SIZE]

        try:
            summary = self._generate_summary(content)

            return self._create_enriched_artifact(
                artifact, {"summary": summary, "summary_length": len(summary)}
            )

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return self._create_failure(artifact, str(e))

    def _generate_summary(self, content: str) -> str:
        """
        Generate summary using available method.

        Rule #4: Function < 60 lines.
        """
        generator = self._get_generator()

        if generator is not None:
            try:
                return generator.summarize(content)[: self._max_length]
            except Exception:
                pass  # Fall through to extractive

        # Extractive fallback
        return self._extractive_summary(content)

    def _extractive_summary(self, content: str) -> str:
        """
        Generate extractive summary (first sentences).

        Rule #4: Function < 60 lines.
        """
        sentences = content.split(".")
        summary_parts = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current_length + len(sentence) > self._max_length:
                break
            summary_parts.append(sentence)
            current_length += len(sentence) + 2  # Account for ". "

        if not summary_parts:
            return content[: self._max_length]

        return ". ".join(summary_parts) + "."

    def _create_enriched_artifact(
        self, source: IFChunkArtifact, enrichment: Dict[str, Any]
    ) -> IFChunkArtifact:
        """Create enriched IFChunkArtifact with new metadata."""
        new_metadata = dict(source.metadata)
        new_metadata.update(enrichment)
        new_metadata["enriched_by"] = self._processor_id

        return IFChunkArtifact(
            artifact_id=f"{source.artifact_id}-summary-{uuid.uuid4().hex[:8]}",
            document_id=source.document_id,
            content=source.content,
            chunk_index=source.chunk_index,
            total_chunks=source.total_chunks,
            parent_id=source.artifact_id,
            root_artifact_id=source.effective_root_id,
            lineage_depth=source.lineage_depth + 1,
            provenance=source.provenance + [self._processor_id],
            metadata=new_metadata,
        )

    def _create_failure(
        self, artifact: IFArtifact, error_message: str
    ) -> IFFailureArtifact:
        """Create IFFailureArtifact for error cases."""
        return IFFailureArtifact(
            artifact_id=f"{artifact.artifact_id}-summary-failed",
            error_message=error_message,
            failed_processor_id=self._processor_id,
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.effective_root_id,
            lineage_depth=artifact.lineage_depth + 1,
            provenance=artifact.provenance + [self._processor_id],
        )

    def teardown(self) -> bool:
        """Perform resource cleanup."""
        self._generator = None
        return True
