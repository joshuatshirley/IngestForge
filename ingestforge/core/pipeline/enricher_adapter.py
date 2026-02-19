"""
Enricher Adapter for IF-Protocol Migration.

This module provides an adapter to bridge the legacy IEnricher interface
to the new IFProcessor contract, enabling gradual migration.

Convergence - Processor Unification
Follows NASA JPL Power of Ten rules.

IMPORTANT: The IEnricher interface is deprecated. New enrichers should
implement IFProcessor directly. See EntityExtractor and EmbeddingGenerator
for examples.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFFailureArtifact
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Legacy IEnricher Interface (DEPRECATED)
# =============================================================================


class EnrichmentError(Exception):
    """Raised when enrichment fails."""

    pass


class IEnricher(ABC):
    """
    DEPRECATED: Use IFProcessor from ingestforge.core.pipeline.interfaces instead.

    Legacy interface for chunk enrichers. This class is preserved for backward
    compatibility only. New enrichers should implement IFProcessor directly.

    Convergence - Processor Unification.

    See Also:
        - EntityExtractor: IFProcessor-based entity extraction
        - EmbeddingGenerator: IFProcessor-based embedding generation
        - IFEnricherAdapter: Wraps legacy IEnricher as IFProcessor
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Emit deprecation warning when IEnricher is subclassed."""
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"IEnricher is deprecated. Class '{cls.__name__}' should migrate to IFProcessor. "
            "Use IFEnricherAdapter for temporary compatibility.",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    def enrich_chunk(self, chunk: Any) -> Any:
        """
        Enrich a single chunk.

        Args:
            chunk: ChunkRecord to enrich

        Returns:
            The enriched ChunkRecord
        """
        pass

    def enrich_batch(
        self,
        chunks: List[Any],
        batch_size: Optional[int] = None,
        continue_on_error: bool = True,
    ) -> List[Any]:
        """
        Enrich multiple chunks.

        Args:
            chunks: List of ChunkRecords to enrich
            batch_size: Optional batch size for processing
            continue_on_error: If True, continue on failure

        Returns:
            List of enriched ChunkRecords
        """
        if not chunks:
            return chunks

        enriched = []
        for chunk in chunks:
            if continue_on_error:
                try:
                    enriched_chunk = self.enrich_chunk(chunk)
                    enriched.append(enriched_chunk)
                except Exception as e:
                    logger.warning(f"Failed to enrich chunk: {e}")
                    enriched.append(chunk)
            else:
                enriched_chunk = self.enrich_chunk(chunk)
                enriched.append(enriched_chunk)

        return enriched

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the enricher is available."""
        pass

    def get_metadata(self) -> dict:
        """Get enricher metadata."""
        return {
            "name": self.__class__.__name__,
            "available": self.is_available(),
            "version": getattr(self, "__version__", "unknown"),
        }

    def __repr__(self) -> str:
        """String representation of enricher."""
        available = "available" if self.is_available() else "unavailable"
        return f"{self.__class__.__name__}({available})"


class EnrichmentPipeline:
    """
    DEPRECATED: Use IFStage with IFProcessor-based enrichers instead.

    Pipeline for applying multiple enrichers in sequence.

    Convergence - Processor Unification.
    """

    def __init__(
        self, enrichers: List[IEnricher], skip_unavailable: bool = True
    ) -> None:
        """
        Initialize enrichment pipeline.

        Args:
            enrichers: List of enrichers to apply
            skip_unavailable: If True, skip unavailable enrichers
        """
        warnings.warn(
            "EnrichmentPipeline is deprecated. Use IFStage with IFProcessor-based enrichers.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.enrichers = enrichers
        self.skip_unavailable = skip_unavailable

        if skip_unavailable:
            self.active_enrichers = [e for e in enrichers if e.is_available()]
        else:
            self.active_enrichers = enrichers

    def enrich_batch(
        self, chunks: List[Any], batch_size: Optional[int] = None, **kwargs: Any
    ) -> List[Any]:
        """Apply all enrichers to chunks (alias for enrich)."""
        return self.enrich(chunks, batch_size=batch_size)

    def enrich(self, chunks: List[Any], batch_size: Optional[int] = None) -> List[Any]:
        """
        Apply all enrichers to chunks.

        Args:
            chunks: List of ChunkRecords to enrich
            batch_size: Optional batch size

        Returns:
            List of enriched ChunkRecords
        """
        result = chunks

        for enricher in self.active_enrichers:
            logger.info(f"Applying enricher: {enricher}")
            try:
                result = enricher.enrich_batch(
                    result,
                    batch_size=batch_size,
                    continue_on_error=self.skip_unavailable,
                )
            except Exception as e:
                logger.error(f"Enricher {enricher} failed: {e}")
                if not self.skip_unavailable:
                    raise EnrichmentError(f"Enrichment failed: {e}") from e

        return result

    def get_summary(self) -> dict:
        """Get summary of pipeline configuration."""
        return {
            "total_enrichers": len(self.enrichers),
            "active_enrichers": len(self.active_enrichers),
            "skip_unavailable": self.skip_unavailable,
            "enrichers": [e.get_metadata() for e in self.active_enrichers],
        }


# =============================================================================
# IFProcessor Adapter for Legacy Enrichers
# =============================================================================


class IFEnricherAdapter(IFProcessor):
    """
    Adapter that wraps a legacy IEnricher to work as an IFProcessor.

    This enables existing enrichers to participate in the IF-Protocol
    pipeline without requiring immediate rewrite.

    Rule #9: Complete type hints.

    Example:
        >>> from ingestforge.enrichment import EmbeddingGenerator
        >>> legacy_enricher = EmbeddingGenerator(config)
        >>> processor = IFEnricherAdapter(legacy_enricher)
        >>> result = processor.process(chunk_artifact)
    """

    def __init__(
        self,
        enricher: IEnricher,
        processor_id: Optional[str] = None,
        version: str = "1.0.0",
    ):
        """
        Initialize the adapter.

        Args:
            enricher: The legacy IEnricher to wrap.
            processor_id: Optional custom ID. Defaults to enricher class name.
            version: SemVer version string.
        """
        self._enricher = enricher
        self._processor_id = processor_id or f"enricher-{enricher.__class__.__name__}"
        self._version = version

        # Issue deprecation warning for legacy usage tracking
        warnings.warn(
            f"IEnricher '{enricher.__class__.__name__}' wrapped via adapter. "
            "Consider migrating to IFProcessor interface directly.",
            DeprecationWarning,
            stacklevel=2,
        )

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return self._processor_id

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    def is_available(self) -> bool:
        """
        Check if the wrapped enricher is available.

        Rule #7: Check return values.
        """
        return self._enricher.is_available()

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process an artifact using the wrapped enricher.

        Converts IFChunkArtifact to legacy chunk format, applies enrichment,
        and converts back to IFChunkArtifact.

        Rule #4: Method should be < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact (must be IFChunkArtifact).

        Returns:
            Enriched IFChunkArtifact or IFFailureArtifact on error.
        """
        if not isinstance(artifact, IFChunkArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-enrichment-failed",
                error_message=f"IFEnricherAdapter requires IFChunkArtifact, got {type(artifact).__name__}",
                failed_processor_id=self._processor_id,
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self._processor_id],
            )

        try:
            # Convert IFChunkArtifact to legacy chunk format
            legacy_chunk = self._to_legacy_chunk(artifact)

            # Apply enrichment
            enriched_chunk = self._enricher.enrich_chunk(legacy_chunk)

            # Convert back to IFChunkArtifact
            return self._from_legacy_chunk(artifact, enriched_chunk)

        except Exception as e:
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-enrichment-failed",
                error_message=f"Enrichment failed: {str(e)}",
                failed_processor_id=self._processor_id,
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self._processor_id],
            )

    def _to_legacy_chunk(self, artifact: IFChunkArtifact) -> Any:
        """
        Convert IFChunkArtifact to legacy chunk format.

        Creates a simple object with the expected legacy attributes.
        """

        class LegacyChunk:
            """Minimal legacy chunk representation."""

            pass

        chunk = LegacyChunk()
        chunk.chunk_id = artifact.artifact_id
        chunk.content = artifact.content
        chunk.document_id = artifact.document_id
        chunk.chunk_index = artifact.chunk_index
        chunk.total_chunks = artifact.total_chunks
        chunk.metadata = dict(artifact.metadata)

        # Copy any additional metadata fields the enricher might expect
        for key, value in artifact.metadata.items():
            if not hasattr(chunk, key):
                setattr(chunk, key, value)

        return chunk

    def _from_legacy_chunk(
        self, original: IFChunkArtifact, enriched: Any
    ) -> IFChunkArtifact:
        """
        Convert enriched legacy chunk back to IFChunkArtifact.

        Extracts enrichment data and creates a new derived artifact.
        """
        # Extract enrichment data from legacy chunk
        enrichment_data = {}

        # Common enrichment fields
        enrichment_fields = [
            "embedding",
            "entities",
            "concepts",
            "hypothetical_questions",
            "summary",
            "quality_score",
            "sentiment",
            "keywords",
            "topics",
        ]

        for field in enrichment_fields:
            if hasattr(enriched, field):
                value = getattr(enriched, field)
                if value is not None:
                    enrichment_data[field] = value

        # Merge with existing metadata
        new_metadata = dict(original.metadata)
        new_metadata.update(enrichment_data)

        # Create derived artifact with enrichment data
        return original.derive(
            self._processor_id,
            artifact_id=f"{original.artifact_id}-enriched",
            metadata=new_metadata,
        )

    def teardown(self) -> bool:
        """
        Perform resource cleanup.

        Rule #7: Check return values.
        """
        # Legacy enrichers don't have teardown, so always succeed
        return True

    @property
    def wrapped_enricher(self) -> IEnricher:
        """Access the wrapped enricher (for debugging/introspection)."""
        return self._enricher


def adapt_enricher(
    enricher: IEnricher, processor_id: Optional[str] = None
) -> IFEnricherAdapter:
    """
    Factory function to create an adapter for a legacy enricher.

    Args:
        enricher: The IEnricher to wrap.
        processor_id: Optional custom processor ID.

    Returns:
        IFEnricherAdapter wrapping the enricher.

    Example:
        >>> from ingestforge.enrichment import EmbeddingGenerator
        >>> processor = adapt_enricher(EmbeddingGenerator(config))
    """
    return IFEnricherAdapter(enricher, processor_id)


def adapt_enrichers(enrichers: List[IEnricher]) -> List[IFEnricherAdapter]:
    """
    Batch adapt multiple enrichers.

    Args:
        enrichers: List of IEnricher instances.

    Returns:
        List of IFEnricherAdapter instances.
    """
    return [adapt_enricher(e) for e in enrichers]
