"""
Dynamic Domain Enricher.

Migrated to IFProcessor interface.
Uses DomainRouter to classify chunks and apply specialized vertical refiners.

NASA JPL Power of Ten compliant.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFFailureArtifact
from ingestforge.enrichment.router import DomainRouter
from ingestforge.enrichment.utils import MetadataMerger

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_DOMAINS_PER_CHUNK = 5
MAX_REFINER_FAILURES = 3


class DynamicDomainEnricher(IFProcessor):
    """
    Enriches chunks by dynamically routing them to domain-specific refiners.

    Implements IFProcessor interface.
    Supports multi-domain enrichment for mixed-content chunks.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        min_score: int = 3,
        multi_domain_threshold: float = 0.7,
    ) -> None:
        """
        Initialize the dynamic domain enricher.

        Args:
            min_score: Minimum heuristic score to apply any refiner.
            multi_domain_threshold: If secondary domains score > threshold * primary, apply them too.
        """
        self.router = DomainRouter()
        self.merger = MetadataMerger()
        self.min_score = min_score
        self.multi_domain_threshold = multi_domain_threshold
        self._refiners: Dict[str, Any] = {}
        self._version = "2.0.0"

    # -------------------------------------------------------------------------
    # IFProcessor Interface Implementation
    # -------------------------------------------------------------------------

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "dynamic-domain-enricher"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Capabilities provided by this processor."""
        return ["domain-routing", "multi-domain-enrichment", "vertical-refinement"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 100  # Depends on loaded refiners

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """Process artifact by routing to domain-specific refiners. Rule #4: <60 lines."""
        # Validate input type
        if not isinstance(artifact, IFChunkArtifact):
            return self._create_failure(artifact)

        # Classify chunk content
        ranked = self.router.classify_chunk(artifact.content)
        if not ranked:
            return artifact

        # Filter to domains meeting minimum score
        valid_domains = [(d, s) for d, s in ranked if s >= self.min_score]
        if not valid_domains:
            return artifact

        # Determine target domains and apply refiners
        targets = self._determine_target_domains(valid_domains)
        merged_metadata, detected_domains = self._apply_refiners(artifact, targets)

        # Build updated metadata
        new_metadata = dict(artifact.metadata)
        new_metadata.update(merged_metadata)
        new_metadata["domain_enricher_version"] = self.version

        if detected_domains:
            new_metadata["detected_domains"] = detected_domains[:MAX_DOMAINS_PER_CHUNK]
            new_metadata["primary_domain"] = valid_domains[0][0]

        return artifact.derive(
            self.processor_id,
            artifact_id=f"{artifact.artifact_id}-domain",
            metadata=new_metadata,
        )

    def _create_failure(self, artifact: IFArtifact) -> IFFailureArtifact:
        """Create failure artifact for invalid input. Rule #4: Helper function."""
        return IFFailureArtifact(
            artifact_id=f"{artifact.artifact_id}-domain-failure",
            error_message=(
                f"DynamicDomainEnricher requires IFChunkArtifact, "
                f"got {type(artifact).__name__}"
            ),
            failed_processor_id=self.processor_id,
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.effective_root_id,
            lineage_depth=artifact.lineage_depth + 1,
            provenance=artifact.provenance + [self.processor_id],
        )

    def is_available(self) -> bool:
        """
        Dynamic domain enricher is always available.

        Implements IFProcessor.is_available().
        """
        return True

    def teardown(self) -> bool:
        """
        Clean up resources including cached refiners.

        Implements IFProcessor.teardown().
        """
        self._refiners.clear()
        return True

    # -------------------------------------------------------------------------
    # Domain Routing Logic
    # -------------------------------------------------------------------------

    def _determine_target_domains(
        self,
        valid_domains: List[tuple],
    ) -> List[str]:
        """
        Determine which domains to apply based on scores.

        Rule #4: Function < 60 lines.
        """
        primary_domain, primary_score = valid_domains[0]
        targets = [primary_domain]

        # Add secondary domains within threshold
        for domain, score in valid_domains[1:MAX_DOMAINS_PER_CHUNK]:
            if score >= (primary_score * self.multi_domain_threshold):
                targets.append(domain)

        return targets

    def _apply_refiners(
        self,
        artifact: IFChunkArtifact,
        targets: List[str],
    ) -> tuple:
        """
        Apply refiners to artifact and merge metadata.

        Rule #4: Function < 60 lines.
        """
        merged_metadata: Dict[str, Any] = {}
        detected_domains: List[str] = []
        failure_count = 0

        for domain in targets:
            if failure_count >= MAX_REFINER_FAILURES:
                logger.warning("Max refiner failures reached, stopping")
                break

            refiner = self._get_refiner(domain)
            if not refiner:
                continue

            try:
                # Create a mock chunk for the refiner
                mock_chunk = self._create_mock_chunk(artifact)
                enriched = refiner.enrich(mock_chunk)

                # Merge metadata
                if enriched.metadata:
                    merged_metadata = self.merger.merge(
                        merged_metadata,
                        enriched.metadata,
                    )
                detected_domains.append(domain)
            except Exception as e:
                logger.warning(f"Refiner for {domain} failed: {e}")
                failure_count += 1

        return merged_metadata, detected_domains

    def _create_mock_chunk(self, artifact: IFChunkArtifact) -> Any:
        """Create a mock chunk object for legacy refiners."""
        from ingestforge.chunking.semantic_chunker import ChunkRecord

        return ChunkRecord(
            chunk_id=artifact.artifact_id,
            document_id=artifact.document_id,
            content=artifact.content,
            chunk_index=artifact.chunk_index,
            total_chunks=artifact.total_chunks,
            metadata=dict(artifact.metadata),
        )

    def _get_refiner(self, domain: str) -> Optional[Any]:
        """
        Lazy load refiner for a domain.

        Rule #1: Dictionary dispatch.
        Rule #4: Function < 60 lines.
        """
        if domain in self._refiners:
            return self._refiners[domain]

        # Map domain keys to import paths and class names
        refiner_map = {
            "urban": ("ingestforge.enrichment.urban", "UrbanMetadataRefiner"),
            "ai_safety": (
                "ingestforge.enrichment.ai_safety",
                "AISafetyMetadataRefiner",
            ),
            "gaming": ("ingestforge.enrichment.gaming", "GamingMetadataRefiner"),
            "auto": ("ingestforge.enrichment.auto", "AutoMetadataRefiner"),
            "bio": ("ingestforge.enrichment.bio", "BioMetadataRefiner"),
            "museum": ("ingestforge.enrichment.museum", "MuseumMetadataRefiner"),
            "spiritual": (
                "ingestforge.enrichment.spiritual",
                "SpiritualMetadataRefiner",
            ),
            "wellness": ("ingestforge.enrichment.wellness", "WellnessMetadataRefiner"),
            "cyber": ("ingestforge.enrichment.cyber", "CyberMetadataRefiner"),
        }

        if domain not in refiner_map:
            return None

        try:
            module_path, class_name = refiner_map[domain]
            module = __import__(module_path, fromlist=[class_name])
            refiner_cls = getattr(module, class_name)
            refiner = refiner_cls()

            self._refiners[domain] = refiner
            return refiner
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not load refiner for {domain}: {e}")
            return None

    # -------------------------------------------------------------------------
    # Legacy API (Backward Compatibility)
    # -------------------------------------------------------------------------

    def enrich_chunk(self, chunk: Any) -> Any:
        """
        Classify chunk and apply domain-specific enrichment.

        .. deprecated:: 2.0.0
            Use :meth:`process` with IFChunkArtifact instead.
        """
        warnings.warn(
            "enrich_chunk() is deprecated. Use process() with IFChunkArtifact instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Classify content
        ranked = self.router.classify_chunk(chunk.content)
        if not ranked:
            return chunk

        valid_domains = [d for d in ranked if d[1] >= self.min_score]
        if not valid_domains:
            return chunk

        primary_domain, primary_score = valid_domains[0]

        # Identify target domains
        targets = [primary_domain]
        for domain, score in valid_domains[1:]:
            if score >= (primary_score * self.multi_domain_threshold):
                targets.append(domain)

        # Apply refiners
        original_metadata = (chunk.metadata or {}).copy()
        merged_metadata = original_metadata
        detected_domains: List[str] = []

        for domain in targets:
            refiner = self._get_refiner(domain)
            if not refiner:
                continue

            try:
                enriched = refiner.enrich(chunk)
                merged_metadata = self.merger.merge(
                    merged_metadata,
                    enriched.metadata or {},
                )
                detected_domains.append(domain)
            except Exception as e:
                logger.warning(f"Refiner for {domain} failed: {e}")

        # Update chunk
        chunk.metadata = merged_metadata
        if detected_domains:
            chunk.metadata["detected_domains"] = list(set(detected_domains))
            chunk.metadata["primary_domain"] = primary_domain

        return chunk
