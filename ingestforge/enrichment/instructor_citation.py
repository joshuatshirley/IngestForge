"""
Citation Metadata Enrichment via Instructor.

Migrated to IFProcessor interface.
Uses the instructor library to extract high-fidelity structured bibliographic
metadata from document content using LLMs.

NASA JPL Power of Ten compliant.
"""

import logging
import warnings
from typing import Any, List, Optional

from ingestforge.core.config import Config
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFFailureArtifact
from ingestforge.ingest.citation_metadata.pydantic_models import CitationMetadata
from ingestforge.llm.factory import get_llm_client
from ingestforge.llm.openai import OpenAIClient

logger = logging.getLogger(__name__)


class InstructorCitationEnricher(IFProcessor):
    """
    Enricher that extracts structured citation metadata using LLMs and Instructor.

    Implements IFProcessor interface.

    This processor typically runs on the first chunk of a document where
    bibliographic information (title, authors, DOI) is most likely to be found.

    Rule #9: Complete type hints.
    """

    def __init__(self, config: Config, model: Optional[str] = None) -> None:
        """
        Initialize the enricher.

        Args:
            config: IngestForge configuration
            model: Optional model override (defaults to config settings)
        """
        self.config = config
        self.client = get_llm_client(config)
        self.model = model or getattr(config.llm.openai, "model", "gpt-4o-mini")
        self._instructor_client: Any = None
        self._version = "2.0.0"

    # -------------------------------------------------------------------------
    # IFProcessor Interface Implementation
    # -------------------------------------------------------------------------

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "instructor-citation-enricher"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Capabilities provided by this processor."""
        return ["citation-extraction", "metadata-enrichment", "bibliographic"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 150  # LLM-based

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process an artifact to extract citation metadata.

        Implements IFProcessor.process().
        Rule #4: Function < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact (must be IFChunkArtifact).

        Returns:
            Derived IFChunkArtifact with citation metadata.
        """
        # Validate input type
        if not isinstance(artifact, IFChunkArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-citation-failure",
                error_message=(
                    f"InstructorCitationEnricher requires IFChunkArtifact, "
                    f"got {type(artifact).__name__}"
                ),
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
            )

        # Only attempt extraction on first chunk or if explicitly requested
        chunk_index = artifact.chunk_index
        force_extraction = artifact.metadata.get("force_citation_extraction", False)

        if chunk_index != 0 and not force_extraction:
            return artifact  # Skip non-first chunks

        if not self.instructor_client:
            return artifact  # Skip if instructor not available

        # Build updated metadata
        new_metadata = dict(artifact.metadata)

        try:
            metadata = self._extract_metadata(artifact.content)
            if metadata:
                new_metadata["citation"] = metadata.model_dump()
                new_metadata["citation_extractor_version"] = self.version
                logger.info(f"Extracted citation metadata: {metadata.title}")
        except Exception as e:
            logger.warning(
                f"Citation extraction failed for {artifact.artifact_id}: {e}"
            )
            return artifact  # Return unchanged on failure

        # Return derived artifact
        return artifact.derive(
            self.processor_id,
            artifact_id=f"{artifact.artifact_id}-citation",
            metadata=new_metadata,
        )

    def is_available(self) -> bool:
        """
        Check if instructor and a compatible LLM client are available.

        Implements IFProcessor.is_available().
        """
        try:
            import instructor  # noqa: F401

            return isinstance(self.client, OpenAIClient) and self.client.is_available()
        except ImportError:
            return False

    def teardown(self) -> bool:
        """
        Clean up resources.

        Implements IFProcessor.teardown().
        """
        self._instructor_client = None
        return True

    # -------------------------------------------------------------------------
    # Instructor Client
    # -------------------------------------------------------------------------

    @property
    def instructor_client(self) -> Any:
        """Lazy-load and patch the instructor client."""
        if self._instructor_client is not None:
            return self._instructor_client

        try:
            import instructor

            # Only support OpenAI for instructor patching
            if isinstance(self.client, OpenAIClient):
                self._instructor_client = instructor.patch(self.client.client)
                return self._instructor_client
            else:
                logger.debug(
                    f"Instructor enrichment skipped: client {type(self.client)} not supported"
                )
                return None
        except ImportError:
            logger.debug("Instructor library not installed.")
            return None

    # -------------------------------------------------------------------------
    # Extraction Logic
    # -------------------------------------------------------------------------

    def _extract_metadata(self, text: str) -> Optional[CitationMetadata]:
        """
        Internal helper to perform the instructor call.

        Rule #4: Function < 60 lines.
        """
        if not self.instructor_client:
            return None

        return self.instructor_client.chat.completions.create(
            model=self.model,
            response_model=CitationMetadata,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert bibliographic metadata extractor. "
                        "Extract accurate citation information from the provided text."
                    ),
                },
                {"role": "user", "content": text},
            ],
            validation_context={"text": text},
        )

    # -------------------------------------------------------------------------
    # Legacy API (Backward Compatibility)
    # -------------------------------------------------------------------------

    def enrich_chunk(self, chunk: Any) -> Any:
        """
        Extract citation metadata from the chunk.

        .. deprecated:: 2.0.0
            Use :meth:`process` with IFChunkArtifact instead.
        """
        warnings.warn(
            "enrich_chunk() is deprecated. Use process() with IFChunkArtifact instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Only attempt extraction on first chunk
        if getattr(chunk, "chunk_index", 0) != 0:
            if not chunk.metadata.get("force_citation_extraction"):
                return chunk

        if not self.instructor_client:
            return chunk

        try:
            metadata = self._extract_metadata(chunk.content)
            if metadata:
                chunk.metadata["citation"] = metadata.model_dump()
                if metadata.title and not getattr(chunk, "section_title", ""):
                    chunk.section_title = metadata.title
                logger.info(f"Extracted citation metadata: {metadata.title}")
        except Exception as e:
            logger.warning(f"Citation extraction failed for {chunk.chunk_id}: {e}")

        return chunk
