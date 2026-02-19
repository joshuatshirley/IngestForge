"""
Chunk Summary Generator for IngestForge.

Migrated to IFProcessor interface.
TASK-013: Migrated from ChunkRecord to IFChunkArtifact.
Generates focused one-liner summaries for each chunk using local LLM (Ollama/qwen2.5:14b).
Falls back to extractive summarization when LLM is unavailable.

Migration Notes (TASK-013):
    - All internal processing now uses IFChunkArtifact
    - Deprecated methods convert ChunkRecord → IFChunkArtifact → ChunkRecord
    - Main process() method works with IFChunkArtifact natively

NASA JPL Power of Ten compliant.
"""

import re
import warnings
from typing import Any, List, Optional, TYPE_CHECKING

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFFailureArtifact
from ingestforge.shared.lazy_imports import lazy_property

if TYPE_CHECKING:
    from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_CONTENT_LENGTH = 2000
MAX_SUMMARY_LENGTH = 200


class SummaryGenerator(IFProcessor):
    """
    Generate focused summaries for chunks using local LLM.

    Implements IFProcessor interface.
    Uses Ollama with qwen2.5:14b by default (configured in config.yaml).
    Falls back to extractive summarization when LLM is unavailable.

    Summaries are designed to be:
    - One sentence (15-25 words)
    - Specific and informative (not generic)
    - Focused on the key point of the chunk

    Rule #9: Complete type hints.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize summary generator.

        Args:
            config: IngestForge configuration (provides LLM settings)
        """
        self.config = config
        self._availability_checked = False
        self._is_available = False
        self._version = "2.0.0"

    # -------------------------------------------------------------------------
    # IFProcessor Interface Implementation
    # -------------------------------------------------------------------------

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "summary-generator"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Capabilities provided by this processor."""
        return ["summarization", "text-compression", "enrichment"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        # Higher if LLM is used
        return 100 if self._is_available else 50

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process an artifact to generate a summary.

        Implements IFProcessor.process().
        Rule #4: Function < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact (must be IFChunkArtifact).

        Returns:
            Derived IFChunkArtifact with summary in metadata.
        """
        # Validate input type
        if not isinstance(artifact, IFChunkArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-summary-failure",
                error_message=(
                    f"SummaryGenerator requires IFChunkArtifact, "
                    f"got {type(artifact).__name__}"
                ),
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
            )

        # Skip if already has a summary in metadata
        existing_summary = artifact.metadata.get("summary")
        if existing_summary:
            return artifact

        # Skip empty content
        if not artifact.content or not artifact.content.strip():
            new_metadata = dict(artifact.metadata)
            new_metadata["summary"] = ""
            return artifact.derive(
                self.processor_id,
                artifact_id=f"{artifact.artifact_id}-summary",
                metadata=new_metadata,
            )

        # Generate summary
        try:
            summary = self._generate_summary(artifact.content)
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            summary = self._generate_extractive_summary(artifact.content)

        # Build updated metadata
        new_metadata = dict(artifact.metadata)
        new_metadata["summary"] = summary
        new_metadata["summary_generator_version"] = self.version
        new_metadata["summary_method"] = "llm" if self.is_available() else "extractive"

        # Return derived artifact
        return artifact.derive(
            self.processor_id,
            artifact_id=f"{artifact.artifact_id}-summary",
            metadata=new_metadata,
        )

    def is_available(self) -> bool:
        """
        Check if LLM is available for summary generation.

        Implements IFProcessor.is_available().

        Returns:
            True if LLM is available, False otherwise (will use fallback)
        """
        if self._availability_checked:
            return self._is_available

        try:
            client = self.llm_client
            if client is None:
                self._is_available = False
            elif hasattr(client, "is_available"):
                self._is_available = client.is_available()
            else:
                self._is_available = True
        except Exception:
            self._is_available = False

        self._availability_checked = True
        return self._is_available

    def teardown(self) -> bool:
        """
        Clean up resources.

        Implements IFProcessor.teardown().

        Returns:
            True (no resources to clean up).
        """
        self._availability_checked = False
        self._is_available = False
        return True

    # -------------------------------------------------------------------------
    # LLM Client
    # -------------------------------------------------------------------------

    @lazy_property
    def llm_client(self) -> Any:
        """Lazy-load LLM client (prefers Ollama)."""
        try:
            from ingestforge.llm.factory import get_best_available_client

            client = get_best_available_client(self.config)
            if client:
                logger.info(f"SummaryGenerator using LLM: {client.model_name}")
            return client
        except Exception as e:
            logger.warning(f"Could not load LLM client for summaries: {e}")
            return None

    # -------------------------------------------------------------------------
    # Summary Generation Logic
    # -------------------------------------------------------------------------

    def _generate_summary(self, content: str) -> str:
        """
        Generate summary using LLM or extractive fallback.

        Rule #4: Function < 60 lines.

        Args:
            content: Text content to summarize

        Returns:
            Summary string
        """
        if self.is_available():
            return self._generate_llm_summary(content)
        return self._generate_extractive_summary(content)

    def _generate_llm_summary(self, content: str) -> str:
        """
        Generate focused summary using LLM.

        Rule #4: Function < 60 lines.

        Args:
            content: Text content to summarize

        Returns:
            One-sentence summary (15-25 words)
        """
        from ingestforge.llm.base import GenerationConfig

        # Truncate very long content to fit context window
        truncated = content[:MAX_CONTENT_LENGTH]
        if len(content) > MAX_CONTENT_LENGTH:
            truncated += "..."

        system_prompt = (
            "You are a precise summarizer. Generate a ONE sentence summary "
            "(15-25 words) that captures the key point of the text. "
            "Be specific and informative, not generic. "
            "Do not start with 'This text' or 'The passage'."
        )

        user_prompt = f"Summarize this text in ONE sentence:\n\n{truncated}"

        try:
            summary = self.llm_client.generate_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                config=GenerationConfig(
                    max_tokens=60,
                    temperature=0.3,
                ),
            )

            # Clean up the summary
            summary = self._clean_summary(summary)
            return summary

        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")
            return self._generate_extractive_summary(content)

    def _generate_extractive_summary(self, content: str) -> str:
        """
        Generate extractive summary when LLM is unavailable.

        Rule #4: Function < 60 lines.

        Args:
            content: Text content to summarize

        Returns:
            First informative sentence from content
        """
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", content)

        for sentence in sentences:
            sentence = sentence.strip()

            # Skip very short sentences (likely headers or fragments)
            if len(sentence) < 30:
                continue

            # Skip very long sentences
            if len(sentence) > MAX_SUMMARY_LENGTH:
                # Truncate to max length and add ellipsis
                return sentence[: MAX_SUMMARY_LENGTH - 3].rsplit(" ", 1)[0] + "..."

            # Skip sentences that look like headers
            if sentence.isupper() or sentence.startswith("#"):
                continue

            # Found a good sentence
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            return sentence

        # Fallback: truncate content
        if len(content) > 150:
            return content[:147].rsplit(" ", 1)[0] + "..."
        return content.strip()

    def _clean_summary(self, summary: str) -> str:
        """
        Clean up LLM-generated summary.

        Rule #4: Function < 60 lines.

        Args:
            summary: Raw summary from LLM

        Returns:
            Cleaned summary string
        """
        # Strip whitespace
        summary = summary.strip()

        # Remove surrounding quotes if present
        if (summary.startswith('"') and summary.endswith('"')) or (
            summary.startswith("'") and summary.endswith("'")
        ):
            summary = summary[1:-1].strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "Summary:",
            "Summary -",
            "Here is a summary:",
            "The summary is:",
            "One sentence summary:",
        ]
        for prefix in prefixes_to_remove:
            if summary.lower().startswith(prefix.lower()):
                summary = summary[len(prefix) :].strip()

        # Ensure ends with punctuation
        if summary and not summary.endswith((".", "!", "?")):
            summary += "."

        return summary

    # -------------------------------------------------------------------------
    # Legacy API (Backward Compatibility)
    # -------------------------------------------------------------------------

    def enrich_chunk(self, chunk: "ChunkRecord") -> "ChunkRecord":
        """
        Add focused summary to chunk.

        .. deprecated:: 2.0.0
            Use :meth:`process` with IFChunkArtifact instead.

        TASK-013: Internally converts to IFChunkArtifact for processing.

        Args:
            chunk: ChunkRecord to enrich

        Returns:
            ChunkRecord with summary field populated
        """
        warnings.warn(
            "enrich_chunk() is deprecated. Use process() with IFChunkArtifact instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Convert ChunkRecord → IFChunkArtifact
        artifact = IFChunkArtifact.from_chunk_record(chunk)

        # Process using new pipeline
        result = self.process(artifact)

        # If processing failed, return original chunk
        if isinstance(result, IFFailureArtifact):
            logger.warning(f"Summary generation failed: {result.error_message}")
            chunk.metadata["summary"] = self._generate_extractive_summary(chunk.content)
            return chunk

        # Convert back to ChunkRecord
        return result.to_chunk_record()

    def enrich_batch(
        self,
        chunks: List["ChunkRecord"],
        batch_size: Optional[int] = None,
        continue_on_error: bool = True,
        **kwargs: Any,
    ) -> List["ChunkRecord"]:
        """
        Batch process chunks with progress logging.

        .. deprecated:: 2.0.0
            Use :meth:`process` with IFChunkArtifact instead.

        TASK-013: Internally converts to IFChunkArtifact for processing.

        Args:
            chunks: List of ChunkRecords to enrich
            batch_size: Not used (processes one at a time for LLM)
            continue_on_error: If True, continue on individual failures
            **kwargs: Additional arguments (ignored)

        Returns:
            List of enriched ChunkRecords
        """
        warnings.warn(
            "enrich_batch() is deprecated. Use process() with IFChunkArtifact instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if not chunks:
            return chunks

        if not self.is_available():
            logger.info(
                f"LLM not available, using extractive summaries for {len(chunks)} chunks"
            )
        else:
            logger.info(f"Generating LLM summaries for {len(chunks)} chunks")

        enriched: List["ChunkRecord"] = []
        for i, chunk in enumerate(chunks):
            try:
                # Convert ChunkRecord → IFChunkArtifact
                artifact = IFChunkArtifact.from_chunk_record(chunk)

                # Process using new pipeline
                result = self.process(artifact)

                # Handle failure
                if isinstance(result, IFFailureArtifact):
                    if continue_on_error:
                        logger.warning(
                            f"Failed to summarize chunk {chunk.chunk_id}: {result.error_message}"
                        )
                        enriched.append(chunk)
                    else:
                        raise RuntimeError(result.error_message)
                else:
                    # Convert back to ChunkRecord
                    enriched.append(result.to_chunk_record())

                # Progress logging every 10 chunks
                if (i + 1) % 10 == 0:
                    logger.debug(f"Generated summaries: {i + 1}/{len(chunks)}")

            except Exception as e:
                if continue_on_error:
                    logger.warning(f"Failed to summarize chunk {chunk.chunk_id}: {e}")
                    enriched.append(chunk)
                else:
                    raise

        return enriched

    def get_metadata(self) -> dict[str, Any]:
        """Get enricher metadata for logging."""
        return {
            "name": self.__class__.__name__,
            "processor_id": self.processor_id,
            "version": self.version,
            "available": self.is_available(),
            "llm_available": self.is_available(),
            "fallback": "extractive",
        }

    def __repr__(self) -> str:
        available = "llm" if self.is_available() else "extractive"
        return f"SummaryGenerator({available})"
