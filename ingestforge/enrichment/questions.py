"""
Hypothetical question generation for chunks.

Migrated to IFProcessor interface.
TASK-012: Migrated from ChunkRecord to IFChunkArtifact.
Generate questions that the chunk content could answer.
Useful for query expansion and retrieval improvement.

NASA JPL Power of Ten compliant.
"""

import warnings
from typing import Any, Dict, List, TYPE_CHECKING

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFFailureArtifact
from ingestforge.core.pipeline.registry import register_enricher
from ingestforge.shared.lazy_imports import lazy_property

if TYPE_CHECKING:
    from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_QUESTIONS = 10
MAX_CONTENT_LENGTH = 8000


@register_enricher(
    capabilities=["question-generation", "query-expansion"],
    priority=80,
)
class QuestionGenerator(IFProcessor):
    """
    Generate hypothetical questions for chunks.

    Implements IFProcessor interface.
    Uses LLM to generate questions that the chunk answers.
    Falls back to template-based generation if LLM unavailable.

    Rule #9: Complete type hints.
    """

    def __init__(self, config: Config, num_questions: int = 3) -> None:
        """
        Initialize the question generator.

        Args:
            config: IngestForge configuration.
            num_questions: Default number of questions to generate.
        """
        self.config = config
        self._num_questions = min(num_questions, MAX_QUESTIONS)
        self._version = "2.0.0"

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "question-generator"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Capabilities provided by this processor."""
        return ["question-generation", "query-expansion", "enrichment"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 50  # Lightweight unless LLM is used

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process an artifact to generate hypothetical questions.

        Implements IFProcessor.process().
        TASK-012: Migrated to use IFChunkArtifact directly.
        Rule #4: Function < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact (must be IFChunkArtifact).

        Returns:
            Derived IFChunkArtifact with questions in metadata.
        """
        # Validate input type
        if not isinstance(artifact, IFChunkArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-question-failure",
                error_message=(
                    f"QuestionGenerator requires IFChunkArtifact, "
                    f"got {type(artifact).__name__}"
                ),
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
            )

        # Generate questions using IFChunkArtifact directly
        questions = self._generate_questions(artifact, self._num_questions)

        # Build updated metadata
        new_metadata = dict(artifact.metadata)
        new_metadata["hypothetical_questions"] = questions
        new_metadata["question_count"] = len(questions)
        new_metadata["question_generator_version"] = self.version

        # Return derived artifact
        return artifact.derive(
            self.processor_id,
            artifact_id=f"{artifact.artifact_id}-questions",
            metadata=new_metadata,
        )

    def _get_content(self, artifact: IFChunkArtifact) -> str:
        """
        Get truncated content from artifact for question generation.

        TASK-012: Replaced ChunkRecord conversion with direct access.
        Rule #4: Helper function < 60 lines.

        Args:
            artifact: Chunk artifact to extract content from.

        Returns:
            Truncated content string.
        """
        return artifact.content[:MAX_CONTENT_LENGTH]

    def _get_metadata_field(
        self, artifact: IFChunkArtifact, field: str, default: str = ""
    ) -> str:
        """
        Get metadata field from artifact with default.

        TASK-012: Helper for metadata access.
        Rule #4: Helper function < 60 lines.
        Rule #9: Complete type hints.

        Args:
            artifact: Chunk artifact to extract metadata from.
            field: Metadata field name.
            default: Default value if field not present.

        Returns:
            Metadata field value or default.
        """
        value = artifact.metadata.get(field, default)
        return str(value) if value is not None else default

    @lazy_property
    def llm_client(self) -> Any:
        """Lazy-load LLM client."""
        try:
            from ingestforge.llm.factory import get_llm_client

            return get_llm_client(self.config)
        except Exception as e:
            logger.warning(f"Could not load LLM client: {e}")
            return None

    def _generate_questions(
        self, artifact: IFChunkArtifact, num_questions: int = 3
    ) -> List[str]:
        """
        Generate questions for a chunk artifact.

        TASK-012: Migrated to use IFChunkArtifact directly.
        Rule #9: Complete type hints.

        Args:
            artifact: Chunk artifact to generate questions for.
            num_questions: Number of questions to generate.

        Returns:
            List of generated questions.
        """
        if self.llm_client:
            return self._generate_llm(artifact, num_questions)
        return self._generate_template(artifact, num_questions)

    def generate(self, chunk: "ChunkRecord", num_questions: int = 3) -> List[str]:
        """
        Generate questions for a legacy ChunkRecord.

        .. deprecated:: 3.0.0
            Use :meth:`_generate_questions` with IFChunkArtifact instead.

        Args:
            chunk: Legacy ChunkRecord to generate questions for.
            num_questions: Number of questions to generate.

        Returns:
            List of generated questions.
        """
        warnings.warn(
            "generate(ChunkRecord) is deprecated. "
            "Use process() with IFChunkArtifact instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Convert ChunkRecord to IFChunkArtifact for processing
        artifact = IFChunkArtifact.from_chunk_record(chunk)
        return self._generate_questions(artifact, num_questions)

    def _generate_llm(self, artifact: IFChunkArtifact, num_questions: int) -> List[str]:
        """
        Generate questions using LLM.

        TASK-012: Migrated to use IFChunkArtifact.
        Rule #1: Reduced nesting from 4 → 2 levels via extraction.
        Rule #4: Function <60 lines.
        Rule #9: Complete type hints.

        Args:
            artifact: Chunk artifact to generate questions for.
            num_questions: Number of questions to generate.

        Returns:
            List of generated questions.
        """
        content = self._get_content(artifact)[:2000]
        prompt = f"""Based on the following text, generate {num_questions} questions that this text could answer.
The questions should:
- Be natural and conversational
- Cover the main topics in the text
- Be specific enough to be answered by this content

Text:
{content}

Generate exactly {num_questions} questions, one per line:"""

        try:
            response = self.llm_client.generate(prompt)
            lines = response.strip().split("\n")
            questions = [
                self._process_question_line(line)
                for line in lines
                if self._process_question_line(line)
            ]
            return questions[:num_questions]
        except Exception as e:
            logger.warning(f"LLM question generation failed: {e}")
            return self._generate_template(artifact, num_questions)

    def _process_question_line(self, line: str) -> str:
        """
        Process and clean a single question line.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines

        Args:
            line: Raw question line from LLM

        Returns:
            Cleaned question string, or empty string if invalid
        """
        line = line.strip()
        if not line:
            return ""

        # Remove numbering if present
        if line[0].isdigit():
            line = line.lstrip("0123456789.)-] ")
        if "?" not in line:
            return ""

        # Extract just the question part
        return line[: line.index("?") + 1]

    def _generate_template(
        self, artifact: IFChunkArtifact, num_questions: int
    ) -> List[str]:
        """
        Generate questions using templates.

        TASK-012: Migrated to use IFChunkArtifact.
        Rule #1: Reduced nesting from 4 → 2 levels via extraction.
        Rule #4: Function <60 lines.
        Rule #9: Complete type hints.

        Args:
            artifact: Chunk artifact to generate questions for.
            num_questions: Number of questions to generate.

        Returns:
            List of generated questions.
        """
        chunk_type = self._get_metadata_field(artifact, "chunk_type", "content")

        generators: Dict[str, Any] = {
            "definition": self._generate_definition_questions,
            "procedure": self._generate_procedure_questions,
            "example": self._generate_example_questions,
        }

        generator = generators.get(chunk_type, self._generate_general_questions)
        questions = generator(artifact)
        return questions[:num_questions]

    def _generate_definition_questions(self, artifact: IFChunkArtifact) -> List[str]:
        """
        Generate definition-style questions.

        TASK-012: Migrated to use IFChunkArtifact.
        Rule #1: Extracted definition question generator (<60 lines, 1 nesting level).
        Rule #9: Complete type hints.

        Args:
            artifact: Chunk artifact to generate questions for.

        Returns:
            List of definition questions.
        """
        questions = []
        section_title = self._get_metadata_field(artifact, "section_title")
        if section_title:
            questions.append(f"What is {section_title}?")
            questions.append(f"How is {section_title} defined?")
        questions.append("What does this term mean?")
        return questions

    def _generate_procedure_questions(self, artifact: IFChunkArtifact) -> List[str]:
        """
        Generate procedure-style questions.

        TASK-012: Migrated to use IFChunkArtifact.
        Rule #1: Extracted procedure question generator (<60 lines, 1 nesting level).
        Rule #9: Complete type hints.

        Args:
            artifact: Chunk artifact to generate questions for.

        Returns:
            List of procedure questions.
        """
        questions = []
        section_title = self._get_metadata_field(artifact, "section_title")
        if section_title:
            questions.append(f"How do I {section_title.lower()}?")
            questions.append(f"What are the steps for {section_title.lower()}?")
        questions.append("What is the process?")
        questions.append("What are the required steps?")
        return questions

    def _generate_example_questions(self, artifact: IFChunkArtifact) -> List[str]:
        """
        Generate example-style questions.

        TASK-012: Migrated to use IFChunkArtifact.
        Rule #1: Extracted example question generator (<60 lines, 1 nesting level).
        Rule #9: Complete type hints.

        Args:
            artifact: Chunk artifact to generate questions for.

        Returns:
            List of example questions.
        """
        questions = ["Can you give an example?"]
        section_title = self._get_metadata_field(artifact, "section_title")
        if section_title:
            questions.append(f"What is an example of {section_title.lower()}?")
        return questions

    def _generate_general_questions(self, artifact: IFChunkArtifact) -> List[str]:
        """
        Generate general questions based on content keywords.

        TASK-012: Migrated to use IFChunkArtifact.
        Rule #1: Extracted general question generator (<60 lines, 1 nesting level).
        Rule #9: Complete type hints.

        Args:
            artifact: Chunk artifact to generate questions for.

        Returns:
            List of general questions.
        """
        questions = []
        content = self._get_content(artifact).lower()
        section_title = self._get_metadata_field(artifact, "section_title")

        # Section-based questions
        if section_title:
            questions.append(f"What is {section_title}?")
            questions.append(f"Tell me about {section_title}")

        # Content-based questions
        if "how" in content:
            questions.append("How does this work?")
        if "why" in content:
            questions.append("Why is this important?")
        if "when" in content:
            questions.append("When should this be used?")
        if "requirement" in content or "must" in content:
            questions.append("What are the requirements?")
        if "benefit" in content or "advantage" in content:
            questions.append("What are the benefits?")

        return questions

    def enrich_chunk(
        self, chunk: "ChunkRecord", num_questions: int = 3
    ) -> "ChunkRecord":
        """
        Add generated questions to chunk.

        .. deprecated:: 2.0.0
            Use :meth:`process` with IFChunkArtifact instead.

        Args:
            chunk: Chunk to enrich.
            num_questions: Number of questions.

        Returns:
            Chunk with hypothetical_questions populated.
        """
        warnings.warn(
            "enrich_chunk() is deprecated. Use process() with IFChunkArtifact instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Convert to artifact, process, then back to record
        artifact = IFChunkArtifact.from_chunk_record(chunk)
        questions = self._generate_questions(artifact, num_questions)
        chunk.hypothetical_questions = questions
        return chunk

    def is_available(self) -> bool:
        """
        Check if question generator is available.

        Implements IFProcessor.is_available().

        Returns:
            True (template-based generation always available)
        """
        return True  # Template fallback always available

    def teardown(self) -> bool:
        """
        Clean up resources.

        Implements IFProcessor.teardown().

        Returns:
            True (no resources to clean up).
        """
        return True

    def enrich(self, chunk: "ChunkRecord", num_questions: int = 3) -> "ChunkRecord":
        """
        Deprecated: use process() with IFChunkArtifact instead.

        .. deprecated:: 2.0.0

        Args:
            chunk: Chunk to enrich.
            num_questions: Number of questions.

        Returns:
            Chunk with hypothetical_questions populated.
        """
        warnings.warn(
            "enrich() is deprecated. Use process() with IFChunkArtifact instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Convert to artifact, process, then back to record
        artifact = IFChunkArtifact.from_chunk_record(chunk)
        questions = self._generate_questions(artifact, num_questions)
        chunk.hypothetical_questions = questions
        return chunk
