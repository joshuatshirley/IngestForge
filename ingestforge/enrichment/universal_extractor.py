"""
Universal Entity Extractor using Instructor and LLMs.

Universal Entity Extraction
Extracts PERSON, ORG, LOC, DATE and domain-specific entities (Dockets, PartNumbers, CVEs)
using LLM-powered structured extraction via Instructor library.

NASA JPL Power of Ten compliant.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator

from ingestforge.core.pipeline.interfaces import IFProcessor
from ingestforge.core.pipeline.artifacts import (
    IFArtifact,
    IFChunkArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)
from ingestforge.core.config import Config
from ingestforge.llm.factory import get_llm_client
from ingestforge.llm.openai import OpenAIClient
from ingestforge.core.logging import get_logger
from ingestforge.enrichment.model_escalator import ModelEscalator, EscalationResult

# Few-Shot Learning Integration (lazy imports to avoid circular deps)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestforge.core.pipeline.prompt_tuner import IFPromptTuner
    from ingestforge.core.pipeline.learning_registry import IFExampleRegistry
    from ingestforge.enrichment.embeddings import EmbeddingGenerator

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_ENTITIES_PER_EXTRACTION = 100
MAX_TEXT_LENGTH = 32000
MAX_ENTITY_TEXT_LENGTH = 500


# ---------------------------------------------------------------------------
# Pydantic Models for Structured LLM Extraction
# ---------------------------------------------------------------------------

EntityType = Literal[
    "PERSON",
    "ORG",
    "LOC",
    "DATE",  # Standard NER
    "DOCKET",
    "PART_NUMBER",
    "CVE",  # Domain-specific
    "MONEY",
    "PERCENT",
    "EMAIL",
    "URL",
    "PHONE",  # Additional
    "CORRUPT",  # AC#3: Invalid offset marker
]


class ExtractedEntity(BaseModel):
    """
    Single extracted entity with position and confidence.

    Rule #9: Complete type hints.
    """

    text: str = Field(..., description="The entity text as it appears in source")
    entity_type: EntityType = Field(..., description="Type classification")
    start_char: int = Field(..., ge=0, description="Start character offset in source")
    end_char: int = Field(..., ge=0, description="End character offset in source")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    context: Optional[str] = Field(None, description="Surrounding context snippet")
    extraction_rationale: Optional[str] = Field(
        None, description="AC#3: Rationale for why this entity was extracted"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Rule #2: Bounded text length."""
        if len(v) > MAX_ENTITY_TEXT_LENGTH:
            return v[:MAX_ENTITY_TEXT_LENGTH]
        return v

    @property
    def source_link(self) -> str:
        """Generate source link in format start:end."""
        return f"{self.start_char}:{self.end_char}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata storage."""
        result = {
            "text": self.text,
            "type": self.entity_type,
            "start": self.start_char,
            "end": self.end_char,
            "confidence": self.confidence,
            "source_link": self.source_link,
        }
        # AC#3: Include rationale if present
        if self.extraction_rationale:
            result["extraction_rationale"] = self.extraction_rationale
        return result


class EntityExtractionResult(BaseModel):
    """
    Container for all extracted entities from a text.

    Rule #2: Bounded list size.
    Rule #9: Complete type hints.
    """

    entities: List[ExtractedEntity] = Field(
        default_factory=list, description="List of extracted entities"
    )

    @field_validator("entities")
    @classmethod
    def validate_entities(cls, v: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Rule #2: Fixed upper bound on entities."""
        if len(v) > MAX_ENTITIES_PER_EXTRACTION:
            return v[:MAX_ENTITIES_PER_EXTRACTION]
        return v


# ---------------------------------------------------------------------------
# Regex-based fallback patterns (when LLM unavailable)
# ---------------------------------------------------------------------------


@dataclass
class RegexPattern:
    """Pattern definition for fallback extraction."""

    pattern: re.Pattern
    entity_type: EntityType
    confidence: float = 0.7


FALLBACK_PATTERNS: List[RegexPattern] = [
    # CVE IDs
    RegexPattern(re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE), "CVE", 0.95),
    # Docket numbers
    RegexPattern(
        re.compile(r"\b(?:Case|Docket)\s*No\.\s*([\w\d\-\:]+)\b", re.IGNORECASE),
        "DOCKET",
        0.85,
    ),
    RegexPattern(
        re.compile(r"\b(\d{1,2}-[a-zA-Z]{1,4}-\d{3,7}(?:-[a-zA-Z]{1,4})?)\b"),
        "DOCKET",
        0.80,
    ),
    # Part numbers
    RegexPattern(
        re.compile(r"(?:Part|P/N|Ref)[:\s\-#]+([A-Z0-9\-]{7,25})\b", re.IGNORECASE),
        "PART_NUMBER",
        0.85,
    ),
    # Dates
    RegexPattern(
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b"
        ),
        "DATE",
        0.90,
    ),
    RegexPattern(re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "DATE", 0.95),
    # Money
    RegexPattern(
        re.compile(
            r"\$\s?\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:million|billion|M|B|K)?",
            re.IGNORECASE,
        ),
        "MONEY",
        0.90,
    ),
    # Email
    RegexPattern(
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "EMAIL", 0.95
    ),
    # URL
    RegexPattern(re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'), "URL", 0.95),
    # Phone
    RegexPattern(
        re.compile(r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),
        "PHONE",
        0.85,
    ),
    # Person (title + name pattern)
    RegexPattern(
        re.compile(
            r"\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+\b"
        ),
        "PERSON",
        0.75,
    ),
    # Organization (suffix pattern)
    RegexPattern(
        re.compile(
            r"\b(?:[A-Z][a-z]*\.?\s*)+(?:Inc\.?|Corp\.?|Ltd\.?|LLC|LLP|"
            r"Company|Corporation|Foundation|Institute|University|Bank)\b",
            re.IGNORECASE,
        ),
        "ORG",
        0.70,
    ),
]


# ---------------------------------------------------------------------------
# IFUniversalExtractor Implementation
# ---------------------------------------------------------------------------


class IFUniversalExtractor(IFProcessor):
    """
    Universal Entity Extractor using Instructor/LLM with regex fallback.

    Universal Entity Extraction
    - Uses Instructor with poly-entity Pydantic model
    - Detects standard NER (PERSON, ORG, LOC, DATE)
    - Detects domain markers (Dockets, PartNumbers, CVEs)
    - Every entity has confidence score and source_link

    Multi-Model Fallback
    - Fast model first, escalate to smart model if confidence low
    - Configurable fallback_threshold (default 0.6)
    - Escalation events logged and tracked in provenance

    NASA JPL Power of Ten compliant.
    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    # JPL Rule #2: Fixed bounds for escalation
    MAX_ESCALATION_ATTEMPTS = 1
    DEFAULT_FALLBACK_THRESHOLD = 0.6

    def __init__(
        self,
        config: Optional[Config] = None,
        model: Optional[str] = None,
        min_confidence: float = 0.5,
        use_llm: bool = True,
        fast_model: Optional[str] = None,
        smart_model: Optional[str] = None,
        fallback_threshold: float = 0.6,
        enable_few_shot: bool = False,
        embedder: Optional["EmbeddingGenerator"] = None,
        example_registry: Optional["IFExampleRegistry"] = None,
        cloud_provider: bool = True,
    ):
        """
        Initialize the extractor.

        Args:
            config: IngestForge configuration (optional for fallback mode)
            model: LLM model to use (defaults to config or gpt-4o-mini)
            min_confidence: Minimum confidence threshold for entities
            use_llm: Whether to attempt LLM extraction (False = regex only)
            fast_model: Fast model for initial extraction (default: gpt-4o-mini)
            smart_model: Smart model for fallback (default: gpt-4o)
            fallback_threshold: Confidence threshold for escalation (default: 0.6)
            enable_few_shot: Enable few-shot learning from golden examples
            embedder: Embedder for semantic similarity matching
            example_registry: Registry of golden examples
            cloud_provider: Sanitize PII when using cloud LLMs
        """
        self.config = config
        self.model = model or "gpt-4o-mini"
        self.min_confidence = min_confidence
        self.use_llm = use_llm
        # Multi-Model Fallback configuration
        self.fast_model = fast_model or "gpt-4o-mini"
        self.smart_model = smart_model or "gpt-4o"
        self.fallback_threshold = fallback_threshold
        self._instructor_client: Any = None
        self._llm_available: Optional[bool] = None
        # ModelEscalator for multi-model fallback
        self._escalator = ModelEscalator(
            fast_model=self.fast_model,
            smart_model=self.smart_model,
            fallback_threshold=self.fallback_threshold,
        )
        # Few-Shot Learning configuration
        self._enable_few_shot = enable_few_shot
        self._embedder = embedder
        self._example_registry = example_registry
        self._cloud_provider = cloud_provider
        self._prompt_tuner: Optional["IFPromptTuner"] = None

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "universal-entity-extractor"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        """Capabilities provided by this processor."""
        return [
            "ner",
            "entity-extraction",
            "universal-extraction",
            "domain-extraction",
        ]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 150  # LLM client + regex patterns

    def is_available(self) -> bool:
        """
        Check if extractor is available.

        Always returns True since regex fallback is always available.
        """
        return True

    def _check_llm_available(self) -> bool:
        """
        Check if LLM extraction is available.

        Rule #4: Helper function < 60 lines.
        """
        if self._llm_available is not None:
            return self._llm_available

        if not self.use_llm:
            self._llm_available = False
            return False

        if not self.config:
            self._llm_available = False
            return False

        try:
            import instructor

            client = get_llm_client(self.config)
            if isinstance(client, OpenAIClient) and client.is_available():
                self._instructor_client = instructor.patch(client.client)
                self._llm_available = True
                return True
        except ImportError:
            logger.debug("Instructor library not installed")
        except Exception as e:
            logger.debug(f"LLM client initialization failed: {e}")

        self._llm_available = False
        return False

    def _extract_with_llm(
        self, text: str, model: Optional[str] = None
    ) -> List[ExtractedEntity]:
        """
        Extract entities using LLM via Instructor.

        Rule #4: Function < 60 lines.
        Rule #7: Handle errors gracefully.

        Args:
            text: Text to extract entities from
            model: Model to use (defaults to self.model)

        Returns:
            List of extracted entities
        """
        if not self._instructor_client:
            return []

        # Use specified model or default
        target_model = model or self.model

        # Truncate text if too long
        truncated = text[:MAX_TEXT_LENGTH] if len(text) > MAX_TEXT_LENGTH else text

        try:
            # Pass text for few-shot example matching
            system_prompt = self._get_system_prompt(text=truncated)

            result: EntityExtractionResult = self._instructor_client.chat.completions.create(
                model=target_model,
                response_model=EntityExtractionResult,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Extract all entities from the following text:\n\n{truncated}",
                    },
                ],
                max_retries=2,
            )
            return result.entities
        except Exception as e:
            logger.warning(f"LLM extraction failed with {target_model}: {e}")
            return []

    # -------------------------------------------------------------------------
    # Few-Shot Learning Integration
    # -------------------------------------------------------------------------

    def _init_prompt_tuner(self) -> None:
        """
        Lazy-initialize the prompt tuner for few-shot learning.

        Initialize IFPromptTuner on first use.
        Rule #4: Helper function < 60 lines.
        """
        if self._prompt_tuner is not None:
            return

        from ingestforge.core.pipeline.prompt_tuner import IFPromptTuner

        self._prompt_tuner = IFPromptTuner(
            model=self.model,
            cloud_provider=self._cloud_provider,
            max_examples=3,
        )
        logger.debug("Initialized IFPromptTuner for few-shot learning")

    def _get_few_shot_examples(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get semantically similar golden examples for few-shot injection.

        Fetches examples from registry using embedding similarity.
        Rule #4: Helper function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            text: Source text to find similar examples for.

        Returns:
            List of (chunk_text, entities_dict) pairs for prompt injection.
        """
        assert text is not None, "text cannot be None"

        if not self._enable_few_shot:
            return []

        if self._embedder is None or self._example_registry is None:
            logger.debug("Few-shot disabled: embedder or registry not configured")
            return []

        try:
            # Generate embedding for current text
            embedding = self._embedder.embed(text[:MAX_TEXT_LENGTH])

            # Find similar examples from registry
            examples = self._example_registry.find_similar(
                chunk_embedding=embedding,
                limit=3,
            )

            if examples:
                logger.info(f"Found {len(examples)} few-shot examples for injection")

            return examples

        except Exception as e:
            logger.warning(f"Few-shot example retrieval failed: {e}")
            return []

    def _get_system_prompt(self, text: Optional[str] = None) -> str:
        """
        Build system prompt for LLM extraction.

        Optionally injects few-shot examples if enabled.
        Rule #4: Function < 60 lines.

        Args:
            text: Optional source text for few-shot example matching.

        Returns:
            System prompt string, optionally enhanced with examples.
        """
        base_prompt = """You are an expert entity extraction system. Extract all entities from the provided text.

For each entity, identify:
1. text: The exact text as it appears in the source
2. entity_type: One of PERSON, ORG, LOC, DATE, DOCKET, PART_NUMBER, CVE, MONEY, PERCENT, EMAIL, URL, PHONE
3. start_char: Character offset where entity starts (0-indexed)
4. end_char: Character offset where entity ends
5. confidence: Your confidence (0.0-1.0) in this classification
6. extraction_rationale: Brief explanation of why this entity was extracted (AC#3)

Entity type definitions:
- PERSON: Names of people (e.g., "John Smith", "Dr. Jane Doe")
- ORG: Organizations, companies, institutions (e.g., "Microsoft", "FBI", "Harvard University")
- LOC: Locations, places, geographic entities (e.g., "New York", "Pacific Ocean")
- DATE: Dates in any format (e.g., "January 15, 2024", "2024-01-15")
- DOCKET: Legal docket/case numbers (e.g., "20-1234", "1:22-cv-00001")
- PART_NUMBER: Manufacturing part numbers (e.g., "P/N ABC-12345")
- CVE: CVE vulnerability IDs (e.g., "CVE-2024-1234")
- MONEY: Monetary amounts (e.g., "$1,000", "500 million dollars")
- PERCENT: Percentages (e.g., "15%", "twenty percent")
- EMAIL: Email addresses
- URL: Web URLs
- PHONE: Phone numbers

Be precise with character offsets. Only extract clear, unambiguous entities."""

        # Inject few-shot examples if enabled
        if text is not None and self._enable_few_shot:
            examples = self._get_few_shot_examples(text)
            if examples:
                self._init_prompt_tuner()
                if self._prompt_tuner is not None:
                    enhanced = self._prompt_tuner.inject_examples(base_prompt, examples)
                    return enhanced

        return base_prompt

    def _extract_with_regex(self, text: str) -> List[ExtractedEntity]:
        """
        Fallback extraction using regex patterns.

        Rule #2: Fixed upper bound (MAX_ENTITIES_PER_EXTRACTION).
        Rule #4: Function < 60 lines.
        """
        entities: List[ExtractedEntity] = []
        seen: set = set()  # Deduplicate by (text, start, end)

        for pattern_def in FALLBACK_PATTERNS:
            if len(entities) >= MAX_ENTITIES_PER_EXTRACTION:
                break

            for match in pattern_def.pattern.finditer(text):
                # Get matched text (group 1 if exists, else group 0)
                matched_text = match.group(1) if match.lastindex else match.group(0)
                start = match.start(1) if match.lastindex else match.start(0)
                end = match.end(1) if match.lastindex else match.end(0)

                # Deduplicate
                key = (matched_text.strip(), start, end)
                if key in seen:
                    continue
                seen.add(key)

                # Build entity with rationale (AC#3)
                entity = ExtractedEntity(
                    text=matched_text.strip(),
                    entity_type=pattern_def.entity_type,
                    start_char=start,
                    end_char=end,
                    confidence=pattern_def.confidence,
                    extraction_rationale=f"Matched regex pattern for {pattern_def.entity_type}",
                )
                entities.append(entity)

                if len(entities) >= MAX_ENTITIES_PER_EXTRACTION:
                    break

        return entities

    def _validate_offsets(
        self, entities: List[ExtractedEntity], source_text: str
    ) -> List[ExtractedEntity]:
        """
        Validate entity offsets and tag CORRUPT if invalid.

        AC#3: If offsets are invalid, entity is tagged as CORRUPT.
        Rule #4: Function < 60 lines.
        Rule #9: Complete type hints.

        Args:
            entities: List of extracted entities to validate
            source_text: Original source text for offset validation

        Returns:
            List of entities with invalid ones tagged as CORRUPT
        """
        validated: List[ExtractedEntity] = []
        text_len = len(source_text)

        for entity in entities:
            is_valid = True

            # Check 1: start_char must be < end_char
            if entity.start_char >= entity.end_char:
                logger.warning(
                    f"CORRUPT entity '{entity.text}': start_char >= end_char "
                    f"({entity.start_char} >= {entity.end_char})"
                )
                is_valid = False

            # Check 2: Offsets within text bounds
            elif entity.start_char < 0 or entity.end_char > text_len:
                logger.warning(
                    f"CORRUPT entity '{entity.text}': offsets out of bounds "
                    f"({entity.start_char}:{entity.end_char}) for text length {text_len}"
                )
                is_valid = False

            # Check 3: Text at offset matches entity text (if valid bounds)
            elif is_valid:
                actual_text = source_text[entity.start_char : entity.end_char]
                if actual_text.strip() != entity.text.strip():
                    logger.warning(
                        f"CORRUPT entity '{entity.text}': text mismatch at offset "
                        f"(expected '{entity.text}', found '{actual_text}')"
                    )
                    is_valid = False

            if is_valid:
                validated.append(entity)
            else:
                # Tag as CORRUPT with original confidence lowered
                corrupt_entity = ExtractedEntity(
                    text=entity.text,
                    entity_type="CORRUPT",
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                    confidence=0.0,  # CORRUPT entities have zero confidence
                    context=f"Original type: {entity.entity_type}",
                )
                validated.append(corrupt_entity)

        return validated

    def _merge_entities(
        self, llm_entities: List[ExtractedEntity], regex_entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """
        Merge LLM and regex entities, preferring LLM when overlapping.

        Rule #4: Function < 60 lines.
        """
        # Start with LLM entities (higher quality)
        merged: List[ExtractedEntity] = list(llm_entities)
        llm_spans: set = {(e.start_char, e.end_char) for e in llm_entities}

        # Add regex entities that don't overlap with LLM
        for entity in regex_entities:
            if (entity.start_char, entity.end_char) not in llm_spans:
                # Check for partial overlap
                overlap = False
                for start, end in llm_spans:
                    if not (entity.end_char <= start or entity.start_char >= end):
                        overlap = True
                        break
                if not overlap:
                    merged.append(entity)

        # Sort by position
        merged.sort(key=lambda e: (e.start_char, e.end_char))

        # Enforce limit
        return merged[:MAX_ENTITIES_PER_EXTRACTION]

    def _calculate_avg_confidence(self, entities: List[ExtractedEntity]) -> float:
        """
        Calculate average confidence score for entities.

        Multi-Model Fallback - determine if escalation needed.
        Rule #4: Helper function < 60 lines.

        Args:
            entities: List of extracted entities

        Returns:
            Average confidence score, or 1.0 if no entities
        """
        if not entities:
            return 1.0  # No entities = no need to escalate
        return sum(e.confidence for e in entities) / len(entities)

    def _should_escalate(self, entities: List[ExtractedEntity]) -> bool:
        """
        Determine if extraction should escalate to smart model.

        Multi-Model Fallback AC.
        Rule #4: Helper function < 60 lines.

        Args:
            entities: Entities from fast model extraction

        Returns:
            True if escalation needed (avg confidence < fallback_threshold)
        """
        if not entities:
            return False  # No entities to improve

        avg_confidence = self._calculate_avg_confidence(entities)
        should_escalate = avg_confidence < self.fallback_threshold

        if should_escalate:
            logger.info(
                f"Escalation triggered: avg_confidence={avg_confidence:.2f} "
                f"< threshold={self.fallback_threshold}"
            )

        return should_escalate

    def _extract_internal(self, text: str) -> Tuple[List[ExtractedEntity], bool]:
        """
        Internal extraction with escalation tracking.

        Multi-Model Fallback
        - Uses fast model first via ModelEscalator
        - Escalates to smart model if confidence < fallback_threshold

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (list of extracted entities, escalation_occurred)
        """
        escalated = False

        # Get LLM entities if available
        llm_entities: List[ExtractedEntity] = []
        if self._check_llm_available():
            # Use ModelEscalator for multi-model fallback
            def confidence_extractor(e: ExtractedEntity) -> float:
                return e.confidence

            result: EscalationResult = self._escalator.extract_with_fallback(
                extract_fn=lambda t: self._extract_with_llm(
                    t, model=self._escalator.fast_model
                ),
                text=text,
                confidence_extractor=confidence_extractor,
            )
            llm_entities = result.result
            escalated = result.escalated

        # Always get regex entities as supplement/fallback
        regex_entities = self._extract_with_regex(text)

        # Merge and filter
        all_entities = self._merge_entities(llm_entities, regex_entities)

        # AC#3: Validate offsets and tag CORRUPT if invalid
        validated_entities = self._validate_offsets(all_entities, text)

        # Filter by confidence (CORRUPT entities have 0.0 confidence)
        filtered = [
            e for e in validated_entities if e.confidence >= self.min_confidence
        ]

        return (filtered, escalated)

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from text (backward-compatible interface).

        Args:
            text: Input text to analyze

        Returns:
            List of extracted entities
        """
        entities, _ = self._extract_internal(text)
        return entities

    def extract_with_escalation(self, text: str) -> Tuple[List[ExtractedEntity], bool]:
        """
        Extract entities with escalation info.

        Multi-Model Fallback
        Returns tuple with escalation status for tracking purposes.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (list of extracted entities, was_escalated)
        """
        return self._extract_internal(text)

    def extract_all_including_corrupt(
        self, text: str
    ) -> Tuple[List[ExtractedEntity], List[ExtractedEntity]]:
        """
        Extract all entities including CORRUPT ones.

        AC#3: Negative Test support.
        Returns both valid and CORRUPT entities for auditing.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (valid_entities, corrupt_entities)
        """
        escalated = False

        # Get LLM entities if available
        llm_entities: List[ExtractedEntity] = []
        if self._check_llm_available():

            def confidence_extractor(e: ExtractedEntity) -> float:
                return e.confidence

            result: EscalationResult = self._escalator.extract_with_fallback(
                extract_fn=lambda t: self._extract_with_llm(
                    t, model=self._escalator.fast_model
                ),
                text=text,
                confidence_extractor=confidence_extractor,
            )
            llm_entities = result.result
            escalated = result.escalated

        # Get regex entities
        regex_entities = self._extract_with_regex(text)

        # Merge
        all_entities = self._merge_entities(llm_entities, regex_entities)

        # Validate offsets (tags CORRUPT)
        validated = self._validate_offsets(all_entities, text)

        # Separate valid and corrupt
        valid = [e for e in validated if e.entity_type != "CORRUPT"]
        corrupt = [e for e in validated if e.entity_type == "CORRUPT"]

        return (valid, corrupt)

    def extract_simple(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities without escalation info (deprecated, use extract()).

        Args:
            text: Input text to analyze

        Returns:
            List of extracted entities
        """
        return self.extract(text)

    @property
    def escalation_rate(self) -> float:
        """
        Get the escalation rate for health metrics.

        Track "Escalation Rate" in health metrics.

        Returns:
            Escalation rate as percentage (0.0 - 100.0)
        """
        return self._escalator.get_escalation_rate()

    def _categorize_confidence(
        self, entities: List[ExtractedEntity]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize entities into High/Med/Low confidence buckets.

        Confidence-Aware-Extraction.
        Rule #4: Function < 60 lines.

        Args:
            entities: List of extracted entities

        Returns:
            Dictionary with High, Med, Low buckets containing entity dicts
        """
        buckets: Dict[str, List[Dict[str, Any]]] = {
            "High": [],
            "Med": [],
            "Low": [],
        }

        for entity in entities:
            entity_dict = entity.to_dict()
            conf = entity.confidence

            # AC: If confidence < 0.5, log LowConfidenceWarning
            if conf < 0.5:
                logger.warning(
                    f"LowConfidenceWarning: Entity '{entity.text}' ({entity.entity_type}) "
                    f"has low confidence {conf:.2f}"
                )
                buckets["Low"].append(entity_dict)
            elif conf >= 0.8:
                buckets["High"].append(entity_dict)
            else:
                buckets["Med"].append(entity_dict)

        return buckets

    def _build_extraction_metadata(
        self,
        artifact: IFArtifact,
        entities: List[ExtractedEntity],
        escalated: bool,
    ) -> Dict[str, Any]:
        """Build metadata dict for extraction result. Rule #4 helper."""
        confidence_buckets = self._categorize_confidence(entities)
        new_metadata = dict(artifact.metadata)
        new_metadata["entities_structured"] = [e.to_dict() for e in entities]
        new_metadata["entity_count"] = len(entities)
        new_metadata["entity_types_found"] = sorted(
            set(e.entity_type for e in entities)
        )
        new_metadata["extraction_method"] = "llm" if self._llm_available else "regex"
        new_metadata["confidence_buckets"] = confidence_buckets
        new_metadata["model_escalated"] = escalated
        new_metadata["extraction_model"] = (
            self.smart_model if escalated else self.fast_model
        )
        new_metadata["escalation_rate"] = self.escalation_rate
        return new_metadata

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process artifact to extract entities.

        Implements IFProcessor.process().
        Rule #4: Function < 60 lines.
        Rule #7: Handle all input types.

        Args:
            artifact: Input artifact (IFTextArtifact or IFChunkArtifact)

        Returns:
            Derived artifact with entities in metadata
        """
        # Determine content based on artifact type
        if isinstance(artifact, (IFTextArtifact, IFChunkArtifact)):
            content = artifact.content
        else:
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-extraction-failure",
                error_message=f"Requires IFTextArtifact/IFChunkArtifact, got {type(artifact).__name__}",
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
            )

        # Extract entities with multi-model fallback
        entities, escalated = self._extract_internal(content)

        # Build metadata and provenance
        new_metadata = self._build_extraction_metadata(artifact, entities, escalated)
        provenance_chain = artifact.provenance + [self.processor_id]
        if escalated:
            provenance_chain.append(f"{self.processor_id}:escalated")

        return artifact.derive(
            self.processor_id,
            artifact_id=f"{artifact.artifact_id}-entities",
            metadata=new_metadata,
            provenance=provenance_chain,
        )

    def teardown(self) -> bool:
        """Clean up resources."""
        self._instructor_client = None
        self._llm_available = None
        return True
