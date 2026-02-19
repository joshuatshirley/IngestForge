"""Named Entity Recognition (NER) enrichment.

Extracts named entities (people, organizations, locations, etc.)
from text chunks using spaCy (production) or regex fallback.

Production Features:
- spaCy integration for high-accuracy NER (>85% precision/recall)
- Extracts: PERSON, ORG, GPE, DATE, MONEY, PERCENT, EVENT, WORK_OF_ART
- Confidence scores and entity contexts
- Fallback to regex patterns if spaCy unavailable
- Batch processing for efficiency"""

from __future__ import annotations

from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
import re
from functools import lru_cache

from ingestforge.core.errors import SafeErrorMessage
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Entity:
    """Structured entity with position, confidence, and normalization.

    Attributes:
        text: Original entity text as found in document
        type: Entity type (PERSON, ORG, GPE, DATE, EVENT, WORK_OF_ART)
        start: Character offset where entity starts
        end: Character offset where entity ends
        confidence: Confidence score from 0.0 to 1.0
        normalized: Optional normalized/canonical form for linking
    """

    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0
    normalized: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary format.

        Returns:
            Dictionary with all entity fields
        """
        return {
            "text": self.text,
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "normalized": self.normalized,
        }

    def __hash__(self) -> int:
        """Enable hashing for deduplication."""
        return hash((self.type, self.text.lower(), self.start, self.end))

    def __eq__(self, other: object) -> bool:
        """Enable equality comparison."""
        if not isinstance(other, Entity):
            return False
        return (
            self.type == other.type
            and self.text.lower() == other.text.lower()
            and self.start == other.start
            and self.end == other.end
        )


@lru_cache(maxsize=1)
def _load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load spaCy model with caching.

    Rule #1: Simple lazy loading with caching
    Rule #7: Defensive - handles missing model gracefully
    """
    try:
        import spacy

        try:
            return spacy.load(model_name)
        except OSError:
            # Model not installed, try to download
            logger.warning(
                f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}"
            )
            return None
    except ImportError:
        logger.warning("spaCy not installed. Using regex fallback for NER.")
        return None


class SpacyNEREnricher:
    """Production-quality NER using spaCy.

    Extracts entities with high precision/recall (>85%) using
    pre-trained transformer models.
    """

    # Entity types to extract (spaCy standard)
    ENTITY_TYPES = {
        "PERSON",  # People, including fictional
        "ORG",  # Companies, agencies, institutions
        "GPE",  # Countries, cities, states
        "LOC",  # Non-GPE locations
        "DATE",  # Absolute or relative dates
        "TIME",  # Times smaller than a day
        "MONEY",  # Monetary values
        "PERCENT",  # Percentages
        "PRODUCT",  # Objects, vehicles, foods, etc.
        "EVENT",  # Named hurricanes, battles, wars, etc.
        "WORK_OF_ART",  # Titles of books, songs, etc.
        "LAW",  # Named documents made into laws
        "LANGUAGE",  # Any named language
    }

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Initialize spaCy NER enricher.

        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
                       Options: en_core_web_sm, en_core_web_md, en_core_web_lg
        """
        self.model_name = model_name
        self._nlp = None  # Lazy-load

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            self._nlp = _load_spacy_model(self.model_name)
        return self._nlp

    def is_available(self) -> bool:
        """Check if spaCy is available."""
        return self.nlp is not None

    def enrich(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich chunk with named entities using spaCy.

        Args:
            chunk: Chunk dictionary with 'text' or 'content' field

        Returns:
            Enriched chunk with 'entities' field

        Rule #4: Function <60 lines
        Rule #7: Validates input
        """
        # Get text from chunk (support both 'text' and 'content' keys)
        text = chunk.get("text") or chunk.get("content", "")

        if not text or not isinstance(text, str):
            chunk["entities"] = []
            chunk["entity_counts"] = {}
            return chunk

        if not self.is_available():
            logger.warning("spaCy not available, cannot extract entities")
            chunk["entities"] = []
            chunk["entity_counts"] = {}
            return chunk

        # Extract entities
        entities = self._extract_entities(text)

        # Add to chunk
        chunk["entities"] = entities
        chunk["entity_counts"] = self._count_entity_types(entities)

        return chunk

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using spaCy.

        Rule #4: Function <60 lines
        """
        if not self.nlp:
            return []

        # Process text with spaCy
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            # Only include specified entity types
            if ent.label_ in self.ENTITY_TYPES:
                entity_dict = {
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }

                # Add context (surrounding text)
                context = self._get_entity_context(text, ent.start_char, ent.end_char)
                if context:
                    entity_dict["context"] = context

                entities.append(entity_dict)

        # Deduplicate while preserving order
        return self._deduplicate_entities(entities)

    def _get_entity_context(
        self, text: str, start: int, end: int, window: int = 50
    ) -> str:
        """Get surrounding context for an entity.

        Args:
            text: Full text
            start: Entity start position
            end: Entity end position
            window: Characters to include on each side

        Returns:
            Context string

        Rule #4: Function <60 lines
        """
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)

        context = text[context_start:context_end]

        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."

        return context.strip()

    def _deduplicate_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate entities.

        Rule #4: Function <60 lines
        """
        seen = set()
        unique = []

        for entity in entities:
            # Create key from type and normalized text
            key = (entity["type"], entity["text"].lower())

            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return sorted(unique, key=lambda x: x["start"])

    def _count_entity_types(self, entities: List[Dict]) -> Dict[str, int]:
        """Count entities by type.

        Args:
            entities: List of entity dictionaries

        Returns:
            Dictionary of counts by type
        """
        counts: Dict[str, int] = {}

        for entity in entities:
            entity_type = entity["type"]
            counts[entity_type] = counts.get(entity_type, 0) + 1

        return counts


class NEREnricher:
    """Extract named entities from text."""

    def __init__(self) -> None:
        """Initialize NER enricher."""
        # Simple pattern-based NER (for production, use spaCy or transformers)
        self.entity_patterns = {
            "PERSON": self._build_person_patterns(),
            "ORG": self._build_org_patterns(),
            "GPE": self._build_location_patterns(),
            "DATE": self._build_date_patterns(),
        }

    def enrich(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich chunk with named entities.

        Args:
            chunk: Chunk dictionary with 'text' field

        Returns:
            Enriched chunk with 'entities' field
        """
        text = chunk.get("text", "")

        if not text:
            chunk["entities"] = []
            return chunk

        # Extract entities
        entities = self._extract_entities(text)

        # Add to chunk
        chunk["entities"] = entities
        chunk["entity_counts"] = self._count_entity_types(entities)

        return chunk

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities from text.

        Args:
            text: Input text

        Returns:
            List of entity dictionaries
        """
        entities = []
        seen = set()

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                self._extract_pattern_matches(
                    text, pattern, entity_type, entities, seen
                )

        return sorted(entities, key=lambda x: x["start"])

    def _extract_pattern_matches(
        self,
        text: str,
        pattern: str,
        entity_type: str,
        entities: List[Dict],
        seen: set,
    ) -> None:
        """Extract matches for a single pattern.

        Args:
            text: Input text
            pattern: Regex pattern
            entity_type: Entity type name
            entities: List to append matches to
            seen: Set of seen entity keys
        """
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            entity_text = match.group(0)
            key = f"{entity_type}:{entity_text.lower()}"

            if key not in seen:
                entities.append(
                    {
                        "text": entity_text,
                        "type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
                seen.add(key)

    def _count_entity_types(self, entities: List[Dict]) -> Dict[str, int]:
        """Count entities by type.

        Args:
            entities: List of entity dictionaries

        Returns:
            Dictionary of counts by type
        """
        counts: Dict[str, int] = {}

        for entity in entities:
            entity_type = entity["type"]
            counts[entity_type] = counts.get(entity_type, 0) + 1

        return counts

    def _build_person_patterns(self) -> List[str]:
        """Build patterns for person names."""
        # Title + Name patterns
        titles = r"(?:Dr|Mr|Mrs|Ms|Prof|Professor)\.?"
        name = r"[A-Z][a-z]+"

        return [
            rf"{titles}\s+{name}(?:\s+{name})*",  # Dr. John Smith
            rf"{name}\s+{name}(?:\s+{name})?",  # John Smith / John Q. Smith
        ]

    def _build_org_patterns(self) -> List[str]:
        """Build patterns for organizations."""
        org_suffixes = r"(?:Inc|Corp|Corporation|LLC|Ltd|Company|Co)"

        return [
            rf"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+{org_suffixes}\.?",
            r"(?:University of|Institute of)\s+[A-Z][a-zA-Z]+",
        ]

    def _build_location_patterns(self) -> List[str]:
        """Build patterns for locations."""
        return [
            r"[A-Z][a-z]+(?:,\s*[A-Z]{2})",  # City, ST
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z][a-z]+",  # New York, USA
        ]

    def _build_date_patterns(self) -> List[str]:
        """Build patterns for dates."""
        months = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

        return [
            rf"{months}\s+\d{{1,2}},?\s+\d{{4}}",  # January 1, 2024
            r"\d{1,2}/\d{1,2}/\d{2,4}",  # 1/1/2024
            r"\d{4}-\d{2}-\d{2}",  # 2024-01-01
        ]


def get_ner_enricher(use_spacy: bool = True) -> Any:
    """Get NER enricher (spaCy or regex fallback).

    Args:
        use_spacy: Try to use spaCy if available

    Returns:
        SpacyNEREnricher if available, else NEREnricher (regex)

    Rule #1: Simple factory pattern
    """
    if use_spacy:
        spacy_enricher = SpacyNEREnricher()
        if spacy_enricher.is_available():
            logger.info("Using spaCy NER (production quality)")
            return spacy_enricher

    logger.info("Using regex NER (fallback)")
    return NEREnricher()


def enrich_chunk(chunk: Dict[str, Any], use_spacy: bool = True) -> Dict[str, Any]:
    """Enrich chunk with named entities.

    Uses spaCy for production-quality NER if available,
    falls back to regex patterns.

    Args:
        chunk: Chunk dictionary
        use_spacy: Try to use spaCy (default: True)

    Returns:
        Enriched chunk with entities
    """
    enricher = get_ner_enricher(use_spacy)
    return enricher.enrich(chunk)


def enrich_chunks(
    chunks: List[Dict[str, Any]], use_spacy: bool = True
) -> List[Dict[str, Any]]:
    """Enrich multiple chunks with named entities.

    Uses spaCy for production-quality NER if available,
    falls back to regex patterns.

    Args:
        chunks: List of chunks
        use_spacy: Try to use spaCy (default: True)

    Returns:
        List of enriched chunks
    """
    enricher = get_ner_enricher(use_spacy)
    return [enricher.enrich(chunk) for chunk in chunks]


def extract_entities_from_text(
    text: str, use_spacy: bool = True
) -> List[Dict[str, Any]]:
    """Extract entities from raw text.

    Convenience function for direct entity extraction.

    Args:
        text: Input text
        use_spacy: Try to use spaCy (default: True)

    Returns:
        List of entity dictionaries
    """
    chunk = {"text": text}
    enriched = enrich_chunk(chunk, use_spacy=use_spacy)
    return enriched.get("entities", [])


# =============================================================================
# NERExtractor - Production-Quality Named Entity Recognition
# =============================================================================


class NERExtractor:
    """Production-quality Named Entity Recognition extractor.

    Extracts named entities using spaCy with regex fallback.
    Returns Entity dataclass objects with confidence scores.

    Supported entity types:
        - PERSON: People, including fictional
        - ORG: Companies, agencies, institutions
        - GPE: Countries, cities, states
        - DATE: Absolute or relative dates
        - EVENT: Named events (battles, hurricanes, etc.)
        - WORK_OF_ART: Titles of books, songs, etc.

    Example:
        >>> extractor = NERExtractor(model="en_core_web_sm")
        >>> entities = extractor.extract("Apple Inc. was founded by Steve Jobs.")
        >>> for e in entities:
        ...     print(f"{e.text} ({e.type}): {e.confidence:.2f}")

            - Rule #1: Max 3 nesting levels, early returns
        - Rule #4: Functions <60 lines
        - Rule #5: No silent exceptions
        - Rule #9: Full type hints
    """

    # Supported entity types
    ENTITY_TYPES: List[str] = [
        "PERSON",
        "ORG",
        "GPE",
        "DATE",
        "EVENT",
        "WORK_OF_ART",
        "LOC",
        "TIME",
        "MONEY",
        "PERCENT",
        "PRODUCT",
        "LAW",
        "LANGUAGE",
    ]

    def __init__(self, model: str = "en_core_web_sm") -> None:
        """Initialize NER extractor.

        Args:
            model: spaCy model name (default: en_core_web_sm)
                Options: en_core_web_sm, en_core_web_md, en_core_web_lg

        Rule #7: Defensive - stores model name for lazy loading
        """
        self.model_name = model
        self._nlp = None
        self._spacy_available = True
        self._fallback_enricher: Optional[NEREnricher] = None

    def _try_load_model(self) -> bool:
        """Try to load the spaCy model.

        Rule #1: Early returns for clarity
        Rule #4: Function <60 lines
        Rule #5: Logs all errors

        Returns:
            True if model loaded successfully
        """
        if self._nlp is not None:
            return True

        try:
            import spacy

            self._nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
            return True
        except ImportError:
            logger.warning("spaCy not installed. Using regex fallback.")
            self._spacy_available = False
            return False
        except OSError:
            logger.warning(
                f"Model '{self.model_name}' not found. "
                f"Install with: python -m spacy download {self.model_name}"
            )
            self._spacy_available = False
            return False

    def _get_fallback_enricher(self) -> NEREnricher:
        """Get or create fallback regex enricher.

        Rule #6: Variable at smallest scope

        Returns:
            NEREnricher instance for regex-based extraction
        """
        if self._fallback_enricher is None:
            self._fallback_enricher = NEREnricher()
        return self._fallback_enricher

    def _estimate_confidence(self, ent: Any) -> float:
        """Estimate confidence score for spaCy entity.

        Uses heuristics since spaCy doesn't provide built-in scores.

        Args:
            ent: spaCy entity span

        Returns:
            Confidence score from 0.0 to 1.0

        Rule #4: Function <60 lines
        """
        # Base confidence by model size
        if "lg" in self.model_name:
            confidence = 0.90
        elif "md" in self.model_name:
            confidence = 0.85
        else:
            confidence = 0.80

        # Boost for multi-word entities
        if len(ent.text.split()) >= 2:
            confidence += 0.03

        # Boost for proper capitalization
        if ent.label_ in ("PERSON", "ORG", "GPE") and ent.text[0].isupper():
            confidence += 0.02

        # Reduce for very short entities
        if len(ent.text) <= 2:
            confidence -= 0.10

        return min(max(confidence, 0.0), 1.0)

    def _normalize_entity_text(self, text: str, entity_type: str) -> str:
        """Create normalized form of entity text.

        Args:
            text: Original entity text
            entity_type: Entity type (PERSON, ORG, etc.)

        Returns:
            Normalized text for linking

        Rule #4: Function <60 lines
        """
        normalized = text.strip()

        # Remove titles from person names
        if entity_type == "PERSON":
            titles = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Professor"]
            for title in titles:
                if normalized.startswith(title):
                    normalized = normalized[len(title) :].strip()

        # Standardize organization suffixes
        if entity_type == "ORG":
            suffix_map = {
                "Corp.": "Corporation",
                "Corp": "Corporation",
                "Inc.": "Incorporated",
                "Inc": "Incorporated",
                "Ltd.": "Limited",
                "Ltd": "Limited",
            }
            for abbrev, full in suffix_map.items():
                if normalized.endswith(abbrev):
                    normalized = normalized[: -len(abbrev)] + full

        return normalized

    def extract(self, text: str) -> List[Entity]:
        """Extract entities from text.

        Primary extraction method returning Entity objects with
        confidence scores and normalized forms.

        Args:
            text: Text to analyze

        Returns:
            List of Entity objects sorted by position

        Rule #1: Early return for empty text
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        if not text or not text.strip():
            return []

        # Try spaCy extraction
        if self._spacy_available and self._try_load_model():
            return self._extract_spacy(text)

        # Fallback to regex
        return self._extract_regex(text)

    def _extract_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy NLP.

        Args:
            text: Text to process

        Returns:
            List of Entity objects

        Rule #4: Function <60 lines
        """
        doc = self._nlp(text)
        entities: List[Entity] = []
        seen: Set[Tuple[str, str, int, int]] = set()

        for ent in doc.ents:
            # Skip unsupported types
            if ent.label_ not in self.ENTITY_TYPES:
                continue

            # Create dedup key
            key = (ent.label_, ent.text.lower(), ent.start_char, ent.end_char)
            if key in seen:
                continue
            seen.add(key)

            # Create entity with confidence
            entity = Entity(
                text=ent.text,
                type=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=self._estimate_confidence(ent),
                normalized=self._normalize_entity_text(ent.text, ent.label_),
            )
            entities.append(entity)

        return sorted(entities, key=lambda e: e.start)

    def _extract_regex(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns.

        Fallback when spaCy is not available.

        Args:
            text: Text to process

        Returns:
            List of Entity objects with lower confidence

        Rule #4: Function <60 lines
        """
        enricher = self._get_fallback_enricher()
        chunk = {"text": text}
        enriched = enricher.enrich(chunk)

        entities: List[Entity] = []
        for ent_dict in enriched.get("entities", []):
            entity = Entity(
                text=ent_dict["text"],
                type=ent_dict["type"],
                start=ent_dict["start"],
                end=ent_dict["end"],
                confidence=0.60,  # Lower confidence for regex
                normalized=self._normalize_entity_text(
                    ent_dict["text"], ent_dict["type"]
                ),
            )
            entities.append(entity)

        return sorted(entities, key=lambda e: e.start)

    def extract_batch(self, texts: List[str]) -> List[List[Entity]]:
        """Extract entities from multiple texts efficiently.

        Uses spaCy's pipe() for batch processing when available.

        Args:
            texts: List of texts to analyze

        Returns:
            List of entity lists, one per input text

        Rule #1: Early return for empty input
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        if not texts:
            return []

        # Use spaCy pipe for batch processing
        if self._spacy_available and self._try_load_model():
            return self._extract_batch_spacy(texts)

        # Fallback to sequential regex
        return [self._extract_regex(text) for text in texts]

    def _extract_batch_spacy(self, texts: List[str]) -> List[List[Entity]]:
        """Batch extract using spaCy pipe.

        Args:
            texts: List of texts

        Returns:
            List of entity lists

        Rule #4: Function <60 lines
        """
        results: List[List[Entity]] = []

        # Use spaCy pipe for efficiency
        for doc in self._nlp.pipe(texts, batch_size=50):
            doc_entities: List[Entity] = []
            seen: Set[Tuple[str, str, int, int]] = set()

            for ent in doc.ents:
                if ent.label_ not in self.ENTITY_TYPES:
                    continue

                key = (ent.label_, ent.text.lower(), ent.start_char, ent.end_char)
                if key in seen:
                    continue
                seen.add(key)

                entity = Entity(
                    text=ent.text,
                    type=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=self._estimate_confidence(ent),
                    normalized=self._normalize_entity_text(ent.text, ent.label_),
                )
                doc_entities.append(entity)

            results.append(sorted(doc_entities, key=lambda e: e.start))

        return results

    def get_entity_types(self) -> List[str]:
        """Get list of supported entity types.

        Returns:
            List of entity type strings

        Rule #4: Simple accessor
        """
        return self.ENTITY_TYPES.copy()

    def is_available(self) -> bool:
        """Check if NER extraction is available.

        Returns:
            True (always available via regex fallback)
        """
        return True

    def is_spacy_available(self) -> bool:
        """Check if spaCy model is available.

        Returns:
            True if spaCy model can be loaded
        """
        return self._spacy_available and self._try_load_model()


# Alias for backwards compatibility
NamedEntityRecognizer = NERExtractor


# =============================================================================
# NERProcessor - IFProcessor Implementation
# =============================================================================

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFFailureArtifact
from ingestforge.core.pipeline.registry import register_enricher


@register_enricher(
    capabilities=["ner", "entity-extraction", "named-entity-recognition"],
    priority=85,
)
class NERProcessor(IFProcessor):
    """
    IFProcessor implementation for Named Entity Recognition.

    Wraps NERExtractor to provide the IFProcessor interface for
    modular pipeline architecture.

    Convergence - Processor Unification.
    Registered via @register_enricher decorator.

    Rule #9: Complete type hints.

    Attributes:
        extractor: Underlying NERExtractor instance.
        _version: Semantic version of this processor.
    """

    def __init__(self, model: str = "en_core_web_sm") -> None:
        """
        Initialize NER processor.

        Args:
            model: spaCy model name (default: en_core_web_sm).
                   Options: en_core_web_sm, en_core_web_md, en_core_web_lg
        """
        self.extractor = NERExtractor(model=model)
        self._version = "1.0.0"
        self._model_name = model

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process an artifact and extract named entities.

        Implements IFProcessor.process().

        Required method for IFProcessor interface.
        Rule #4: Function under 60 lines.
        Rule #7: Explicit return values.

        Args:
            artifact: Input artifact (must be IFChunkArtifact).

        Returns:
            Derived IFChunkArtifact with entities in metadata,
            or IFFailureArtifact on error.
        """
        # Validate input type
        if not isinstance(artifact, IFChunkArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-ner-failed",
                error_message=f"NERProcessor requires IFChunkArtifact, got {type(artifact).__name__}",
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
            )

        try:
            # Extract entities using NERExtractor
            entities = self.extractor.extract(artifact.content)

            # Store in metadata
            new_metadata = dict(artifact.metadata)
            new_metadata["ner_entities"] = [e.to_dict() for e in entities]
            new_metadata["ner_entity_count"] = len(entities)

            # Group entities by type for convenience
            by_type: Dict[str, List[str]] = {}
            for entity in entities:
                if entity.type not in by_type:
                    by_type[entity.type] = []
                by_type[entity.type].append(entity.text)
            new_metadata["ner_by_type"] = by_type

            # Return derived artifact
            return artifact.derive(
                self.processor_id,
                artifact_id=f"{artifact.artifact_id}-ner",
                metadata=new_metadata,
            )

        except Exception as e:
            logger.error(f"NER extraction failed: {e}", exc_info=True)
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-ner-failed",
                # SEC-002: Sanitize error message
                error_message=SafeErrorMessage.sanitize(
                    e, "ner_extraction_failed", logger
                ),
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
            )

    def is_available(self) -> bool:
        """
        Check if NER processor is available.

        Implements IFProcessor.is_available().

        Required method.

        Returns:
            True (always available via regex fallback).
        """
        return self.extractor.is_available()

    @property
    def processor_id(self) -> str:
        """
        Unique identifier for this processor.

        Implements IFProcessor.processor_id.

        Returns:
            Processor ID string.
        """
        return "ner-processor"

    @property
    def version(self) -> str:
        """
        Semantic version of this processor.

        Implements IFProcessor.version.

        Returns:
            SemVer string.
        """
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """
        Capabilities provided by this processor.

        Implements IFProcessor.capabilities.

        Returns:
            List of capability strings.
        """
        return ["ner", "entity-extraction", "named-entity-recognition"]

    @property
    def memory_mb(self) -> int:
        """
        Estimated memory requirement in megabytes.

        Implements IFProcessor.memory_mb.

        Returns:
            Memory estimate in MB.
        """
        # spaCy models require varying amounts of memory
        if self._model_name == "en_core_web_lg":
            return 800  # Large model
        elif self._model_name == "en_core_web_md":
            return 400  # Medium model
        elif self._model_name == "en_core_web_sm":
            return 200  # Small model
        return 50  # Regex-only fallback

    def teardown(self) -> bool:
        """
        Clean up resources.

        Implements IFProcessor.teardown().

        Returns:
            True if cleanup successful.
        """
        self.extractor._nlp = None
        self.extractor._fallback_enricher = None
        return True
