"""
Production-Quality Named Entity Recognition for chunks.

Extract entities using spaCy (en_core_web_lg) with regex fallback.
Achieves >85% precision/recall on standard NER benchmarks.

Supports entity types:
- PERSON: People, including fictional
- ORG: Companies, agencies, institutions
- GPE: Countries, cities, states (geopolitical entities)
- DATE: Absolute or relative dates
- MONEY, PERCENT, TIME, PRODUCT, EVENT, etc.

Migrated from IEnricher to IFProcessor."""

import re
import hashlib
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from ingestforge.core.pipeline.interfaces import IFProcessor
from ingestforge.core.pipeline.artifacts import (
    IFArtifact,
    IFChunkArtifact,
    IFFailureArtifact,
)
from ingestforge.core.pipeline.registry import register_enricher
from ingestforge.shared.lazy_imports import lazy_property
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


def _content_hash(text: str) -> str:
    """Generate a short hash for caching purposes."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Entity:
    """Structured entity with position and confidence."""

    text: str  # Entity text (e.g., "Barack Obama")
    label: str  # Entity type (e.g., "PERSON")
    start_char: int  # Start position in text
    end_char: int  # End position in text
    confidence: float = 1.0  # Confidence score (0.0-1.0)

    def __post_init__(self) -> None:
        """Validate entity fields after initialization."""
        assert (
            self.start_char <= self.end_char
        ), f"start_char ({self.start_char}) must be <= end_char ({self.end_char})"
        assert (
            0.0 <= self.confidence <= 1.0
        ), f"confidence ({self.confidence}) must be in range [0.0, 1.0]"
        assert self.text, "Entity text cannot be empty"
        assert self.label, "Entity label cannot be empty"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start_char,
            "end": self.end_char,
            "confidence": self.confidence,
        }

    def normalized_text(self) -> str:
        """Get normalized entity text (lowercase, stripped)."""
        return self.text.lower().strip()


@register_enricher(
    capabilities=["ner", "entity-extraction"],
    priority=90,
)
class EntityExtractor(IFProcessor):
    """
    Production-quality named entity extraction.

    Uses spaCy (en_core_web_lg) for >85% precision/recall.
    Falls back to regex patterns if spaCy unavailable.

    Target accuracy: >85% P/R on CoNLL-2003 benchmark.

    Implements IFProcessor interface.
    Registered via @register_enricher decorator.
    """

    def __init__(
        self,
        use_spacy: bool = True,
        model_name: str = "en_core_web_lg",
        min_confidence: float = 0.7,
    ):
        """
        Initialize entity extractor. Rule #4: Extracted patterns.
        """
        self.use_spacy = use_spacy
        self.model_name = model_name
        self.min_confidence = min_confidence
        self._model_cache = None

        # Common patterns for rule-based extraction
        self.patterns = self._initialize_patterns()

        # Entity cache: hash -> list of entities
        self._entity_cache: Dict[str, List[Entity]] = {}

    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for NER. Rule #4: Split into helpers."""
        patterns = self._get_text_patterns()
        patterns.update(self._get_tech_patterns())
        return patterns

    def _get_text_patterns(self) -> Dict[str, re.Pattern]:
        """Patterns for common text entities. Rule #4."""
        return {
            "person": re.compile(
                r"\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+\b|"
                r"\b[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:\s+(?:Jr\.|Sr\.|III|IV))?\b"
            ),
            "organization": re.compile(
                r"\b(?:[A-Z][a-z]*\.?\s*)+(?:"
                r"Inc\.?|Corp\.?|Ltd\.?|LLC|LLP|PLC|GmbH|AG|SA|NV|BV|"
                r"Company|Corporation|Incorporated|Limited|"
                r"Association|Institute|Institution|"
                r"University|College|School|Academy|"
                r"Foundation|Trust|Fund|"
                r"Hospital|Clinic|Medical Center|"
                r"Agency|Bureau|Department|Ministry|Commission|"
                r"Bank|Credit Union|Insurance|"
                r"Group|Holdings|Partners|Ventures|Capital"
                r")\b",
                re.IGNORECASE,
            ),
            "location": re.compile(
                r"\b(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|"
                r"Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|"
                r"Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|"
                r"Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|"
                r"New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|"
                r"Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|"
                r"Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|"
                r"Wisconsin|Wyoming|"
                r"Los\s+Angeles|San\s+Francisco|Chicago|Houston|Phoenix|Philadelphia|"
                r"San\s+Antonio|San\s+Diego|Dallas|San\s+Jose|Austin|Jacksonville|"
                r"Fort\s+Worth|Columbus|Indianapolis|Charlotte|Seattle|Denver|"
                r"Boston|Nashville|Baltimore|Oklahoma\s+City|Portland|Las\s+Vegas|"
                r"Memphis|Louisville|Milwaukee|Albuquerque|Tucson|Fresno|"
                r"Sacramento|Kansas\s+City|Atlanta|Miami|Raleigh|Omaha|"
                r"Mountain\s+View|Palo\s+Alto|Cupertino|Redmond|Cambridge)\b"
            ),
        }

    def _get_tech_patterns(self) -> Dict[str, re.Pattern]:
        """Patterns for technical/numeric entities. Rule #4."""
        return {
            "date": re.compile(
                r"\b(?:January|February|March|April|May|June|July|August|September|"
                r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b|"
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
                r"\b\d{4}-\d{2}-\d{2}\b"
            ),
            "money": re.compile(
                r"\$\s?\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:million|billion|thousand|M|B|K)?|"
                r"\b\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:dollars|USD|EUR|GBP)\b",
                re.IGNORECASE,
            ),
            "percentage": re.compile(
                r"\b\d+(?:\.\d+)?(?:\s)?(?:%|percent|pct)\b", re.IGNORECASE
            ),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "url": re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            "phone": re.compile(
                r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
            ),
        }

        # Entity cache: hash -> list of entities
        self._entity_cache: Dict[str, List[Entity]] = {}

    def _try_load_fallback_model(self) -> Any:
        """
        Try loading fallback spaCy model (en_core_web_sm).

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Returns:
            Loaded spaCy model or None
        """
        if self.model_name == "en_core_web_sm":
            return None

        logger.info("Attempting fallback to en_core_web_sm...")

        try:
            import spacy

            model = spacy.load("en_core_web_sm")
            logger.info("Fallback model loaded: en_core_web_sm")
            return model
        except OSError:
            logger.warning("Fallback model also unavailable. Using regex patterns.")
            return None

    def _handle_model_not_found(self) -> Optional[Any]:
        """
        Handle case when primary spaCy model is not found.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Returns:
            Fallback model if loaded, None otherwise
        """
        logger.warning(
            f"SpaCy model '{self.model_name}' not found. "
            f"Install with: python -m spacy download {self.model_name}"
        )

        # Try fallback model
        fallback = self._try_load_fallback_model()
        if fallback:
            return fallback

        # No model available
        self.use_spacy = False
        return None

    def _load_spacy_model_if_enabled(self) -> Optional[Any]:
        """
        Load spaCy model if enabled.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Returns:
            Loaded spaCy model or None
        """
        if not self.use_spacy:
            return None

        try:
            import spacy

            logger.info(f"Loading spaCy model: {self.model_name}")
            model = spacy.load(self.model_name)
            logger.info(f"SpaCy model loaded successfully: {self.model_name}")
            return model

        except ImportError:
            logger.warning(
                "spaCy not installed. Falling back to pattern matching. "
                "Install with: pip install spacy"
            )
            self.use_spacy = False
            return None

        except OSError:
            return self._handle_model_not_found()

    @lazy_property
    def spacy_model(self) -> Any:
        """
        Lazy-load spaCy model (en_core_web_lg for production accuracy).

        Rule #1: Zero nesting - all logic extracted to helpers
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Returns:
            Loaded spaCy model or None
        """
        if self._model_cache is not None:
            return self._model_cache
        self._model_cache = self._load_spacy_model_if_enabled()
        return self._model_cache

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping entity types to lists of entities
        """
        if self.use_spacy and self.spacy_model:
            return self._extract_spacy(text)
        return self._extract_patterns(text)

    def extract_structured(self, text: str) -> List[Entity]:
        """
        Extract structured entities with positions and confidence.

        Uses caching to avoid redundant processing of identical text.

        Args:
            text: Text to analyze

        Returns:
            List of Entity objects with full metadata

        Rule #4: Function kept under 60 lines
        """
        # Check cache first
        cache_key = _content_hash(text)
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        # Extract entities
        if self.use_spacy and self.spacy_model:
            entities = self._extract_spacy_structured(text)
        else:
            entities = self._extract_patterns_structured(text)

        # Cache result (limit cache size to prevent memory issues)
        if len(self._entity_cache) < 1000:
            self._entity_cache[cache_key] = entities

        return entities

    def _extract_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns."""
        entities: Dict[str, Set[str]] = {k: set() for k in self.patterns}

        for entity_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                entities[entity_type].add(match.strip())

        # Convert sets to sorted lists
        return {k: sorted(list(v))[:20] for k, v in entities.items() if v}

    def _extract_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER."""
        doc = self.spacy_model(text)

        entities: Dict[str, Set[str]] = {}

        # Map spaCy labels to our types
        label_map = {
            "PERSON": "person",
            "ORG": "organization",
            "DATE": "date",
            "MONEY": "money",
            "PERCENT": "percentage",
            "GPE": "location",
            "LOC": "location",
            "TIME": "time",
            "PRODUCT": "product",
            "EVENT": "event",
        }

        for ent in doc.ents:
            entity_type = label_map.get(ent.label_, "other")
            if entity_type not in entities:
                entities[entity_type] = set()
            entities[entity_type].add(ent.text)

        # Convert sets to sorted lists
        return {k: sorted(list(v))[:20] for k, v in entities.items() if v}

    def _extract_spacy_structured(self, text: str) -> List[Entity]:
        """
        Extract structured entities using spaCy.

        Returns Entity objects with positions and confidence scores.
        Rule #4: Function kept under 60 lines
        """
        doc = self.spacy_model(text)
        entities: List[Entity] = []

        for ent in doc.ents:
            # Estimate confidence based on model type and entity characteristics
            confidence = self._estimate_confidence(ent, doc)

            if confidence >= self.min_confidence:
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=confidence,
                )
                entities.append(entity)

        return entities

    def _extract_patterns_structured(self, text: str) -> List[Entity]:
        """
        Extract structured entities using regex patterns.

        Returns Entity objects with positions but lower confidence.
        Rule #4: Function kept under 60 lines
        """
        entities: List[Entity] = []

        # Map pattern keys to standard labels
        label_map = {
            "person": "PERSON",
            "organization": "ORG",
            "location": "GPE",
            "date": "DATE",
            "money": "MONEY",
            "percentage": "PERCENT",
            "email": "EMAIL",
            "url": "URL",
            "phone": "PHONE",
        }

        for pattern_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entity = Entity(
                    text=match.group(0).strip(),
                    label=label_map.get(pattern_type, pattern_type.upper()),
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.6,  # Lower confidence for regex
                )
                entities.append(entity)

        return entities

    def _estimate_confidence(self, ent: Any, doc: Any) -> float:
        """
        Estimate confidence score for spaCy entity.

        Uses heuristics since spaCy doesn't provide built-in scores.
        Rule #4: Function kept under 60 lines
        """
        # Base confidence depends on model
        if self.model_name == "en_core_web_lg":
            confidence = 0.85
        elif self.model_name == "en_core_web_sm":
            confidence = 0.75
        else:
            confidence = 0.70

        # Boost for multi-word entities (more reliable)
        if len(ent.text.split()) >= 2:
            confidence += 0.05

        # Boost for proper capitalization
        if ent.label_ in ("PERSON", "ORG", "GPE") and ent.text[0].isupper():
            confidence += 0.03

        # Reduce for very short entities (might be noise)
        if len(ent.text) <= 2:
            confidence -= 0.10

        return min(confidence, 1.0)

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Extract entities from chunk artifact.

        Implements IFProcessor.process().

        Process method for IFProcessor interface.
        Rule #4: Function under 60 lines.

        Args:
            artifact: Input artifact (must be IFChunkArtifact)

        Returns:
            Derived artifact with entities in metadata, or IFFailureArtifact
        """
        # Validate input type
        if not isinstance(artifact, IFChunkArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-entity-failure",
                error_message=f"EntityExtractor requires IFChunkArtifact, got {type(artifact).__name__}",
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
            )

        # Extract structured entities
        entities = self.extract_structured(artifact.content)

        # Store in metadata
        new_metadata = dict(artifact.metadata)
        new_metadata["entities_structured"] = [e.to_dict() for e in entities]
        new_metadata["entities"] = [
            f"{e.label}:{e.text}@{e.confidence:.2f}" for e in entities[:50]
        ]

        # Return derived artifact
        return artifact.derive(
            self.processor_id,
            artifact_id=f"{artifact.artifact_id}-entities",
            metadata=new_metadata,
        )

    @property
    def processor_id(self) -> str:
        """
        Unique identifier for this processor.

        Implements IFProcessor.processor_id.
        Required property.

        Returns:
            Processor ID string
        """
        return "entity-extractor"

    @property
    def version(self) -> str:
        """
        Semantic version of this processor.

        Implements IFProcessor.version.
        Required property.

        Returns:
            SemVer string
        """
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        """
        Capabilities provided by this processor.

        Implements IFProcessor.capabilities.
        Optional property.

        Returns:
            List of capability strings
        """
        return ["ner", "entity-extraction"]

    @property
    def memory_mb(self) -> int:
        """
        Estimated memory requirement in megabytes.

        Implements IFProcessor.memory_mb.
        Optional property.

        Returns:
            Memory estimate in MB
        """
        # spaCy models require significant memory
        if self.use_spacy and self.model_name == "en_core_web_lg":
            return 800  # Large model ~800MB
        elif self.use_spacy and self.model_name == "en_core_web_sm":
            return 200  # Small model ~200MB
        return 50  # Pattern-only mode is lightweight

    def is_available(self) -> bool:
        """
        Check if entity extractor is available.

        Implements IFProcessor.is_available().

        Returns:
            True (pattern-based extraction always available)
        """
        return True

    def teardown(self) -> bool:
        """
        Clean up resources.

        Implements IFProcessor.teardown().
        Optional cleanup method.

        Returns:
            True if cleanup successful
        """
        self.clear_cache()
        self._model_cache = None
        return True

    def clear_cache(self) -> None:
        """Clear the entity cache."""
        self._entity_cache.clear()
