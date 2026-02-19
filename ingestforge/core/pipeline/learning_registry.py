"""
Golden Example Registry for Few-Shot Learning.

Implements a local storage system for human-verified extractions
that serve as ground truth for few-shot learning.

Adds semantic similarity matching for intelligent example selection.

Follows NASA JPL Power of Ten rules.
"""

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_EXAMPLES_PER_VERTICAL = 1000
MAX_VERTICALS = 100
MAX_ENTITIES_PER_EXAMPLE = 500
MAX_LINES_ON_LOAD = 100_000

# Semantic matching constants
MAX_SIMILAR_EXAMPLES = 10
MAX_EMBEDDING_DIMENSION = 4096
SIMILARITY_CACHE_SIZE = 128
SIMILARITY_TIMEOUT_MS = 100


def _calculate_sha256(data: str) -> str:
    """
    Calculate SHA-256 hash of a string.

    Rule #4: Function under 60 lines.
    Rule #5: Assert preconditions.
    Rule #7: Explicit return value.
    """
    assert data is not None, "data cannot be None"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Core similarity function for semantic matching.
    Optimized for performance (< 100ms requirement).

    Rule #4: Function under 60 lines.
    Rule #5: Assert preconditions.
    Rule #7: Explicit return value.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity in range [-1, 1].
    """
    assert vec_a is not None, "vec_a cannot be None"
    assert vec_b is not None, "vec_b cannot be None"
    assert len(vec_a) == len(vec_b), "Vectors must have same dimension"
    assert (
        len(vec_a) <= MAX_EMBEDDING_DIMENSION
    ), f"Vector exceeds max dimension {MAX_EMBEDDING_DIMENSION}"

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for a, b in zip(vec_a, vec_b):
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _serialize_single_entity(entity: Any) -> Any:
    """
    Serialize a single entity to a dict.

    Rule #1: Extracted to reduce nesting.
    Rule #4: Function under 60 lines.
    """
    if hasattr(entity, "model_dump"):
        return entity.model_dump()
    if hasattr(entity, "to_dict"):
        return entity.to_dict()
    if isinstance(entity, dict):
        return entity
    return str(entity)


def _serialize_entities(entities: List[Any]) -> str:
    """
    Serialize entities to a canonical JSON string for hashing.

    Rule #4: Function under 60 lines.
    Rule #5: Assert preconditions.
    Rule #7: Explicit return value.

    Args:
        entities: List of entity objects (EntityNode or dicts).

    Returns:
        Canonical JSON string representation.
    """
    assert entities is not None, "entities cannot be None"
    serialized = [_serialize_single_entity(e) for e in entities]
    return json.dumps(serialized, sort_keys=True, ensure_ascii=False)


@dataclass
class GoldenExample:
    """
    A single golden example for few-shot learning.

    Stores human-verified extraction with provenance.
    Optional embedding for semantic similarity matching.

    Rule #9: Complete type hints.

    Attributes:
        example_id: Unique identifier for this example.
        vertical_id: Pipeline vertical this example belongs to.
        entity_type: Primary entity type for filtering.
        chunk_content: Source text chunk content.
        chunk_hash: SHA-256 hash of chunk content for integrity.
        entities: List of extracted entities (serialized).
        entities_hash: SHA-256 hash of serialized entities.
        approved_at: ISO timestamp when example was approved.
        approved_by: User or system that approved the example.
        metadata: Additional metadata for the example.
        embedding: Optional chunk embedding for similarity search.
    """

    example_id: str
    vertical_id: str
    entity_type: str
    chunk_content: str
    chunk_hash: str
    entities: List[Dict[str, Any]]
    entities_hash: str
    approved_at: str
    approved_by: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "example_id": self.example_id,
            "vertical_id": self.vertical_id,
            "entity_type": self.entity_type,
            "chunk_content": self.chunk_content,
            "chunk_hash": self.chunk_hash,
            "entities": self.entities,
            "entities_hash": self.entities_hash,
            "approved_at": self.approved_at,
            "approved_by": self.approved_by,
            "metadata": self.metadata,
        }
        # Include embedding if present
        if self.embedding is not None:
            result["embedding"] = self.embedding
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenExample":
        """Create from dictionary."""
        assert data is not None, "data cannot be None"
        return cls(
            example_id=data["example_id"],
            vertical_id=data["vertical_id"],
            entity_type=data["entity_type"],
            chunk_content=data["chunk_content"],
            chunk_hash=data["chunk_hash"],
            entities=data["entities"],
            entities_hash=data["entities_hash"],
            approved_at=data["approved_at"],
            approved_by=data.get("approved_by", "user"),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )

    def verify_integrity(self) -> bool:
        """
        Verify SHA-256 hashes match the content.

        JPL Rule #10: Verify hash integrity.
        Rule #7: Explicit return value.

        Returns:
            True if both chunk and entities hashes are valid.
        """
        # Verify chunk hash
        computed_chunk_hash = _calculate_sha256(self.chunk_content)
        if computed_chunk_hash != self.chunk_hash:
            logger.warning(
                f"Chunk hash mismatch for example {self.example_id}: "
                f"expected {self.chunk_hash}, got {computed_chunk_hash}"
            )
            return False

        # Verify entities hash
        entities_str = _serialize_entities(self.entities)
        computed_entities_hash = _calculate_sha256(entities_str)
        if computed_entities_hash != self.entities_hash:
            logger.warning(
                f"Entities hash mismatch for example {self.example_id}: "
                f"expected {self.entities_hash}, got {computed_entities_hash}"
            )
            return False

        return True


class IFExampleRegistry:
    """
    Thread-safe singleton registry for Golden Examples.

    Stores human-verified extractions for few-shot learning.

    JPL Rule #2: Fixed upper bounds on examples per vertical.
    JPL Rule #10: SHA-256 verification before saving.

    Attributes:
        _instance: Singleton instance.
        _lock: Thread lock for safe access.
        _data_path: Path to the examples.jsonl file.
        _examples: In-memory cache of examples by vertical.
        _vertical_counts: Count of examples per vertical.
    """

    _instance: Optional["IFExampleRegistry"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, data_path: Optional[Path] = None) -> "IFExampleRegistry":
        """Thread-safe singleton constructor."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, data_path: Optional[Path] = None) -> None:
        """Initialize the registry (runs once)."""
        with self._lock:
            if self._initialized:
                return

            if data_path is None:
                data_path = Path(".data") / "learning"

            self._data_path: Path = data_path
            self._examples_file: Path = data_path / "examples.jsonl"
            self._examples: Dict[str, List[GoldenExample]] = {}
            self._vertical_counts: Dict[str, int] = {}
            # Similarity cache for document-level queries
            self._similarity_cache: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
            self._cache_document_id: Optional[str] = None

            self._data_path.mkdir(parents=True, exist_ok=True)
            self._load()
            self._initialized = True

    # -------------------------------------------------------------------------
    # Rule #1: Extracted helpers for _load() to reduce nesting
    # -------------------------------------------------------------------------

    def _parse_example_line(self, line: str) -> Optional[GoldenExample]:
        """
        Parse a single JSONL line into a GoldenExample.

        Rule #1: Extracted helper to reduce nesting.
        Rule #4: Function under 60 lines.
        """
        stripped = line.strip()
        if not stripped:
            return None

        try:
            data = json.loads(stripped)
            example = GoldenExample.from_dict(data)
            if not example.verify_integrity():
                logger.warning(f"Skipping corrupted example: {example.example_id}")
                return None
            return example
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to parse example line: {e}")
            return None

    def _add_example_to_cache(self, example: GoldenExample) -> None:
        """
        Add an example to the in-memory cache.

        Rule #1: Extracted helper to reduce nesting.
        Rule #4: Function under 60 lines.
        """
        vertical = example.vertical_id
        if vertical not in self._examples:
            self._examples[vertical] = []
            self._vertical_counts[vertical] = 0

        self._examples[vertical].append(example)
        self._vertical_counts[vertical] += 1

    def _load(self) -> None:
        """
        Load examples from JSONL file.

        Rule #1: Linear control flow (nesting reduced via helpers).
        Rule #2: Fixed upper bounds.
        Rule #4: Function under 60 lines.
        """
        if not self._examples_file.exists():
            logger.debug(f"No existing examples file at {self._examples_file}")
            return

        lines_read = 0

        try:
            with open(self._examples_file, "r", encoding="utf-8") as f:
                for line in f:
                    lines_read += 1
                    if lines_read > MAX_LINES_ON_LOAD:
                        logger.error(f"Safety limit reached: {MAX_LINES_ON_LOAD} lines")
                        break

                    example = self._parse_example_line(line)
                    if example is not None:
                        self._add_example_to_cache(example)

            logger.info(f"Loaded {sum(self._vertical_counts.values())} golden examples")

        except Exception as e:
            logger.error(f"Failed to load examples: {e}")

    # -------------------------------------------------------------------------
    # Rule #1: Extracted helpers for _atomic_append() to reduce nesting
    # -------------------------------------------------------------------------

    def _write_temp_file(self, example: GoldenExample) -> Tuple[int, str]:
        """
        Write example to temp file for atomic save.

        Rule #1: Extracted helper to reduce nesting.
        Rule #4: Function under 60 lines.

        Returns:
            Tuple of (file_descriptor, temp_path).
        """
        fd, temp_path = tempfile.mkstemp(
            dir=self._data_path,
            prefix=".example_",
            suffix=".tmp",
        )

        # Copy existing content if file exists
        if self._examples_file.exists():
            with open(self._examples_file, "rb") as src:
                os.write(fd, src.read())

        # Append new example
        line = json.dumps(example.to_dict(), ensure_ascii=False) + "\n"
        os.write(fd, line.encode("utf-8"))

        return fd, temp_path

    def _atomic_append(self, example: GoldenExample) -> bool:
        """
        Atomically append an example to the JSONL file.

        Rule #1: Linear control flow (reduced nesting).
        Rule #4: Function under 60 lines.
        Rule #5: Assert preconditions.
        Rule #7: Explicit return value.
        """
        assert example is not None, "example cannot be None"
        temp_path: Optional[str] = None

        try:
            fd, temp_path = self._write_temp_file(example)
            os.close(fd)
            os.replace(temp_path, self._examples_file)
            return True

        except Exception as e:
            logger.error(f"Failed to save example: {e}")
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            return False

    # -------------------------------------------------------------------------
    # Rule #4: Split save_example into smaller helpers
    # -------------------------------------------------------------------------

    def _validate_save_limits(
        self,
        entities: List[Any],
        vertical_id: str,
    ) -> Optional[str]:
        """
        Validate limits before saving an example.

        Rule #1: Extracted helper.
        Rule #4: Function under 60 lines.

        Returns:
            Error message if validation fails, None if OK.
        """
        # Check vertical limit
        current_count = self._vertical_counts.get(vertical_id, 0)
        if current_count >= MAX_EXAMPLES_PER_VERTICAL:
            return f"Vertical '{vertical_id}' has reached limit of {MAX_EXAMPLES_PER_VERTICAL}"

        # Check total verticals limit
        if (
            vertical_id not in self._vertical_counts
            and len(self._vertical_counts) >= MAX_VERTICALS
        ):
            return f"Maximum verticals limit reached: {MAX_VERTICALS}"

        # Check entities limit
        if len(entities) > MAX_ENTITIES_PER_EXAMPLE:
            return f"Too many entities ({len(entities)}), max is {MAX_ENTITIES_PER_EXAMPLE}"

        return None

    def _serialize_entity_list(self, entities: List[Any]) -> List[Dict[str, Any]]:
        """
        Serialize a list of entities for storage.

        Rule #1: Extracted helper to reduce nesting.
        Rule #4: Function under 60 lines.
        """
        serialized: List[Dict[str, Any]] = []
        for entity in entities:
            if hasattr(entity, "model_dump"):
                serialized.append(entity.model_dump())
            elif hasattr(entity, "to_dict"):
                serialized.append(entity.to_dict())
            elif isinstance(entity, dict):
                serialized.append(entity)
            else:
                serialized.append({"raw": str(entity)})
        return serialized

    def _build_example(
        self,
        chunk_content: str,
        entities: List[Any],
        vertical_id: str,
        entity_type: str,
        approved_by: str,
        metadata: Optional[Dict[str, Any]],
    ) -> GoldenExample:
        """
        Build a GoldenExample with calculated hashes.

        Rule #1: Extracted helper.
        Rule #4: Function under 60 lines.
        Rule #5: Assert postconditions.
        """
        chunk_hash = _calculate_sha256(chunk_content)
        entities_str = _serialize_entities(entities)
        entities_hash = _calculate_sha256(entities_str)
        serialized_entities = self._serialize_entity_list(entities)

        timestamp = datetime.now(timezone.utc).isoformat()
        example_id = f"{vertical_id}_{chunk_hash[:8]}_{entities_hash[:8]}"

        example = GoldenExample(
            example_id=example_id,
            vertical_id=vertical_id,
            entity_type=entity_type,
            chunk_content=chunk_content,
            chunk_hash=chunk_hash,
            entities=serialized_entities,
            entities_hash=entities_hash,
            approved_at=timestamp,
            approved_by=approved_by,
            metadata=metadata or {},
        )

        # Postcondition: example should verify its own integrity
        assert example.verify_integrity(), "Newly built example failed integrity check"
        return example

    def save_example(
        self,
        chunk_content: str,
        entities: List[Any],
        vertical_id: str = "default",
        entity_type: str = "general",
        approved_by: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Save a golden example to the registry.

        Stores human-verified extraction with hash verification.

        Rule #4: Function under 60 lines.
        Rule #5: Assert preconditions.
        Rule #7: Explicit return value.

        Returns:
            Example ID if saved successfully, None if limit reached or error.
        """
        # Rule #5: Preconditions
        assert chunk_content is not None, "chunk_content cannot be None"
        assert entities is not None, "entities cannot be None"

        if not chunk_content.strip():
            logger.warning("chunk_content cannot be empty")
            return None

        with self._lock:
            # Validate limits
            error = self._validate_save_limits(entities, vertical_id)
            if error:
                logger.warning(error)
                return None

            # Build example with hashes
            example = self._build_example(
                chunk_content, entities, vertical_id, entity_type, approved_by, metadata
            )

            # Atomic save
            if not self._atomic_append(example):
                return None

            # Update cache
            self._add_example_to_cache(example)

            # Rule #5: Postcondition
            assert self._vertical_counts[vertical_id] <= MAX_EXAMPLES_PER_VERTICAL

            logger.info(
                f"Saved golden example {example.example_id} for vertical '{vertical_id}'"
            )
            return example.example_id

    # -------------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------------

    def list_examples(
        self,
        vertical_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[GoldenExample]:
        """
        List golden examples with optional filtering.

        Rule #4: Function under 60 lines.
        Rule #5: Assert preconditions.
        """
        assert limit > 0, "limit must be positive"

        with self._lock:
            results: List[GoldenExample] = []
            verticals = [vertical_id] if vertical_id in self._examples else []
            if vertical_id is None:
                verticals = list(self._examples.keys())

            for v in verticals:
                for example in self._examples.get(v, []):
                    if entity_type is not None and example.entity_type != entity_type:
                        continue
                    results.append(example)
                    if len(results) >= limit:
                        return results

            return results

    def get_example(self, example_id: str) -> Optional[GoldenExample]:
        """Get a specific example by ID."""
        assert example_id, "example_id cannot be empty"

        with self._lock:
            for examples in self._examples.values():
                for example in examples:
                    if example.example_id == example_id:
                        return example
            return None

    def count_examples(self, vertical_id: Optional[str] = None) -> int:
        """Count examples, optionally filtered by vertical."""
        with self._lock:
            if vertical_id is not None:
                return self._vertical_counts.get(vertical_id, 0)
            return sum(self._vertical_counts.values())

    def get_verticals(self) -> List[str]:
        """Get list of all verticals with examples."""
        with self._lock:
            return sorted(self._examples.keys())

    # -------------------------------------------------------------------------
    # Semantic Similarity Methods
    # -------------------------------------------------------------------------

    def _get_examples_with_embeddings(
        self,
        vertical_id: Optional[str] = None,
    ) -> List[GoldenExample]:
        """
        Get examples that have embeddings.

        Rule #1: Extracted helper.
        Rule #4: Function under 60 lines.
        """
        examples: List[GoldenExample] = []
        verticals = [vertical_id] if vertical_id else list(self._examples.keys())

        for v in verticals:
            for example in self._examples.get(v, []):
                if example.embedding is not None:
                    examples.append(example)

        return examples

    def _select_diverse_examples(
        self,
        scored: List[Tuple[float, GoldenExample]],
        limit: int,
    ) -> List[GoldenExample]:
        """
        Select diverse examples from scored candidates.

        Ensures selected examples cover different entity types.

        Rule #1: Extracted helper.
        Rule #4: Function under 60 lines.
        """
        selected: List[GoldenExample] = []
        seen_types: Dict[str, int] = {}

        # Sort by similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)

        for score, example in scored:
            entity_type = example.entity_type
            type_count = seen_types.get(entity_type, 0)

            # Favor diversity: limit examples per type
            if type_count < 2 or len(selected) < limit // 2:
                selected.append(example)
                seen_types[entity_type] = type_count + 1

            if len(selected) >= limit:
                break

        return selected

    def _build_cache_key(
        self, chunk_embedding: List[float], document_id: Optional[str]
    ) -> Optional[str]:
        """Build cache key for similarity query. Rule #1: Extracted helper."""
        if document_id is None:
            return None
        return f"{document_id}:{hash(tuple(chunk_embedding[:10]))}"

    def _update_cache(
        self, cache_key: Optional[str], results: List[Tuple[str, Dict[str, Any]]]
    ) -> None:
        """Update similarity cache with FIFO eviction. Rule #1: Extracted helper."""
        if cache_key is None:
            return
        self._similarity_cache[cache_key] = results
        if len(self._similarity_cache) > SIMILARITY_CACHE_SIZE:
            oldest = next(iter(self._similarity_cache))
            del self._similarity_cache[oldest]

    def _score_examples(
        self, chunk_embedding: List[float], examples: List[GoldenExample]
    ) -> List[Tuple[float, GoldenExample]]:
        """Score examples by cosine similarity. Rule #1: Extracted helper."""
        assert examples is not None, "examples cannot be None"
        scored: List[Tuple[float, GoldenExample]] = []
        for example in examples:
            score = _cosine_similarity(chunk_embedding, example.embedding)
            scored.append((score, example))
        return scored

    def find_similar(
        self,
        chunk_embedding: List[float],
        limit: int = 3,
        vertical_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Find semantically similar golden examples for few-shot injection.

        Rule #4: Function under 60 lines. Rule #5: Assert preconditions.
        """
        assert chunk_embedding is not None, "chunk_embedding cannot be None"
        assert len(chunk_embedding) > 0, "chunk_embedding cannot be empty"
        assert limit > 0, "limit must be positive"
        assert (
            limit <= MAX_SIMILAR_EXAMPLES
        ), f"limit exceeds max {MAX_SIMILAR_EXAMPLES}"

        start_time = time.time()
        cache_key = self._build_cache_key(chunk_embedding, document_id)

        if cache_key and cache_key in self._similarity_cache:
            logger.debug(f"Cache hit for similarity query: {cache_key}")
            return self._similarity_cache[cache_key][:limit]

        with self._lock:
            examples = self._get_examples_with_embeddings(vertical_id)
            if not examples:
                logger.debug("No examples with embeddings found")
                return []

            scored = self._score_examples(chunk_embedding, examples)
            selected = self._select_diverse_examples(scored, limit)
            results: List[Tuple[str, Dict[str, Any]]] = [
                (ex.chunk_content, {"entities": ex.entities}) for ex in selected
            ]

        self._update_cache(cache_key, results)

        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > SIMILARITY_TIMEOUT_MS:
            logger.warning(f"Similarity search took {elapsed_ms:.1f}ms")

        return results

    def clear_similarity_cache(self) -> None:
        """Clear the similarity cache."""
        self._similarity_cache.clear()
        self._cache_document_id = None
        logger.debug("Cleared similarity cache")

    def set_example_embedding(
        self,
        example_id: str,
        embedding: List[float],
    ) -> bool:
        """
        Set the embedding for an existing example.

        Allows adding embeddings to examples after creation.

        Rule #5: Assert preconditions.
        Rule #7: Explicit return value.

        Args:
            example_id: ID of the example to update.
            embedding: Embedding vector to store.

        Returns:
            True if example was updated, False if not found.
        """
        assert example_id, "example_id cannot be empty"
        assert embedding is not None, "embedding cannot be None"
        assert len(embedding) > 0, "embedding cannot be empty"
        assert (
            len(embedding) <= MAX_EMBEDDING_DIMENSION
        ), "embedding exceeds max dimension"

        with self._lock:
            for examples in self._examples.values():
                for example in examples:
                    if example.example_id == example_id:
                        example.embedding = embedding
                        logger.debug(f"Set embedding for example {example_id}")
                        return True

        logger.warning(f"Example not found: {example_id}")
        return False

    def clear(self) -> None:
        """Clear all examples (for testing)."""
        with self._lock:
            self._examples.clear()
            self._vertical_counts.clear()
            self._similarity_cache.clear()
            self._cache_document_id = None
            if self._examples_file.exists():
                self._examples_file.unlink()
            logger.info("Cleared all golden examples")

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False
