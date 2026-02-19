"""
Knowledge Manifest for Cross-Document Entity Linking.

Knowledge Manifest
Tracks extracted entities during batch ingestion for automatic cross-document linking.

NASA JPL Power of Ten compliant.
"""

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import Field

from ingestforge.core.pipeline.interfaces import IFArtifact
from ingestforge.core.pipeline.artifacts import IFTextArtifact
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
# AC: 10,000 entity limit per session
MAX_ENTITIES_IN_MANIFEST = 10000
MAX_REFERENCES_PER_ENTITY = 1000
MAX_JOIN_ARTIFACTS_PER_SESSION = 50000


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class EntityReference:
    """
    Reference to an entity occurrence in a document.

    Rule #9: Complete type hints.
    """

    document_id: str
    artifact_id: str
    chunk_id: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "artifact_id": self.artifact_id,
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
        }


@dataclass
class ManifestEntry:
    """
    Entry for an entity in the manifest.

    Rule #2: Bounded references list.
    Rule #9: Complete type hints.
    """

    entity_hash: str
    entity_text: str
    entity_type: str
    references: List[EntityReference] = field(default_factory=list)
    first_seen_document: Optional[str] = None
    first_seen_artifact: Optional[str] = None

    def add_reference(self, ref: EntityReference) -> bool:
        """
        Add a reference to this entity.

        Rule #2: Bounded list size.

        Returns:
            True if added, False if limit reached.
        """
        if len(self.references) >= MAX_REFERENCES_PER_ENTITY:
            return False
        self.references.append(ref)
        return True

    @property
    def document_count(self) -> int:
        """Number of unique documents containing this entity."""
        return len(set(ref.document_id for ref in self.references))

    @property
    def is_cross_document(self) -> bool:
        """True if entity appears in multiple documents."""
        return self.document_count > 1

    def get_documents(self) -> List[str]:
        """List of unique document IDs containing this entity."""
        return sorted(set(ref.document_id for ref in self.references))


class IFJoinArtifact(IFTextArtifact):
    """
    Artifact representing a cross-document entity link.

    Auto-generated when cross-document entities are found.
    """

    join_type: str = Field("entity_link", description="Type of join relationship")
    source_document: str = Field(..., description="First document ID")
    target_document: str = Field(..., description="Second document ID")
    linked_entity: str = Field(..., description="Entity text that creates the link")
    entity_type: str = Field(..., description="Type of the linked entity")
    source_artifact: str = Field(..., description="Artifact ID in source document")
    target_artifact: str = Field(..., description="Artifact ID in target document")

    def derive(self, processor_id: str, **kwargs: Any) -> "IFJoinArtifact":
        """Create a derived join artifact."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )


# ---------------------------------------------------------------------------
# Knowledge Manifest Implementation
# ---------------------------------------------------------------------------


class IFKnowledgeManifest:
    """
    Thread-safe singleton for tracking entities during batch ingestion.

    Knowledge Manifest
    - O(1) lookup for existing entities (SHA-256 hash keys)
    - Auto-generates Join artifacts for cross-document entities
    - Thread-safe for concurrent batch processing

    NASA JPL Power of Ten compliant.
    Rule #2: Fixed upper bounds on entities.
    Rule #9: Complete type hints.
    """

    _instance: Optional["IFKnowledgeManifest"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "IFKnowledgeManifest":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialize()
                    cls._instance = instance
        return cls._instance

    def _initialize(self) -> None:
        """Initialize manifest state."""
        self._entities: Dict[str, ManifestEntry] = {}
        self._join_artifacts: List[IFJoinArtifact] = []
        self._data_lock: threading.RLock = threading.RLock()
        self._session_active: bool = False
        self._processed_documents: Set[str] = set()

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance for testing.

        Call between test cases or batch sessions.
        """
        with cls._lock:
            cls._instance = None

    def start_session(self) -> bool:
        """
        Start a new batch session.

        Clears all existing data and marks session as active.

        Returns:
            True if session started successfully.
        """
        with self._data_lock:
            self._entities.clear()
            self._join_artifacts.clear()
            self._processed_documents.clear()
            self._session_active = True
            logger.info("Knowledge manifest session started")
            return True

    def end_session(self) -> Tuple[int, int]:
        """
        End the current batch session.

        Returns:
            Tuple of (entity_count, join_artifact_count).
        """
        with self._data_lock:
            entity_count = len(self._entities)
            join_count = len(self._join_artifacts)
            self._session_active = False
            logger.info(
                f"Knowledge manifest session ended: "
                f"{entity_count} entities, {join_count} joins"
            )
            return (entity_count, join_count)

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self._session_active

    @staticmethod
    def _hash_entity(text: str, entity_type: str) -> str:
        """
        Generate SHA-256 hash key for entity.

        Rule #4: Function < 60 lines.

        Args:
            text: Normalized entity text
            entity_type: Entity type string

        Returns:
            SHA-256 hash as hex string
        """
        normalized = f"{entity_type}:{text.lower().strip()}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _build_reference(
        self,
        document_id: str,
        artifact_id: str,
        chunk_id: Optional[str],
        start_char: int,
        end_char: int,
        confidence: float,
    ) -> EntityReference:
        """Build EntityReference object. Rule #4: Helper < 60 lines."""
        return EntityReference(
            document_id=document_id,
            artifact_id=artifact_id,
            chunk_id=chunk_id,
            start_char=start_char,
            end_char=end_char,
            confidence=confidence,
        )

    def _handle_existing_entity(
        self,
        entry: ManifestEntry,
        ref: EntityReference,
        document_id: str,
        entity_text: str,
    ) -> Tuple[bool, Optional[List[str]]]:
        """Handle registration for existing entity. Rule #4: Helper < 60 lines."""
        existing_docs = entry.get_documents()
        if not entry.add_reference(ref):
            logger.debug(f"Reference limit reached for entity: {entity_text}")
        if document_id not in existing_docs:
            self._create_join_artifacts(entry, ref, existing_docs)
            return (False, existing_docs)
        return (False, None)

    def register_entity(
        self,
        entity_text: str,
        entity_type: str,
        document_id: str,
        artifact_id: str,
        chunk_id: Optional[str] = None,
        start_char: int = 0,
        end_char: int = 0,
        confidence: float = 1.0,
    ) -> Tuple[bool, Optional[List[str]]]:
        """
        Register an entity occurrence. O(1) lookup via hash key.
        Rule #4: Function < 60 lines (uses helpers).
        Rule #5: Assertions for parameter validation.
        """
        # JPL Rule #5: Assert preconditions
        assert document_id, "document_id must be non-empty"
        assert artifact_id, "artifact_id must be non-empty"
        assert 0.0 <= confidence <= 1.0, "confidence must be in [0.0, 1.0]"
        assert start_char >= 0, "start_char must be non-negative"
        assert end_char >= start_char, "end_char must be >= start_char"

        with self._data_lock:
            if not self._session_active:
                logger.warning("Cannot register entity: no active session")
                return (False, None)
            if len(self._entities) >= MAX_ENTITIES_IN_MANIFEST:
                logger.warning(
                    f"Manifest entity limit reached ({MAX_ENTITIES_IN_MANIFEST})"
                )
                return (False, None)

            entity_hash = self._hash_entity(entity_text, entity_type)
            ref = self._build_reference(
                document_id, artifact_id, chunk_id, start_char, end_char, confidence
            )

            if entity_hash in self._entities:
                return self._handle_existing_entity(
                    self._entities[entity_hash], ref, document_id, entity_text
                )

            entry = ManifestEntry(
                entity_hash=entity_hash,
                entity_text=entity_text,
                entity_type=entity_type,
                references=[ref],
                first_seen_document=document_id,
                first_seen_artifact=artifact_id,
            )
            self._entities[entity_hash] = entry
            return (True, None)

    def _create_join_artifacts(
        self, entry: ManifestEntry, new_ref: EntityReference, existing_docs: List[str]
    ) -> None:
        """
        Create join artifacts for cross-document entity links.

        Rule #2: Bounded join artifacts.
        Rule #4: Function < 60 lines.
        """
        if len(self._join_artifacts) >= MAX_JOIN_ARTIFACTS_PER_SESSION:
            logger.warning("Join artifact limit reached")
            return

        # Create join for each existing document
        for existing_doc in existing_docs[:10]:  # Limit to first 10 docs
            if len(self._join_artifacts) >= MAX_JOIN_ARTIFACTS_PER_SESSION:
                break

            # Find the first reference in this document
            existing_ref = next(
                (r for r in entry.references if r.document_id == existing_doc), None
            )
            if not existing_ref:
                continue

            # Create join artifact
            join_artifact = IFJoinArtifact(
                artifact_id=f"join-{entry.entity_hash[:16]}-{len(self._join_artifacts)}",
                content=f"Entity '{entry.entity_text}' ({entry.entity_type}) links documents",
                join_type="entity_link",
                source_document=existing_doc,
                target_document=new_ref.document_id,
                linked_entity=entry.entity_text,
                entity_type=entry.entity_type,
                source_artifact=existing_ref.artifact_id,
                target_artifact=new_ref.artifact_id,
                metadata={
                    "entity_hash": entry.entity_hash,
                    "source_chunk": existing_ref.chunk_id,
                    "target_chunk": new_ref.chunk_id,
                    "confidence": min(existing_ref.confidence, new_ref.confidence),
                },
            )
            self._join_artifacts.append(join_artifact)
            logger.debug(
                f"Created join: {existing_doc} <-> {new_ref.document_id} "
                f"via '{entry.entity_text}'"
            )

    def lookup_entity(
        self, entity_text: str, entity_type: str
    ) -> Optional[ManifestEntry]:
        """
        O(1) lookup for an entity.

        Args:
            entity_text: Entity text to find
            entity_type: Entity type

        Returns:
            ManifestEntry if found, None otherwise

        Rule #5: Assertions for parameter validation.
        """
        assert entity_type, "entity_type must be non-empty"

        with self._data_lock:
            entity_hash = self._hash_entity(entity_text, entity_type)
            return self._entities.get(entity_hash)

    def get_all_entities(self) -> List[ManifestEntry]:
        """
        Get all entities in the manifest.

        Proactive Scout
        Returns all entities for gap analysis.

        Returns:
            List of all ManifestEntry objects
        """
        with self._data_lock:
            return list(self._entities.values())

    def get_cross_document_entities(self) -> List[ManifestEntry]:
        """
        Get all entities that appear in multiple documents.

        Returns:
            List of ManifestEntry objects with document_count > 1
        """
        with self._data_lock:
            return [
                entry for entry in self._entities.values() if entry.is_cross_document
            ]

    def get_join_artifacts(self) -> List[IFJoinArtifact]:
        """
        Get all generated join artifacts.

        Returns:
            List of IFJoinArtifact objects
        """
        with self._data_lock:
            return list(self._join_artifacts)

    def get_relationships(self) -> List[IFJoinArtifact]:
        """
        Get all found cross-document relationships.

        AC: Provides get_relationships() method.
        Alias for get_join_artifacts() for API clarity.

        Returns:
            List of IFJoinArtifact representing cross-document links
        """
        return self.get_join_artifacts()

    def get_document_links(self, document_id: str) -> List[Tuple[str, str, str]]:
        """
        Get all documents linked to the specified document.

        Args:
            document_id: The document to find links for

        Returns:
            List of (linked_document_id, entity_text, entity_type) tuples
        """
        with self._data_lock:
            links: List[Tuple[str, str, str]] = []
            for entry in self._entities.values():
                if entry.document_count < 2:
                    continue
                docs = entry.get_documents()
                if document_id in docs:
                    for doc in docs:
                        if doc != document_id:
                            links.append((doc, entry.entity_text, entry.entity_type))
            return links

    @property
    def entity_count(self) -> int:
        """Number of unique entities in manifest."""
        with self._data_lock:
            return len(self._entities)

    @property
    def join_count(self) -> int:
        """Number of join artifacts generated."""
        with self._data_lock:
            return len(self._join_artifacts)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get manifest statistics.

        Returns:
            Dictionary with statistics
        """
        with self._data_lock:
            cross_doc_entities = self.get_cross_document_entities()
            return {
                "session_active": self._session_active,
                "total_entities": len(self._entities),
                "cross_document_entities": len(cross_doc_entities),
                "total_references": sum(
                    len(e.references) for e in self._entities.values()
                ),
                "join_artifacts": len(self._join_artifacts),
                "documents_processed": len(self._processed_documents),
            }

    def mark_document_processed(self, document_id: str) -> None:
        """Mark a document as processed in this session."""
        with self._data_lock:
            self._processed_documents.add(document_id)

    def is_document_processed(self, document_id: str) -> bool:
        """Check if a document has been processed in this session."""
        with self._data_lock:
            return document_id in self._processed_documents


# ---------------------------------------------------------------------------
# Post-Extraction Hook Integration
# ---------------------------------------------------------------------------


def register_extracted_entities(
    artifact: IFArtifact,
    document_id: str,
    manifest: Optional[IFKnowledgeManifest] = None,
) -> List[str]:
    """
    Post-extraction hook to register entities from an artifact.

    Integrates with IFPipelineRunner as a post-extraction callback.

    Args:
        artifact: Artifact containing extracted entities in metadata
        document_id: Document identifier
        manifest: Optional manifest instance (uses singleton if not provided)

    Returns:
        List of document IDs that this artifact links to
    """
    if manifest is None:
        manifest = IFKnowledgeManifest()

    if not manifest.is_active:
        return []

    # Get entities from artifact metadata
    entities = artifact.metadata.get("entities_structured", [])
    if not entities:
        return []

    linked_docs: Set[str] = set()

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        is_new, existing_docs = manifest.register_entity(
            entity_text=entity.get("text", ""),
            entity_type=entity.get("type", "UNKNOWN"),
            document_id=document_id,
            artifact_id=artifact.artifact_id,
            chunk_id=artifact.metadata.get("chunk_id"),
            start_char=entity.get("start", 0),
            end_char=entity.get("end", 0),
            confidence=entity.get("confidence", 1.0),
        )

        if existing_docs:
            linked_docs.update(existing_docs)

    # Mark document as processed
    manifest.mark_document_processed(document_id)

    return list(linked_docs)
