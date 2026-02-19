"""
Data models for multi-document session management.

Provides data structures for sessions, documents, references, and collections.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ReferenceType(Enum):
    """Types of cross-document references."""

    CITATION = "citation"  # Document A cites Document B
    RELATED = "related"  # Documents are topically related
    CONTRADICTS = "contradicts"  # Documents have contradicting claims
    SUPPORTS = "supports"  # Document A supports claims in B
    SUPERSEDES = "supersedes"  # Document A replaces/updates B
    DERIVED = "derived"  # Document A is derived from B
    MENTIONS = "mentions"  # Document A mentions B
    SAME_AUTHOR = "same_author"  # Documents share authorship
    SAME_TOPIC = "same_topic"  # Documents share topic/theme
    SEQUENTIAL = "sequential"  # Documents are part of a sequence


@dataclass
class DocumentReference:
    """A reference between two documents."""

    source_doc_id: str
    target_doc_id: str
    reference_type: ReferenceType
    confidence: float = 1.0  # 0.0 to 1.0
    context: str = ""  # Text surrounding the reference
    location: Optional[str] = None  # Location in source doc (page, section)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    auto_detected: bool = False  # Whether this was auto-detected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_doc_id": self.source_doc_id,
            "target_doc_id": self.target_doc_id,
            "reference_type": self.reference_type.value,
            "confidence": self.confidence,
            "context": self.context,
            "location": self.location,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "auto_detected": self.auto_detected,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentReference":
        return cls(
            source_doc_id=data["source_doc_id"],
            target_doc_id=data["target_doc_id"],
            reference_type=ReferenceType(data["reference_type"]),
            confidence=data.get("confidence", 1.0),
            context=data.get("context", ""),
            location=data.get("location"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            auto_detected=data.get("auto_detected", False),
        )


@dataclass
class SessionDocument:
    """A document within a multi-document session."""

    id: str
    name: str
    path: str
    file_type: str
    status: str = "pending"  # pending, processing, completed, failed
    added_at: str = ""
    processed_at: Optional[str] = None
    word_count: int = 0
    page_count: int = 0
    # Metadata from document
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    # Analysis results
    summary: Optional[str] = None
    key_concepts: List[str] = field(default_factory=list)
    # Collections/groups this document belongs to
    collections: List[str] = field(default_factory=list)
    # User annotations
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    # Processing metadata
    chunks_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionDocument":
        return cls(**data)


@dataclass
class DocumentCollection:
    """A named collection/group of documents within a session."""

    id: str
    name: str
    description: str = ""
    document_ids: List[str] = field(default_factory=list)
    color: Optional[str] = None  # UI color
    icon: Optional[str] = None  # UI icon
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentCollection":
        return cls(**data)


@dataclass
class SharedContext:
    """Shared context/state across all documents in a session."""

    research_question: Optional[str] = None
    thesis_statement: Optional[str] = None
    key_themes: List[str] = field(default_factory=list)
    glossary: Dict[str, str] = field(default_factory=dict)  # term -> definition
    notes: str = ""
    outline: List[str] = field(default_factory=list)
    bibliography_style: str = "apa"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharedContext":
        return cls(**data)


@dataclass
class MultiDocumentSession:
    """
    A session containing multiple documents with cross-references.
    """

    id: str
    name: str
    description: str = ""
    created_at: str = ""
    modified_at: str = ""
    status: str = "active"  # active, paused, completed, archived

    # Documents
    documents: Dict[str, SessionDocument] = field(default_factory=dict)

    # Cross-document references
    references: List[DocumentReference] = field(default_factory=list)

    # Document collections/groups
    collections: Dict[str, DocumentCollection] = field(default_factory=dict)

    # Shared context
    shared_context: SharedContext = field(default_factory=SharedContext)

    # Session metadata
    tags: List[str] = field(default_factory=list)
    working_directory: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "status": self.status,
            "documents": {k: v.to_dict() for k, v in self.documents.items()},
            "references": [r.to_dict() for r in self.references],
            "collections": {k: v.to_dict() for k, v in self.collections.items()},
            "shared_context": self.shared_context.to_dict(),
            "tags": self.tags,
            "working_directory": self.working_directory,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiDocumentSession":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data["created_at"],
            modified_at=data["modified_at"],
            status=data.get("status", "active"),
            documents={
                k: SessionDocument.from_dict(v)
                for k, v in data.get("documents", {}).items()
            },
            references=[
                DocumentReference.from_dict(r) for r in data.get("references", [])
            ],
            collections={
                k: DocumentCollection.from_dict(v)
                for k, v in data.get("collections", {}).items()
            },
            shared_context=SharedContext.from_dict(data.get("shared_context", {})),
            tags=data.get("tags", []),
            working_directory=data.get("working_directory"),
        )
