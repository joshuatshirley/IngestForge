"""
Multi-document session manager.

Provides MultiDocumentSessionManager for managing multi-document sessions.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ingestforge.ingest.multi_document_session.models import (
    DocumentCollection,
    DocumentReference,
    MultiDocumentSession,
    ReferenceType,
    SessionDocument,
    SharedContext,
)


class MultiDocumentSessionManager:
    """
    Manages multi-document sessions with cross-document references.
    """

    def __init__(self, sessions_dir: Optional[Path] = None) -> None:
        self.sessions_dir = sessions_dir or self._get_default_sessions_dir()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._current_session: Optional[MultiDocumentSession] = None

    def _get_default_sessions_dir(self) -> Path:
        """Get default sessions directory."""
        return Path.home() / ".splitanalyze" / "multi_sessions"

    def _generate_id(self) -> str:
        """Generate unique ID."""
        import uuid

        return uuid.uuid4().hex[:12]

    def _get_session_path(self, session_id: str) -> Path:
        """Get path to session file."""
        return self.sessions_dir / f"{session_id}.json"

    def _now(self) -> str:
        """Get current timestamp."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ============== Session Management ==============

    def create_session(
        self,
        name: str,
        description: str = "",
        working_directory: Optional[str] = None,
    ) -> MultiDocumentSession:
        """Create a new multi-document session."""
        session = MultiDocumentSession(
            id=self._generate_id(),
            name=name,
            description=description,
            created_at=self._now(),
            modified_at=self._now(),
            working_directory=working_directory,
        )

        self._current_session = session
        self.save_session(session)
        return session

    def save_session(self, session: Optional[MultiDocumentSession] = None) -> bool:
        """Save session to disk."""
        session = session or self._current_session
        if not session:
            return False

        session.modified_at = self._now()
        session_path = self._get_session_path(session.id)

        try:
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    def load_session(self, session_id: str) -> Optional[MultiDocumentSession]:
        """Load a session from disk."""
        session_path = self._get_session_path(session_id)

        if not session_path.exists():
            return None

        try:
            with open(session_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            session = MultiDocumentSession.from_dict(data)
            self._current_session = session
            return session
        except Exception as e:
            print(f"Error loading session: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session_path = self._get_session_path(session_id)
        if not session_path.exists():
            return False

        try:
            session_path.unlink()
            if self._current_session and self._current_session.id == session_id:
                self._current_session = None
            return True
        except Exception:
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all multi-document sessions."""
        sessions = []

        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append(
                    {
                        "id": data["id"],
                        "name": data["name"],
                        "description": data.get("description", ""),
                        "created_at": data["created_at"],
                        "modified_at": data["modified_at"],
                        "status": data.get("status", "active"),
                        "document_count": len(data.get("documents", {})),
                        "reference_count": len(data.get("references", [])),
                        "collection_count": len(data.get("collections", {})),
                    }
                )
            except Exception:
                continue

        sessions.sort(key=lambda s: s["modified_at"], reverse=True)
        return sessions

    def get_current_session(self) -> Optional[MultiDocumentSession]:
        """Get current session."""
        return self._current_session

    # ============== Document Management ==============

    def add_document(
        self,
        path: str,
        name: Optional[str] = None,
        file_type: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SessionDocument]:
        """Add a document to a session."""
        session = self._get_session(session_id)
        if not session:
            return None

        file_path = Path(path)
        doc_id = self._generate_id()

        doc = SessionDocument(
            id=doc_id,
            name=name or file_path.name,
            path=str(file_path),
            file_type=file_type or file_path.suffix.lower().lstrip("."),
            added_at=self._now(),
        )

        if metadata:
            doc.title = metadata.get("title")
            doc.authors = metadata.get("authors", [])
            doc.publication_date = metadata.get("publication_date")
            doc.abstract = metadata.get("abstract")
            doc.keywords = metadata.get("keywords", [])

        session.documents[doc_id] = doc
        self.save_session(session)
        return doc

    def remove_document(
        self,
        doc_id: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Remove a document from a session."""
        session = self._get_session(session_id)
        if not session or doc_id not in session.documents:
            return False

        # Remove document
        del session.documents[doc_id]

        # Remove references involving this document
        session.references = [
            r
            for r in session.references
            if r.source_doc_id != doc_id and r.target_doc_id != doc_id
        ]

        # Remove from collections
        for collection in session.collections.values():
            if doc_id in collection.document_ids:
                collection.document_ids.remove(doc_id)

        self.save_session(session)
        return True

    def get_document(
        self,
        doc_id: str,
        session_id: Optional[str] = None,
    ) -> Optional[SessionDocument]:
        """Get a document by ID."""
        session = self._get_session(session_id)
        if not session:
            return None
        return session.documents.get(doc_id)

    def update_document(
        self,
        doc_id: str,
        session_id: Optional[str] = None,
        **updates: Any,
    ) -> Optional[SessionDocument]:
        """Update document properties."""
        session = self._get_session(session_id)
        if not session or doc_id not in session.documents:
            return None

        doc = session.documents[doc_id]

        for key, value in updates.items():
            if hasattr(doc, key) and value is not None:
                setattr(doc, key, value)

        self.save_session(session)
        return doc

    def list_documents(
        self,
        session_id: Optional[str] = None,
        collection_id: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[SessionDocument]:
        """List documents with optional filters."""
        session = self._get_session(session_id)
        if not session:
            return []

        docs = list(session.documents.values())

        # Filter by collection
        if collection_id and collection_id in session.collections:
            collection = session.collections[collection_id]
            docs = [d for d in docs if d.id in collection.document_ids]

        # Filter by status
        if status:
            docs = [d for d in docs if d.status == status]

        # Filter by tag
        if tag:
            docs = [d for d in docs if tag in d.tags]

        return docs

    # ============== Reference Management ==============

    def add_reference(
        self,
        source_doc_id: str,
        target_doc_id: str,
        reference_type: ReferenceType,
        confidence: float = 1.0,
        context: str = "",
        location: Optional[str] = None,
        auto_detected: bool = False,
        session_id: Optional[str] = None,
    ) -> Optional[DocumentReference]:
        """Add a cross-document reference."""
        session = self._get_session(session_id)
        if not session:
            return None

        # Validate document IDs
        if source_doc_id not in session.documents:
            return None
        if target_doc_id not in session.documents:
            return None

        ref = DocumentReference(
            source_doc_id=source_doc_id,
            target_doc_id=target_doc_id,
            reference_type=reference_type,
            confidence=confidence,
            context=context,
            location=location,
            created_at=self._now(),
            auto_detected=auto_detected,
        )

        session.references.append(ref)
        self.save_session(session)
        return ref

    def remove_reference(
        self,
        source_doc_id: str,
        target_doc_id: str,
        reference_type: Optional[ReferenceType] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Remove a reference between documents."""
        session = self._get_session(session_id)
        if not session:
            return False

        original_count = len(session.references)

        session.references = [
            r
            for r in session.references
            if not (
                r.source_doc_id == source_doc_id
                and r.target_doc_id == target_doc_id
                and (reference_type is None or r.reference_type == reference_type)
            )
        ]

        if len(session.references) < original_count:
            self.save_session(session)
            return True
        return False

    def get_document_references(
        self,
        doc_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
        reference_type: Optional[ReferenceType] = None,
        session_id: Optional[str] = None,
    ) -> List[DocumentReference]:
        """Get references for a document."""
        session = self._get_session(session_id)
        if not session:
            return []

        refs = []

        for ref in session.references:
            if reference_type and ref.reference_type != reference_type:
                continue

            if direction in ("both", "outgoing") and ref.source_doc_id == doc_id:
                refs.append(ref)
            elif direction in ("both", "incoming") and ref.target_doc_id == doc_id:
                refs.append(ref)

        return refs

    def get_related_documents(
        self,
        doc_id: str,
        session_id: Optional[str] = None,
    ) -> List[Tuple[SessionDocument, DocumentReference]]:
        """Get all documents related to a given document."""
        session = self._get_session(session_id)
        if not session:
            return []

        related = []
        seen_ids: Set[str] = set()

        for ref in session.references:
            related_id = None
            if ref.source_doc_id == doc_id:
                related_id = ref.target_doc_id
            elif ref.target_doc_id == doc_id:
                related_id = ref.source_doc_id

            if related_id and related_id not in seen_ids:
                if related_id in session.documents:
                    related.append((session.documents[related_id], ref))
                    seen_ids.add(related_id)

        return related

    def get_reference_graph(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the reference graph as nodes and edges."""
        session = self._get_session(session_id)
        if not session:
            return {"nodes": [], "edges": []}

        nodes = [
            {
                "id": doc.id,
                "name": doc.name,
                "title": doc.title,
                "type": doc.file_type,
                "status": doc.status,
            }
            for doc in session.documents.values()
        ]

        edges = [
            {
                "source": ref.source_doc_id,
                "target": ref.target_doc_id,
                "type": ref.reference_type.value,
                "confidence": ref.confidence,
            }
            for ref in session.references
        ]

        return {"nodes": nodes, "edges": edges}

    # ============== Collection Management ==============

    def create_collection(
        self,
        name: str,
        description: str = "",
        document_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> Optional[DocumentCollection]:
        """Create a document collection."""
        session = self._get_session(session_id)
        if not session:
            return None

        collection = DocumentCollection(
            id=self._generate_id(),
            name=name,
            description=description,
            document_ids=document_ids or [],
            created_at=self._now(),
        )

        session.collections[collection.id] = collection
        self.save_session(session)
        return collection

    def delete_collection(
        self,
        collection_id: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Delete a collection."""
        session = self._get_session(session_id)
        if not session or collection_id not in session.collections:
            return False

        del session.collections[collection_id]
        self.save_session(session)
        return True

    def add_to_collection(
        self,
        collection_id: str,
        doc_id: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Add a document to a collection."""
        session = self._get_session(session_id)
        if not session:
            return False
        if collection_id not in session.collections:
            return False
        if doc_id not in session.documents:
            return False

        collection = session.collections[collection_id]
        if doc_id not in collection.document_ids:
            collection.document_ids.append(doc_id)
            # Also update document's collections list
            session.documents[doc_id].collections.append(collection_id)
            self.save_session(session)

        return True

    def remove_from_collection(
        self,
        collection_id: str,
        doc_id: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Remove a document from a collection."""
        session = self._get_session(session_id)
        if not session or collection_id not in session.collections:
            return False

        collection = session.collections[collection_id]
        if doc_id in collection.document_ids:
            collection.document_ids.remove(doc_id)
            if collection_id in session.documents[doc_id].collections:
                session.documents[doc_id].collections.remove(collection_id)
            self.save_session(session)
            return True
        return False

    # ============== Shared Context ==============

    def update_shared_context(
        self,
        session_id: Optional[str] = None,
        **updates: Any,
    ) -> Optional[SharedContext]:
        """Update shared context."""
        session = self._get_session(session_id)
        if not session:
            return None

        ctx = session.shared_context
        for key, value in updates.items():
            if hasattr(ctx, key) and value is not None:
                setattr(ctx, key, value)

        self.save_session(session)
        return ctx

    def add_to_glossary(
        self,
        term: str,
        definition: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Add a term to the shared glossary."""
        session = self._get_session(session_id)
        if not session:
            return False

        session.shared_context.glossary[term] = definition
        self.save_session(session)
        return True

    def get_shared_context(
        self,
        session_id: Optional[str] = None,
    ) -> Optional[SharedContext]:
        """Get shared context."""
        session = self._get_session(session_id)
        return session.shared_context if session else None

    # ============== Search & Analysis ==============

    def search_documents(
        self,
        query: str,
        session_id: Optional[str] = None,
        search_in: Optional[List[str]] = None,  # fields to search in
    ) -> List[SessionDocument]:
        """Search documents by text query."""
        session = self._get_session(session_id)
        if not session:
            return []

        search_fields = search_in or ["name", "title", "abstract", "notes"]
        query_lower = query.lower()
        results = []

        for doc in session.documents.values():
            for field in search_fields:
                value = getattr(doc, field, None)
                if value and query_lower in str(value).lower():
                    results.append(doc)
                    break

            # Also search keywords and tags
            if doc.keywords and any(query_lower in kw.lower() for kw in doc.keywords):
                if doc not in results:
                    results.append(doc)
            if doc.tags and any(query_lower in tag.lower() for tag in doc.tags):
                if doc not in results:
                    results.append(doc)

        return results

    def get_session_stats(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics about a session."""
        session = self._get_session(session_id)
        if not session:
            return {}

        docs = list(session.documents.values())

        return {
            "document_count": len(docs),
            "reference_count": len(session.references),
            "collection_count": len(session.collections),
            "total_word_count": sum(d.word_count for d in docs),
            "total_page_count": sum(d.page_count for d in docs),
            "status_breakdown": {
                status: len([d for d in docs if d.status == status])
                for status in ["pending", "processing", "completed", "failed"]
            },
            "reference_type_breakdown": {
                rt.value: len([r for r in session.references if r.reference_type == rt])
                for rt in ReferenceType
            },
            "file_type_breakdown": {
                ft: len([d for d in docs if d.file_type == ft])
                for ft in set(d.file_type for d in docs)
            },
        }

    # ============== Helpers ==============

    def _get_session(
        self,
        session_id: Optional[str] = None,
    ) -> Optional[MultiDocumentSession]:
        """Get session by ID or return current."""
        if session_id:
            return self.load_session(session_id)
        return self._current_session


class _ManagerSingleton:
    """Singleton holder for multi-document session manager.

    Rule #6: Encapsulates singleton state in smallest scope.
    """

    _instance: Optional[MultiDocumentSessionManager] = None

    @classmethod
    def get(cls, sessions_dir: Optional[str] = None) -> MultiDocumentSessionManager:
        """Get singleton manager instance."""
        if cls._instance is None:
            path = Path(sessions_dir) if sessions_dir else None
            cls._instance = MultiDocumentSessionManager(path)
        return cls._instance


def get_multi_session_manager(
    sessions_dir: Optional[str] = None,
) -> MultiDocumentSessionManager:
    """Get singleton manager instance."""
    return _ManagerSingleton.get(sessions_dir)
