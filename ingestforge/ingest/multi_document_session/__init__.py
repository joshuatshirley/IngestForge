"""
Multi-document session management with cross-document references.

Extends basic session management to support:
- Multiple documents within a single session
- Cross-document references and relationships
- Document grouping and collections
- Shared context across documents
- Cross-document search and linking


# This module defines cross-document reference tracking awaiting pipeline integration.
# Integration will enable linking related documents (e.g., chapters, versions).
# Estimated integration: P3-COLLAB-001 (Multi-User Projects)
"""

# Models
from ingestforge.ingest.multi_document_session.models import (
    DocumentCollection,
    DocumentReference,
    MultiDocumentSession,
    ReferenceType,
    SessionDocument,
    SharedContext,
)

# Manager
from ingestforge.ingest.multi_document_session.manager import (
    MultiDocumentSessionManager,
    get_multi_session_manager,
)

__all__ = [
    # Enums
    "ReferenceType",
    # Models
    "DocumentReference",
    "SessionDocument",
    "DocumentCollection",
    "SharedContext",
    "MultiDocumentSession",
    # Manager
    "MultiDocumentSessionManager",
    "get_multi_session_manager",
]
