"""
Source Provenance and Citation Tracking for IngestForge.

This module enables academic-quality citations by tracking the precise origin
of every piece of content through the processing pipeline.

Public API
----------
All provenance classes are re-exported here for backward compatibility:

    from ingestforge.core.provenance import SourceLocation, SourceType, Author
    from ingestforge.core.provenance import ContributorIdentity  # TICKET-301

Architecture
------------
Provenance is organized into focused modules:

    provenance/
    ├── models.py      # SourceType, CitationStyle enums
    ├── author.py      # Author dataclass with name parsing, ContributorIdentity
    └── location.py    # SourceLocation class with citation formatting

Usage Example
-------------
    from ingestforge.core.provenance import SourceLocation, SourceType, Author
    from ingestforge.core.provenance import ContributorIdentity, set_author, get_author_info
    from ingestforge.chunking.semantic_chunker import ChunkRecord

    # Create source location
    loc = SourceLocation(
        source_type=SourceType.PDF,
        title="Research Methods",
        authors=[Author("Jane Smith")],
        page_start=47,
    )

    # Generate citation
    citation = loc.to_short_cite()  # [Smith, p.47]

    # Track contributor (TICKET-301)
    contributor = ContributorIdentity(
        author_id="john.doe@example.com",
        author_name="John Doe"
    )
    print(contributor.format_attribution())  # Contributed by: John Doe

    # Set author on chunk using helper function
    chunk = ChunkRecord(chunk_id="c1", document_id="d1", content="text")
    set_author(chunk, "john.doe@example.com", "John Doe")

    # Get author info from chunk
    author_info = get_author_info(chunk)
    print(author_info.format_attribution())  # Contributed by: John Doe
"""

# Core models
from ingestforge.core.provenance.models import CitationStyle, SourceType

# Author and ContributorIdentity
from ingestforge.core.provenance.author import (
    Author,
    ContributorIdentity,
    set_author,
    get_author_info,
)

# SourceLocation
from ingestforge.core.provenance.location import SourceLocation

__all__ = [
    # Enums
    "SourceType",
    "CitationStyle",
    # Classes
    "Author",
    "ContributorIdentity",
    "SourceLocation",
    # Helper Functions (TICKET-301)
    "set_author",
    "get_author_info",
]
