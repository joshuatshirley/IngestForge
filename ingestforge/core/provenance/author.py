"""
Author data model for provenance tracking.

Provides the Author dataclass with automatic name parsing and
contributor identity metadata for attribution (TICKET-301)."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ingestforge.chunking.semantic_chunker import ChunkRecord


@dataclass
class Author:
    """Author information for source document attribution."""

    name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    affiliation: Optional[str] = None

    def __post_init__(self) -> None:
        """Parse name into first/last if not provided."""
        if not self.last_name and self.name:
            parts = self.name.strip().split()
            if len(parts) >= 2:
                self.first_name = parts[0]
                self.last_name = " ".join(parts[1:])
            else:
                self.last_name = self.name

    def format_apa(self) -> str:
        """Format as 'Last, F.' for APA style."""
        if self.last_name and self.first_name:
            return f"{self.last_name}, {self.first_name[0]}."
        return self.name

    def format_full(self) -> str:
        """Format as 'First Last'."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.name


@dataclass
class ContributorIdentity:
    """
    Contributor identity metadata for chunk attribution (TICKET-301).

    Tracks who contributed content to the knowledge base, separate from
    the original document authors. Used for displaying "Contributed by"
    attribution in query results and preserving authorship in exports.

    Attributes:
        author_id: Unique identifier for the contributor (e.g., email, username)
        author_name: Human-readable display name for the contributor

    Examples:
        # Create contributor identity
        contributor = ContributorIdentity(
            author_id="john.doe@example.com",
            author_name="John Doe"
        )

        # Display attribution
        print(f"Contributed by: {contributor.author_name}")

    Rule #7: Both fields are optional to support anonymous ingestion
    Rule #9: Complete type hints
    """

    author_id: Optional[str] = None
    author_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate contributor identity fields.

        Rule #7: Parameter validation at function entry
        """
        # Sanitize empty strings to None
        if self.author_id is not None and not self.author_id.strip():
            self.author_id = None
        if self.author_name is not None and not self.author_name.strip():
            self.author_name = None

    def is_populated(self) -> bool:
        """Check if contributor identity has any populated fields.

        Returns:
            True if at least author_id or author_name is set
        """
        return self.author_id is not None or self.author_name is not None

    def format_attribution(self) -> str:
        """Format contributor for display in query results.

        Returns:
            Attribution string, or empty string if no contributor info

        Examples:
            "Contributed by: John Doe"
            "Contributed by: john.doe@example.com"
            ""  (if no contributor info)
        """
        if self.author_name:
            return f"Contributed by: {self.author_name}"
        if self.author_id:
            return f"Contributed by: {self.author_id}"
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with author_id and author_name
        """
        return {
            "author_id": self.author_id,
            "author_name": self.author_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContributorIdentity":
        """Create ContributorIdentity from dictionary.

        Args:
            data: Dictionary with author_id and/or author_name keys

        Returns:
            ContributorIdentity instance

        Rule #7: Handles missing keys gracefully
        """
        return cls(
            author_id=data.get("author_id"),
            author_name=data.get("author_name"),
        )


# =============================================================================
# Helper Functions for Chunk Author Metadata (TICKET-301)
# =============================================================================


def set_author(
    chunk: "ChunkRecord", author_id: Optional[str], author_name: Optional[str]
) -> "ChunkRecord":
    """Set author metadata on a chunk.

    Helper function to attach contributor identity to a chunk for multi-user
    collaboration support. Modifies the chunk in-place and returns it for
    chaining.

    Args:
        chunk: ChunkRecord to modify
        author_id: Unique identifier for the contributor (e.g., email, username)
        author_name: Human-readable display name for the contributor

    Returns:
        The modified chunk (for method chaining)

    Raises:
        ValueError: If chunk is None

    Examples:
        # Set both author_id and author_name
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content="text")
        set_author(chunk, "john.doe@example.com", "John Doe")

        # Set only author_name (anonymous ingestion with display name)
        set_author(chunk, None, "Anonymous Contributor")

        # Method chaining
        chunk = set_author(chunk, "user123", "Jane Smith")

    Rule #7: Validates parameters at function entry
    Rule #9: Complete type hints
    """
    if chunk is None:
        raise ValueError("chunk cannot be None")

    # Set author fields
    chunk.author_id = author_id
    chunk.author_name = author_name

    return chunk


def get_author_info(chunk: "ChunkRecord") -> ContributorIdentity:
    """Get author information from a chunk as ContributorIdentity.

    Extracts author metadata from a chunk and returns it as a
    ContributorIdentity object for display and serialization.

    Args:
        chunk: ChunkRecord to extract author info from

    Returns:
        ContributorIdentity with author_id and author_name from chunk

    Raises:
        ValueError: If chunk is None

    Examples:
        # Get author info
        chunk = ChunkRecord(
            chunk_id="c1",
            document_id="d1",
            content="text",
            author_id="user123",
            author_name="Jane Smith"
        )
        author_info = get_author_info(chunk)
        print(author_info.format_attribution())  # "Contributed by: Jane Smith"

        # Check if author info is present
        if author_info.is_populated():
            print(f"Author: {author_info.author_name}")

    Rule #7: Validates parameters at function entry
    Rule #9: Complete type hints
    """
    if chunk is None:
        raise ValueError("chunk cannot be None")

    return ContributorIdentity(
        author_id=chunk.author_id,
        author_name=chunk.author_name,
    )
