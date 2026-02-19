"""Tests for ContributorIdentity class (TICKET-301).

Verifies:
1. ContributorIdentity creation with author_id and author_name
2. Validation and sanitization of empty strings
3. is_populated() method
4. format_attribution() method
5. Serialization via to_dict() and from_dict()
"""


from ingestforge.core.provenance import ContributorIdentity


class TestContributorIdentityCreation:
    """Test ContributorIdentity construction."""

    def test_create_with_both_fields(self) -> None:
        """Create with both author_id and author_name."""
        contributor = ContributorIdentity(
            author_id="john.doe@example.com", author_name="John Doe"
        )
        assert contributor.author_id == "john.doe@example.com"
        assert contributor.author_name == "John Doe"

    def test_create_with_only_id(self) -> None:
        """Create with only author_id."""
        contributor = ContributorIdentity(author_id="user123")
        assert contributor.author_id == "user123"
        assert contributor.author_name is None

    def test_create_with_only_name(self) -> None:
        """Create with only author_name."""
        contributor = ContributorIdentity(author_name="Jane Smith")
        assert contributor.author_id is None
        assert contributor.author_name == "Jane Smith"

    def test_create_empty(self) -> None:
        """Create with no fields."""
        contributor = ContributorIdentity()
        assert contributor.author_id is None
        assert contributor.author_name is None


class TestContributorIdentityValidation:
    """Test validation and sanitization."""

    def test_empty_string_sanitized_to_none(self) -> None:
        """Empty strings should be converted to None."""
        contributor = ContributorIdentity(author_id="", author_name="")
        assert contributor.author_id is None
        assert contributor.author_name is None

    def test_whitespace_only_sanitized_to_none(self) -> None:
        """Whitespace-only strings should be converted to None."""
        contributor = ContributorIdentity(author_id="   ", author_name="  \t  ")
        assert contributor.author_id is None
        assert contributor.author_name is None

    def test_valid_whitespace_trimmed_kept(self) -> None:
        """Valid values with surrounding whitespace are kept (not trimmed)."""
        contributor = ContributorIdentity(author_id=" user123 ", author_name=" John ")
        # Note: The current implementation does not trim whitespace,
        # only converts whitespace-only to None
        assert contributor.author_id == " user123 "
        assert contributor.author_name == " John "


class TestContributorIdentityIsPopulated:
    """Test is_populated() method."""

    def test_populated_with_both(self) -> None:
        """Populated when both fields set."""
        contributor = ContributorIdentity(author_id="user123", author_name="John Doe")
        assert contributor.is_populated() is True

    def test_populated_with_id_only(self) -> None:
        """Populated when only author_id set."""
        contributor = ContributorIdentity(author_id="user123")
        assert contributor.is_populated() is True

    def test_populated_with_name_only(self) -> None:
        """Populated when only author_name set."""
        contributor = ContributorIdentity(author_name="John Doe")
        assert contributor.is_populated() is True

    def test_not_populated_when_empty(self) -> None:
        """Not populated when both fields are None."""
        contributor = ContributorIdentity()
        assert contributor.is_populated() is False


class TestContributorIdentityFormatAttribution:
    """Test format_attribution() method."""

    def test_format_with_name(self) -> None:
        """Format attribution uses name when available."""
        contributor = ContributorIdentity(
            author_id="john.doe@example.com", author_name="John Doe"
        )
        assert contributor.format_attribution() == "Contributed by: John Doe"

    def test_format_with_id_only(self) -> None:
        """Format attribution falls back to author_id."""
        contributor = ContributorIdentity(author_id="john.doe@example.com")
        assert (
            contributor.format_attribution() == "Contributed by: john.doe@example.com"
        )

    def test_format_when_empty(self) -> None:
        """Format attribution returns empty string when no data."""
        contributor = ContributorIdentity()
        assert contributor.format_attribution() == ""


class TestContributorIdentitySerialization:
    """Test serialization methods."""

    def test_to_dict(self) -> None:
        """Convert to dictionary."""
        contributor = ContributorIdentity(author_id="user123", author_name="John Doe")
        data = contributor.to_dict()
        assert data == {"author_id": "user123", "author_name": "John Doe"}

    def test_to_dict_with_none(self) -> None:
        """Convert to dictionary preserves None values."""
        contributor = ContributorIdentity()
        data = contributor.to_dict()
        assert data == {"author_id": None, "author_name": None}

    def test_from_dict(self) -> None:
        """Create from dictionary."""
        data = {"author_id": "user123", "author_name": "John Doe"}
        contributor = ContributorIdentity.from_dict(data)
        assert contributor.author_id == "user123"
        assert contributor.author_name == "John Doe"

    def test_from_dict_partial(self) -> None:
        """Create from dictionary with missing keys."""
        data = {"author_name": "John Doe"}
        contributor = ContributorIdentity.from_dict(data)
        assert contributor.author_id is None
        assert contributor.author_name == "John Doe"

    def test_from_dict_empty(self) -> None:
        """Create from empty dictionary."""
        contributor = ContributorIdentity.from_dict({})
        assert contributor.author_id is None
        assert contributor.author_name is None

    def test_roundtrip(self) -> None:
        """Roundtrip through serialization."""
        original = ContributorIdentity(author_id="user123", author_name="John Doe")
        restored = ContributorIdentity.from_dict(original.to_dict())
        assert restored.author_id == original.author_id
        assert restored.author_name == original.author_name
