"""
Tests for Chunk Deduplicator.

This module tests duplicate chunk detection using similarity hashing (MinHash-like).

Test Strategy
-------------
- Focus on deduplication logic and similarity detection
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test dataclass, similarity functions, and deduplication workflow
- Use simple ChunkRecord mocks (no external dependencies)

Organization
------------
- TestDeduplicationReport: DeduplicationReport dataclass
- TestDeduplicatorInit: Initialization and parameters
- TestTextNormalization: _normalize_text function
- TestContentHashing: _compute_content_hash function
- TestShingles: _compute_shingles function
- TestJaccardSimilarity: _compute_jaccard_similarity function
- TestDeduplication: Main deduplicate() method
"""


from ingestforge.chunking.deduplicator import Deduplicator, DeduplicationReport
from ingestforge.chunking.semantic_chunker import ChunkRecord


# ============================================================================
# Test Helpers
# ============================================================================


def make_chunk(
    chunk_id: str,
    content: str,
    word_count: int = 10,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> ChunkRecord:
    """Create a test ChunkRecord."""
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="test_doc",
        content=content,
        word_count=word_count,
        char_count=len(content),
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        source_file="test.txt",
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestDeduplicationReport:
    """Tests for DeduplicationReport dataclass.

    Rule #4: Focused test class - tests only DeduplicationReport
    """

    def test_create_deduplication_report(self):
        """Test creating a DeduplicationReport."""
        report = DeduplicationReport(
            original_count=100,
            final_count=75,
            duplicates_removed=25,
            duplicate_groups=10,
        )

        assert report.original_count == 100
        assert report.final_count == 75
        assert report.duplicates_removed == 25
        assert report.duplicate_groups == 10


class TestDeduplicatorInit:
    """Tests for Deduplicator initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_deduplicator_with_defaults(self):
        """Test creating Deduplicator with default parameters."""
        dedup = Deduplicator()

        assert dedup.similarity_threshold == 0.85
        assert dedup.min_shingle_size == 3

    def test_create_deduplicator_with_custom_params(self):
        """Test creating Deduplicator with custom parameters."""
        dedup = Deduplicator(
            similarity_threshold=0.90,
            min_shingle_size=5,
        )

        assert dedup.similarity_threshold == 0.90
        assert dedup.min_shingle_size == 5


class TestTextNormalization:
    """Tests for text normalization.

    Rule #4: Focused test class - tests _normalize_text only
    """

    def test_normalize_text_lowercase(self):
        """Test normalization converts to lowercase."""
        dedup = Deduplicator()
        text = "Hello World"

        result = dedup._normalize_text(text)

        assert result == "hello world"
        assert result.islower()

    def test_normalize_text_removes_punctuation(self):
        """Test normalization removes punctuation."""
        dedup = Deduplicator()
        text = "Hello, world! How are you?"

        result = dedup._normalize_text(text)

        assert "," not in result
        assert "!" not in result
        assert "?" not in result
        assert result == "hello world how are you"

    def test_normalize_text_collapses_whitespace(self):
        """Test normalization collapses multiple spaces."""
        dedup = Deduplicator()
        text = "Multiple    spaces    here"

        result = dedup._normalize_text(text)

        assert result == "multiple spaces here"

    def test_normalize_text_strips_edges(self):
        """Test normalization strips leading/trailing whitespace."""
        dedup = Deduplicator()
        text = "  content here  "

        result = dedup._normalize_text(text)

        assert result == "content here"
        assert not result.startswith(" ")
        assert not result.endswith(" ")


class TestContentHashing:
    """Tests for content hashing.

    Rule #4: Focused test class - tests _compute_content_hash only
    """

    def test_compute_content_hash_deterministic(self):
        """Test content hash is deterministic."""
        dedup = Deduplicator()
        content = "Hello, world!"

        hash1 = dedup._compute_content_hash(content)
        hash2 = dedup._compute_content_hash(content)

        assert hash1 == hash2

    def test_compute_content_hash_same_normalized_content(self):
        """Test same normalized content produces same hash."""
        dedup = Deduplicator()
        content1 = "Hello, world!"
        content2 = "HELLO, WORLD!"  # Different case, same normalized

        hash1 = dedup._compute_content_hash(content1)
        hash2 = dedup._compute_content_hash(content2)

        # Same after normalization (lowercase, no punctuation)
        assert hash1 == hash2

    def test_compute_content_hash_different_content(self):
        """Test different content produces different hash."""
        dedup = Deduplicator()
        content1 = "Hello, world!"
        content2 = "Goodbye, world!"

        hash1 = dedup._compute_content_hash(content1)
        hash2 = dedup._compute_content_hash(content2)

        assert hash1 != hash2


class TestShingles:
    """Tests for shingle computation.

    Rule #4: Focused test class - tests _compute_shingles only
    """

    def test_compute_shingles_basic(self):
        """Test basic shingle computation."""
        dedup = Deduplicator(min_shingle_size=3)
        content = "This is a test sentence"

        shingles = dedup._compute_shingles(content)

        # Should have shingles like "this is a", "is a test", "a test sentence"
        assert isinstance(shingles, set)
        assert len(shingles) == 3  # 5 words → 3 shingles (5 - 3 + 1)

    def test_compute_shingles_short_text(self):
        """Test shingle computation with text shorter than min size."""
        dedup = Deduplicator(min_shingle_size=5)
        content = "Short text"

        shingles = dedup._compute_shingles(content)

        # Less than min_shingle_size words → return normalized text
        assert len(shingles) == 1
        assert "short text" in shingles

    def test_compute_shingles_normalized(self):
        """Test shingles are normalized (lowercase, no punctuation)."""
        dedup = Deduplicator(min_shingle_size=2)
        content = "Hello, World! This is Great."

        shingles = dedup._compute_shingles(content)

        # All shingles should be lowercase and no punctuation
        for shingle in shingles:
            assert shingle.islower()
            assert "," not in shingle
            assert "!" not in shingle
            assert "." not in shingle


class TestJaccardSimilarity:
    """Tests for Jaccard similarity computation.

    Rule #4: Focused test class - tests _compute_jaccard_similarity only
    """

    def test_compute_jaccard_similarity_identical_sets(self):
        """Test Jaccard similarity of identical sets is 1.0."""
        dedup = Deduplicator()
        set1 = {"a", "b", "c"}
        set2 = {"a", "b", "c"}

        similarity = dedup._compute_jaccard_similarity(set1, set2)

        assert similarity == 1.0

    def test_compute_jaccard_similarity_no_overlap(self):
        """Test Jaccard similarity of disjoint sets is 0.0."""
        dedup = Deduplicator()
        set1 = {"a", "b", "c"}
        set2 = {"d", "e", "f"}

        similarity = dedup._compute_jaccard_similarity(set1, set2)

        assert similarity == 0.0

    def test_compute_jaccard_similarity_partial_overlap(self):
        """Test Jaccard similarity of partially overlapping sets."""
        dedup = Deduplicator()
        set1 = {"a", "b", "c", "d"}
        set2 = {"c", "d", "e", "f"}

        similarity = dedup._compute_jaccard_similarity(set1, set2)

        # Intersection: {c, d} = 2, Union: {a,b,c,d,e,f} = 6
        # Jaccard = 2/6 = 0.333...
        assert abs(similarity - 0.333) < 0.01

    def test_compute_jaccard_similarity_empty_sets(self):
        """Test Jaccard similarity with empty sets."""
        dedup = Deduplicator()
        set1 = set()
        set2 = {"a", "b"}

        similarity = dedup._compute_jaccard_similarity(set1, set2)

        assert similarity == 0.0


class TestDeduplication:
    """Tests for main deduplication workflow.

    Rule #4: Focused test class - tests deduplicate() method
    """

    def test_deduplicate_empty_list(self):
        """Test deduplicating empty chunk list."""
        dedup = Deduplicator()
        chunks = []

        result, report = dedup.deduplicate(chunks)

        assert result == []
        assert report.original_count == 0
        assert report.final_count == 0
        assert report.duplicates_removed == 0

    def test_deduplicate_no_duplicates(self):
        """Test deduplicating chunks with no duplicates."""
        dedup = Deduplicator()
        chunks = [
            make_chunk("1", "This is unique content alpha"),
            make_chunk("2", "This is unique content beta"),
            make_chunk("3", "This is unique content gamma"),
        ]

        result, report = dedup.deduplicate(chunks)

        assert len(result) == 3
        assert report.original_count == 3
        assert report.final_count == 3
        assert report.duplicates_removed == 0

    def test_deduplicate_exact_duplicates(self):
        """Test deduplicating exact duplicate chunks."""
        dedup = Deduplicator()
        chunks = [
            make_chunk("1", "Exact same content", word_count=3),
            make_chunk("2", "Exact same content", word_count=3),
            make_chunk("3", "Exact same content", word_count=5),  # Higher word count
        ]

        result, report = dedup.deduplicate(chunks)

        # Should keep chunk with highest word_count
        assert len(result) == 1
        assert result[0].chunk_id == "3"
        assert report.original_count == 3
        assert report.final_count == 1
        assert report.duplicates_removed == 2

    def test_deduplicate_near_duplicates(self):
        """Test deduplicating near-duplicate chunks."""
        dedup = Deduplicator(similarity_threshold=0.70)
        chunks = [
            make_chunk(
                "1", "The quick brown fox jumps over the lazy dog", word_count=9
            ),
            make_chunk(
                "2", "The quick brown fox jumps over the lazy cat", word_count=9
            ),
            make_chunk("3", "Completely different content here", word_count=4),
        ]

        result, report = dedup.deduplicate(chunks)

        # Chunks 1 and 2 are very similar (differ by 1 word)
        # Should keep one of them plus chunk 3
        assert len(result) == 2
        assert report.duplicates_removed >= 1
        # Ensure chunk 3 (unique) is kept
        chunk_ids = {c.chunk_id for c in result}
        assert "3" in chunk_ids

    def test_deduplicate_reindexes_chunks(self):
        """Test deduplication reindexes remaining chunks."""
        dedup = Deduplicator()
        chunks = [
            make_chunk("1", "Content A", chunk_index=0),
            make_chunk("2", "Content A", chunk_index=1),  # Duplicate
            make_chunk("3", "Content B", chunk_index=2),
        ]

        result, report = dedup.deduplicate(chunks)

        # After removing duplicate, chunks should be reindexed
        assert len(result) == 2
        assert result[0].chunk_index == 0
        assert result[1].chunk_index == 1
        assert result[0].total_chunks == 2
        assert result[1].total_chunks == 2

    def test_deduplicate_keeps_highest_word_count(self):
        """Test deduplication keeps chunk with highest word count."""
        dedup = Deduplicator()
        chunks = [
            make_chunk("1", "Same content", word_count=5),
            make_chunk("2", "Same content", word_count=10),  # Highest
            make_chunk("3", "Same content", word_count=3),
        ]

        result, report = dedup.deduplicate(chunks)

        assert len(result) == 1
        assert result[0].chunk_id == "2"
        assert result[0].word_count == 10


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - DeduplicationReport: 1 test (dataclass creation)
    - Deduplicator init: 2 tests (defaults, custom params)
    - Text normalization: 4 tests (lowercase, punctuation, whitespace, strips)
    - Content hashing: 3 tests (deterministic, same normalized, different content)
    - Shingles: 3 tests (basic, short text, normalized)
    - Jaccard similarity: 4 tests (identical, no overlap, partial, empty)
    - Deduplication: 7 tests (empty, no dups, exact dups, near dups, reindex, word count)

    Total: 24 tests

Design Decisions:
    1. Focus on deduplication logic and similarity detection
    2. Use simple ChunkRecord mocks (no external dependencies)
    3. Test normalization, hashing, and similarity functions separately
    4. Test main deduplication workflow with various scenarios
    5. Simple, clear tests that verify deduplication works
    6. Follows NASA JPL Rule #1 (Simple Control Flow)
    7. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - DeduplicationReport dataclass creation
    - Deduplicator initialization with default and custom parameters
    - Text normalization (lowercase, punctuation removal, whitespace collapse)
    - Content hashing (deterministic, case-insensitive)
    - Shingle computation (n-grams, short text handling)
    - Jaccard similarity (identical, disjoint, partial overlap)
    - Exact duplicate detection and removal
    - Near-duplicate detection with similarity threshold
    - Chunk reindexing after deduplication
    - Selection of best chunk (highest word count) from duplicates

Justification:
    - Deduplication is critical for reducing redundant content
    - Similarity detection needs verification with known inputs
    - Hash and shingle functions are deterministic and testable
    - Main workflow tests cover common deduplication scenarios
    - Simple tests verify deduplication system works correctly
"""
