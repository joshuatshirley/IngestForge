"""
Tests for Parent Document Retriever.

This module tests parent chunk expansion for context-aware retrieval.

Test Strategy
-------------
- Focus on parent-child mapping and context expansion
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Mock storage and mapping store when needed
- Test context window truncation (tokens and chars)

Organization
------------
- TestParentExpandedResult: ParentExpandedResult dataclass
- TestParentRetrieverInit: Initialization
- TestTruncation: Context window truncation methods
- TestExpansion: Main expand_results method
- TestSearchWithExpansion: Integrated search workflow
"""

from unittest.mock import Mock, patch
from pathlib import Path


from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.retrieval.parent_retriever import (
    ParentRetriever,
    ParentExpandedResult,
    create_parent_retriever,
)
from ingestforge.storage.base import SearchResult


# ============================================================================
# Test Helpers
# ============================================================================


def make_chunk(chunk_id: str, content: str) -> ChunkRecord:
    """Create a test ChunkRecord."""
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="test_doc",
        content=content,
        word_count=len(content.split()),
        char_count=len(content),
        source_file="test.txt",
    )


def make_search_result(chunk_id: str, score: float = 0.85) -> SearchResult:
    """Create a test SearchResult."""
    return SearchResult(
        chunk_id=chunk_id,
        content=f"Content for {chunk_id}",
        score=score,
        document_id="test_doc",
        section_title="Test Section",
        chunk_type="text",
        source_file="test.txt",
        word_count=10,
    )


def make_parent_mapping(
    child_id: str,
    parent_id: str,
    position: int = 0,
    total: int = 1,
):
    """Create a mock parent mapping."""
    mapping = Mock()
    mapping.child_chunk_id = child_id
    mapping.parent_chunk_id = parent_id
    mapping.child_position = position
    mapping.total_children = total
    return mapping


# ============================================================================
# Test Classes
# ============================================================================


class TestParentExpandedResult:
    """Tests for ParentExpandedResult dataclass.

    Rule #4: Focused test class - tests only ParentExpandedResult
    """

    def test_create_expanded_result_with_parent(self):
        """Test creating expanded result with parent chunk."""
        child_result = make_search_result("child_1", 0.9)
        parent_chunk = make_chunk("parent_1", "Full parent context")

        expanded = ParentExpandedResult(
            child_result=child_result,
            parent_chunk=parent_chunk,
            child_position=2,
            total_siblings=5,
        )

        assert expanded.child_result.chunk_id == "child_1"
        assert expanded.parent_chunk.chunk_id == "parent_1"
        assert expanded.child_position == 2
        assert expanded.total_siblings == 5
        assert expanded.truncated_content is None

    def test_create_expanded_result_without_parent(self):
        """Test creating expanded result without parent (no mapping)."""
        child_result = make_search_result("child_1", 0.9)

        expanded = ParentExpandedResult(
            child_result=child_result,
            parent_chunk=None,
            child_position=0,
            total_siblings=1,
        )

        assert expanded.child_result.chunk_id == "child_1"
        assert expanded.parent_chunk is None
        assert expanded.child_position == 0
        assert expanded.total_siblings == 1

    def test_create_expanded_result_with_truncation(self):
        """Test expanded result with truncated content."""
        child_result = make_search_result("child_1", 0.9)
        parent_chunk = make_chunk("parent_1", "Very long parent context..." * 100)

        expanded = ParentExpandedResult(
            child_result=child_result,
            parent_chunk=parent_chunk,
            child_position=0,
            total_siblings=1,
            truncated_content="Truncated version",
        )

        assert expanded.truncated_content == "Truncated version"


class TestParentRetrieverInit:
    """Tests for ParentRetriever initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_retriever_with_defaults(self):
        """Test creating ParentRetriever with default settings."""
        storage = Mock()
        mapping_store = Mock()

        retriever = ParentRetriever(storage, mapping_store)

        assert retriever.storage is storage
        assert retriever.mapping_store is mapping_store
        assert retriever.deduplicate_parents is True
        assert retriever.context_window_tokens == 2048
        assert retriever.context_window_chars == 8000

    def test_create_retriever_with_custom_settings(self):
        """Test creating ParentRetriever with custom settings."""
        storage = Mock()
        mapping_store = Mock()

        retriever = ParentRetriever(
            storage,
            mapping_store,
            deduplicate_parents=False,
            context_window_tokens=1024,
            context_window_chars=4000,
        )

        assert retriever.deduplicate_parents is False
        assert retriever.context_window_tokens == 1024
        assert retriever.context_window_chars == 4000


class TestTruncation:
    """Tests for context window truncation.

    Rule #4: Focused test class - tests truncation methods
    """

    def test_truncate_short_content_no_change(self):
        """Test truncating content shorter than window."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(
            storage,
            mapping_store,
            context_window_chars=1000,
        )

        content = "Short content"
        truncated = retriever._truncate_to_window(content)

        assert truncated == content

    def test_truncate_long_content_chars(self):
        """Test truncating content longer than window (char-based)."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(
            storage,
            mapping_store,
            context_window_chars=50,
        )

        # Mock tokenizer to return None (force char-based truncation)
        with patch.object(retriever, "_get_tokenizer", return_value=None):
            content = "This is a very long content that exceeds the context window and should be truncated"
            truncated = retriever._truncate_to_window(content)

            # Should be truncated to ~50 chars (at word boundary)
            assert len(truncated) <= 50
            assert len(truncated) < len(content)
            # Should break at word boundary
            assert not truncated.endswith(" ")

    def test_apply_truncation_disabled(self):
        """Test apply_truncation with apply_window=False."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(storage, mapping_store)

        parent_chunk = make_chunk("parent_1", "Long content" * 100)
        result = retriever._apply_truncation(parent_chunk, apply_window=False)

        assert result is None

    def test_apply_truncation_enabled_no_change(self):
        """Test apply_truncation when content fits window."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(
            storage,
            mapping_store,
            context_window_chars=1000,
        )

        parent_chunk = make_chunk("parent_1", "Short content")
        result = retriever._apply_truncation(parent_chunk, apply_window=True)

        assert result is None

    def test_apply_truncation_enabled_truncates(self):
        """Test apply_truncation when content exceeds window."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(
            storage,
            mapping_store,
            context_window_chars=50,
        )

        # Mock tokenizer to return None (force char-based truncation)
        with patch.object(retriever, "_get_tokenizer", return_value=None):
            long_content = "Long content " * 20
            parent_chunk = make_chunk("parent_1", long_content)
            result = retriever._apply_truncation(parent_chunk, apply_window=True)

            assert result is not None
            assert len(result) < len(long_content)


class TestExpansion:
    """Tests for main expand_results method.

    Rule #4: Focused test class - tests expand_results
    """

    def test_expand_empty_results(self):
        """Test expanding empty results list."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(storage, mapping_store)

        expanded = retriever.expand_results([])

        assert expanded == []

    def test_expand_result_with_parent(self):
        """Test expanding result with parent mapping."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(storage, mapping_store)

        # Setup mocks
        child_result = make_search_result("child_1", 0.9)
        parent_chunk = make_chunk("parent_1", "Full parent context")
        mapping = make_parent_mapping("child_1", "parent_1", position=2, total=5)

        mapping_store.get_mapping.return_value = mapping
        storage.get_chunk.return_value = parent_chunk

        # Expand
        expanded = retriever.expand_results([child_result], apply_window=False)

        assert len(expanded) == 1
        assert expanded[0].child_result.chunk_id == "child_1"
        assert expanded[0].parent_chunk.chunk_id == "parent_1"
        assert expanded[0].child_position == 2
        assert expanded[0].total_siblings == 5
        assert expanded[0].truncated_content is None

    def test_expand_result_without_parent(self):
        """Test expanding result with no parent mapping."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(storage, mapping_store)

        # Setup mocks
        child_result = make_search_result("child_1", 0.9)
        mapping_store.get_mapping.return_value = None

        # Expand
        expanded = retriever.expand_results([child_result])

        assert len(expanded) == 1
        assert expanded[0].child_result.chunk_id == "child_1"
        assert expanded[0].parent_chunk is None
        assert expanded[0].child_position == 0
        assert expanded[0].total_siblings == 1

    def test_expand_deduplicates_parents(self):
        """Test deduplication of parent chunks."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(
            storage,
            mapping_store,
            deduplicate_parents=True,
        )

        # Setup mocks - two children with same parent
        child_1 = make_search_result("child_1", 0.9)
        child_2 = make_search_result("child_2", 0.8)
        parent_chunk = make_chunk("parent_1", "Shared parent")

        mapping_1 = make_parent_mapping("child_1", "parent_1", position=0, total=2)
        mapping_2 = make_parent_mapping("child_2", "parent_1", position=1, total=2)

        mapping_store.get_mapping.side_effect = [mapping_1, mapping_2]
        storage.get_chunk.return_value = parent_chunk

        # Expand
        expanded = retriever.expand_results([child_1, child_2])

        # Only first child should have parent (deduplication)
        assert len(expanded) == 1
        assert expanded[0].child_result.chunk_id == "child_1"
        assert expanded[0].parent_chunk.chunk_id == "parent_1"

    def test_expand_no_deduplication(self):
        """Test expansion without deduplication."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(
            storage,
            mapping_store,
            deduplicate_parents=False,
        )

        # Setup mocks - two children with same parent
        child_1 = make_search_result("child_1", 0.9)
        child_2 = make_search_result("child_2", 0.8)
        parent_chunk = make_chunk("parent_1", "Shared parent")

        mapping_1 = make_parent_mapping("child_1", "parent_1", position=0, total=2)
        mapping_2 = make_parent_mapping("child_2", "parent_1", position=1, total=2)

        mapping_store.get_mapping.side_effect = [mapping_1, mapping_2]
        storage.get_chunk.return_value = parent_chunk

        # Expand
        expanded = retriever.expand_results([child_1, child_2])

        # Both children should have parent (no deduplication)
        assert len(expanded) == 2
        assert expanded[0].child_result.chunk_id == "child_1"
        assert expanded[0].parent_chunk.chunk_id == "parent_1"
        assert expanded[1].child_result.chunk_id == "child_2"
        assert expanded[1].parent_chunk.chunk_id == "parent_1"

    def test_expand_with_truncation(self):
        """Test expansion with context window truncation."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(
            storage,
            mapping_store,
            context_window_chars=50,
        )

        # Setup mocks
        child_result = make_search_result("child_1", 0.9)
        long_content = "Very long parent content " * 20
        parent_chunk = make_chunk("parent_1", long_content)
        mapping = make_parent_mapping("child_1", "parent_1")

        mapping_store.get_mapping.return_value = mapping
        storage.get_chunk.return_value = parent_chunk

        # Mock tokenizer to return None (force char-based truncation)
        with patch.object(retriever, "_get_tokenizer", return_value=None):
            # Expand with truncation
            expanded = retriever.expand_results([child_result], apply_window=True)

            assert len(expanded) == 1
            assert expanded[0].parent_chunk.chunk_id == "parent_1"
            # Should have truncated content
            assert expanded[0].truncated_content is not None
            assert len(expanded[0].truncated_content) < len(long_content)


class TestSearchWithExpansion:
    """Tests for integrated search workflow.

    Rule #4: Focused test class - tests search_with_expansion
    """

    def test_search_with_expansion_basic(self):
        """Test search_with_expansion workflow."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(storage, mapping_store)

        # Setup mocks
        child_result = make_search_result("child_1", 0.9)
        parent_chunk = make_chunk("parent_1", "Parent context")
        mapping = make_parent_mapping("child_1", "parent_1")

        storage.search.return_value = [child_result]
        mapping_store.get_mapping.return_value = mapping
        storage.get_chunk.return_value = parent_chunk

        # Search with expansion
        results = retriever.search_with_expansion("test query", top_k=10)

        # Verify search called correctly
        storage.search.assert_called_once_with("test query", top_k=10)

        # Verify expansion
        assert len(results) == 1
        assert results[0].child_result.chunk_id == "child_1"
        assert results[0].parent_chunk.chunk_id == "parent_1"

    def test_search_with_expansion_no_results(self):
        """Test search_with_expansion with no results."""
        storage = Mock()
        mapping_store = Mock()
        retriever = ParentRetriever(storage, mapping_store)

        storage.search.return_value = []

        results = retriever.search_with_expansion("test query", top_k=10)

        assert results == []


class TestFactoryFunction:
    """Tests for create_parent_retriever factory.

    Rule #4: Focused test class - tests factory function
    """

    @patch("ingestforge.storage.parent_mapping.create_parent_mapping_store")
    def test_create_parent_retriever_defaults(self, mock_create_mapping):
        """Test factory creates retriever with defaults."""
        storage = Mock()
        data_path = Path("/tmp/data")
        mapping_store = Mock()
        mock_create_mapping.return_value = mapping_store

        retriever = create_parent_retriever(storage, data_path)

        # Verify mapping store created
        mock_create_mapping.assert_called_once_with(data_path)

        # Verify defaults
        assert retriever.storage is storage
        assert retriever.mapping_store is mapping_store
        assert retriever.deduplicate_parents is True
        assert retriever.context_window_tokens == 2048
        assert retriever.context_window_chars == 8000

    @patch("ingestforge.storage.parent_mapping.create_parent_mapping_store")
    def test_create_parent_retriever_with_config(self, mock_create_mapping):
        """Test factory uses config settings."""
        storage = Mock()
        data_path = Path("/tmp/data")
        mapping_store = Mock()
        mock_create_mapping.return_value = mapping_store

        # Create config mock
        config = Mock()
        config.retrieval.parent_doc.deduplicate_parents = False
        config.retrieval.parent_doc.context_window_tokens = 1024
        config.retrieval.parent_doc.context_window_chars = 4000

        retriever = create_parent_retriever(storage, data_path, config)

        # Verify config applied
        assert retriever.deduplicate_parents is False
        assert retriever.context_window_tokens == 1024
        assert retriever.context_window_chars == 4000


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - ParentExpandedResult: 3 tests (with parent, without parent, with truncation)
    - ParentRetriever init: 2 tests (defaults, custom settings)
    - Truncation: 5 tests (short content, long content, apply disabled, no change, truncates)
    - Expansion: 6 tests (empty, with parent, without parent, dedup, no dedup, truncation)
    - Search workflow: 2 tests (basic, no results)
    - Factory: 2 tests (defaults, with config)

    Total: 20 tests

Design Decisions:
    1. Focus on parent-child mapping and context expansion
    2. Mock storage and mapping store for isolation
    3. Test deduplication logic thoroughly
    4. Test truncation with various content lengths
    5. Test integration with search workflow
    6. Simple, clear tests that verify parent expansion works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - ParentExpandedResult dataclass creation
    - ParentRetriever initialization with settings
    - Context window truncation (char-based fallback)
    - Truncation application with apply_window flag
    - Parent chunk expansion with mappings
    - Deduplication of parent chunks (on/off)
    - Search integration with automatic expansion
    - Factory function with config support
    - Empty result handling
    - Missing parent handling

Justification:
    - Parent expansion is critical for context-aware retrieval
    - Deduplication prevents redundant parent chunks
    - Context window truncation prevents token overflow
    - Tests verify parent expansion system works correctly
"""
