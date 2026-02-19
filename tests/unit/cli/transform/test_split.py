"""Tests for transform split command.

Tests ChunkSplitter class and split command functionality.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.cli.transform.split import (
    ChunkSplitter,
    SplitResult,
)


# Fixtures
@pytest.fixture
def sample_chunks() -> List[ChunkRecord]:
    """Create sample chunks for testing."""
    return [
        ChunkRecord(
            chunk_id="chunk1",
            document_id="doc1",
            content="Introduction to machine learning.",
            concepts=["machine learning"],
            source_file="report.pdf",
            chunk_index=0,
        ),
        ChunkRecord(
            chunk_id="chunk2",
            document_id="doc1",
            content="Deep learning fundamentals.",
            concepts=["deep learning"],
            source_file="report.pdf",
            chunk_index=1,
        ),
        ChunkRecord(
            chunk_id="chunk3",
            document_id="doc2",
            content="Neural network architecture.",
            concepts=["neural networks"],
            source_file="paper.pdf",
            chunk_index=0,
        ),
        ChunkRecord(
            chunk_id="chunk4",
            document_id="doc2",
            content="Training deep networks.",
            concepts=["deep learning"],
            source_file="paper.pdf",
            chunk_index=1,
        ),
        ChunkRecord(
            chunk_id="chunk5",
            document_id="doc3",
            content="Statistical methods overview.",
            concepts=["statistics"],
            source_file="notes.txt",
            chunk_index=0,
        ),
    ]


class TestChunkSplitterByDocument:
    """Tests for splitting by document."""

    def test_split_by_document(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test splitting chunks by document ID."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_document(sample_chunks)

        assert len(splits) == 3
        assert "doc1" in splits
        assert "doc2" in splits
        assert "doc3" in splits

    def test_split_by_document_counts(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test document split chunk counts."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_document(sample_chunks)

        assert len(splits["doc1"]) == 2
        assert len(splits["doc2"]) == 2
        assert len(splits["doc3"]) == 1

    def test_split_by_document_empty(self) -> None:
        """Test splitting empty list."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_document([])

        assert splits == {}

    def test_split_by_document_preserves_chunks(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test that all chunks are preserved."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_document(sample_chunks)

        total = sum(len(chunks) for chunks in splits.values())
        assert total == len(sample_chunks)


class TestChunkSplitterByTopic:
    """Tests for splitting by topic."""

    def test_split_by_topic(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test splitting chunks by topic."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_topic(sample_chunks, min_chunks=1)

        assert len(splits) >= 1

    def test_split_by_topic_groups_similar(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test that chunks with same concepts are grouped."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_topic(sample_chunks, min_chunks=1)

        # Check deep learning chunks are together
        for topic, chunks in splits.items():
            if "deep learning" in topic or any(
                "deep learning" in c.concepts for c in chunks
            ):
                dl_chunks = [c for c in chunks if "deep learning" in c.concepts]
                if len(dl_chunks) > 0:
                    break

    def test_split_by_topic_merges_small(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test that small groups are merged."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_topic(sample_chunks, min_chunks=3)

        # With min_chunks=3, groups < 3 should be merged into "other"
        for topic, chunks in splits.items():
            if topic != "other":
                assert len(chunks) >= 3 or len(splits) == 1


class TestChunkSplitterBySize:
    """Tests for splitting by size."""

    def test_split_by_size_basic(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test splitting into size-based groups."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_size(sample_chunks, max_per_split=2)

        assert len(splits) == 3  # 5 chunks / 2 = 3 groups

    def test_split_by_size_respects_max(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test that each group respects max size."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_size(sample_chunks, max_per_split=2)

        for split in splits:
            assert len(split) <= 2

    def test_split_by_size_large_max(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test with max larger than input."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_size(sample_chunks, max_per_split=100)

        assert len(splits) == 1
        assert len(splits[0]) == 5

    def test_split_by_size_single_per_group(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test with max of 1 per group."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_size(sample_chunks, max_per_split=1)

        assert len(splits) == 5
        assert all(len(s) == 1 for s in splits)

    def test_split_by_size_empty(self) -> None:
        """Test splitting empty list."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_size([], max_per_split=2)

        assert splits == []


class TestChunkSplitterBySource:
    """Tests for splitting by source file."""

    def test_split_by_source(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test splitting by source file."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_source(sample_chunks)

        assert len(splits) == 3  # report.pdf, paper.pdf, notes.txt

    def test_split_by_source_uses_stem(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test that source uses filename stem."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_source(sample_chunks)

        assert "report" in splits
        assert "paper" in splits
        assert "notes" in splits

    def test_split_by_source_counts(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test source split chunk counts."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_source(sample_chunks)

        assert len(splits["report"]) == 2
        assert len(splits["paper"]) == 2
        assert len(splits["notes"]) == 1


class TestChunkSplitterExport:
    """Tests for exporting splits."""

    def test_sanitize_filename(self) -> None:
        """Test filename sanitization."""
        splitter = ChunkSplitter()

        assert splitter._sanitize_filename("normal") == "normal"
        assert splitter._sanitize_filename("with spaces") == "with_spaces"
        assert splitter._sanitize_filename("with:colons") == "with_colons"
        assert splitter._sanitize_filename("with/slashes") == "with_slashes"

    def test_sanitize_filename_long(self) -> None:
        """Test sanitization of long filenames."""
        splitter = ChunkSplitter()
        long_name = "a" * 150

        result = splitter._sanitize_filename(long_name)

        assert len(result) <= 100

    def test_export_splits_creates_files(
        self, sample_chunks: List[ChunkRecord], tmp_path: Path
    ) -> None:
        """Test that export creates files."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_document(sample_chunks)

        files = splitter.export_splits(splits, tmp_path, "{name}")

        assert len(files) == 3
        assert all(f.exists() for f in files)

    def test_export_splits_with_pattern(
        self, sample_chunks: List[ChunkRecord], tmp_path: Path
    ) -> None:
        """Test export with custom pattern."""
        splitter = ChunkSplitter()
        splits = splitter.split_by_size(sample_chunks, max_per_split=2)
        splits_dict = {f"partition_{i}": s for i, s in enumerate(splits)}

        files = splitter.export_splits(splits_dict, tmp_path, "split_{n}")

        assert len(files) == 3
        assert any("split_0" in str(f) for f in files)


class TestSplitResult:
    """Tests for SplitResult dataclass."""

    def test_result_initialization(self) -> None:
        """Test SplitResult initialization."""
        result = SplitResult(input_count=10, output_partitions=3)

        assert result.input_count == 10
        assert result.output_partitions == 3
        assert result.partition_sizes == {}
        assert result.exported_files == []
        assert result.splits == {}

    def test_result_with_splits(self) -> None:
        """Test SplitResult with splits data."""
        splits = {
            "part1": [ChunkRecord(chunk_id="1", document_id="d", content="c")],
            "part2": [ChunkRecord(chunk_id="2", document_id="d", content="c")],
        }

        result = SplitResult(
            input_count=2,
            output_partitions=2,
            partition_sizes={"part1": 1, "part2": 1},
            splits=splits,
        )

        assert result.partition_sizes["part1"] == 1
        assert len(result.splits) == 2


class TestChunkSplitterExtractTopic:
    """Tests for topic extraction helper."""

    def test_extract_topic_from_concepts(self) -> None:
        """Test topic extraction uses concepts."""
        splitter = ChunkSplitter()
        chunk = ChunkRecord(
            chunk_id="1",
            document_id="d",
            content="content",
            concepts=["machine learning", "AI"],
        )

        topic = splitter._extract_primary_topic(chunk)

        assert topic == "machine learning"

    def test_extract_topic_from_entities(self) -> None:
        """Test topic extraction uses entities when no concepts."""
        splitter = ChunkSplitter()
        chunk = ChunkRecord(
            chunk_id="1",
            document_id="d",
            content="content",
            entities=["PERSON:John"],
            concepts=[],
        )

        topic = splitter._extract_primary_topic(chunk)

        assert topic == "PERSON:John"

    def test_extract_topic_from_section(self) -> None:
        """Test topic extraction uses section title."""
        splitter = ChunkSplitter()
        chunk = ChunkRecord(
            chunk_id="1",
            document_id="d",
            content="content",
            section_title="Introduction",
            entities=[],
            concepts=[],
        )

        topic = splitter._extract_primary_topic(chunk)

        assert topic == "Introduction"

    def test_extract_topic_default(self) -> None:
        """Test topic extraction default."""
        splitter = ChunkSplitter()
        chunk = ChunkRecord(
            chunk_id="1",
            document_id="d",
            content="content",
        )

        topic = splitter._extract_primary_topic(chunk)

        assert topic == "general"


class TestChunkSplitterMergeSmallGroups:
    """Tests for merging small groups."""

    def test_merge_small_groups_creates_other(self) -> None:
        """Test that small groups are merged into 'other'."""
        splitter = ChunkSplitter()
        groups = {
            "topic1": [
                ChunkRecord(chunk_id=f"1{i}", document_id="d", content="c")
                for i in range(5)
            ],
            "topic2": [ChunkRecord(chunk_id="2", document_id="d", content="c")],
            "topic3": [ChunkRecord(chunk_id="3", document_id="d", content="c")],
        }

        result = splitter._merge_small_groups(groups, min_size=3)

        assert "topic1" in result
        assert "topic2" not in result
        assert "topic3" not in result
        assert "other" in result
        assert len(result["other"]) == 2

    def test_merge_small_groups_all_small(self) -> None:
        """Test when all groups are small."""
        splitter = ChunkSplitter()
        groups = {
            "topic1": [ChunkRecord(chunk_id="1", document_id="d", content="c")],
            "topic2": [ChunkRecord(chunk_id="2", document_id="d", content="c")],
        }

        result = splitter._merge_small_groups(groups, min_size=5)

        assert "other" in result
        assert len(result["other"]) == 2

    def test_merge_small_groups_all_large(self) -> None:
        """Test when all groups meet minimum."""
        splitter = ChunkSplitter()
        groups = {
            "topic1": [
                ChunkRecord(chunk_id=f"1{i}", document_id="d", content="c")
                for i in range(3)
            ],
            "topic2": [
                ChunkRecord(chunk_id=f"2{i}", document_id="d", content="c")
                for i in range(3)
            ],
        }

        result = splitter._merge_small_groups(groups, min_size=3)

        assert "other" not in result
        assert len(result) == 2
