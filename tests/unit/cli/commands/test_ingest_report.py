"""Tests for ingestion report command (SEC-001.3).

NASA JPL Commandments compliance:
- Rule #1: Linear test structure
- Rule #4: Functions <60 lines
"""

from __future__ import annotations


from ingestforge.cli.commands.ingest_report import (
    DocumentRedactionStats,
    RedactionReport,
    IngestReportCommand,
    MAX_DOCUMENTS_IN_REPORT,
    MAX_TOP_DOCUMENTS,
)


class TestDocumentRedactionStats:
    """Tests for DocumentRedactionStats dataclass."""

    def test_basic_creation(self) -> None:
        """DocumentRedactionStats stores document data."""
        stats = DocumentRedactionStats(
            document_id="doc-123",
            source_name="document.pdf",
        )
        assert stats.document_id == "doc-123"
        assert stats.source_name == "document.pdf"

    def test_default_values(self) -> None:
        """Default values are zero."""
        stats = DocumentRedactionStats(
            document_id="doc-1",
            source_name="test.pdf",
        )
        assert stats.total_redactions == 0
        assert stats.by_type == {}
        assert stats.skipped == 0

    def test_with_stats(self) -> None:
        """Stats can be set."""
        stats = DocumentRedactionStats(
            document_id="doc-1",
            source_name="test.pdf",
            total_redactions=10,
            by_type={"email": 5, "phone": 5},
            skipped=2,
        )
        assert stats.total_redactions == 10
        assert stats.by_type["email"] == 5
        assert stats.skipped == 2


class TestRedactionReport:
    """Tests for RedactionReport dataclass."""

    def test_empty_report(self) -> None:
        """Empty report has default values."""
        report = RedactionReport()
        assert report.total_documents == 0
        assert report.total_redactions == 0
        assert report.by_type == {}
        assert report.total_skipped == 0
        assert report.documents == []

    def test_report_with_data(self) -> None:
        """Report stores aggregate data."""
        report = RedactionReport(
            total_documents=5,
            total_redactions=100,
            by_type={"email": 50, "phone": 50},
            total_skipped=10,
        )
        assert report.total_documents == 5
        assert report.total_redactions == 100

    def test_top_documents(self) -> None:
        """Top documents returns highest redaction counts."""
        docs = [
            DocumentRedactionStats("d1", "a.pdf", total_redactions=10),
            DocumentRedactionStats("d2", "b.pdf", total_redactions=50),
            DocumentRedactionStats("d3", "c.pdf", total_redactions=30),
        ]
        report = RedactionReport(documents=docs)

        top = report.top_documents
        assert len(top) == 3
        assert top[0].total_redactions == 50  # Highest first
        assert top[1].total_redactions == 30
        assert top[2].total_redactions == 10

    def test_top_documents_limit(self) -> None:
        """Top documents respects MAX_TOP_DOCUMENTS."""
        docs = [
            DocumentRedactionStats(f"d{i}", f"{i}.pdf", total_redactions=i)
            for i in range(20)
        ]
        report = RedactionReport(documents=docs)

        top = report.top_documents
        assert len(top) == MAX_TOP_DOCUMENTS


class TestConstants:
    """Tests for module constants."""

    def test_max_documents_bound(self) -> None:
        """MAX_DOCUMENTS_IN_REPORT is bounded."""
        assert MAX_DOCUMENTS_IN_REPORT == 100

    def test_max_top_documents_bound(self) -> None:
        """MAX_TOP_DOCUMENTS is bounded."""
        assert MAX_TOP_DOCUMENTS == 10


class TestIngestReportCommand:
    """Tests for IngestReportCommand class."""

    def test_command_instantiation(self) -> None:
        """Command can be instantiated."""
        cmd = IngestReportCommand()
        assert cmd is not None

    def test_gather_empty_stats(self) -> None:
        """Gather stats handles empty storage."""

        class MockStorage:
            def get_all_chunks(self):
                return []

        cmd = IngestReportCommand()
        report = cmd._gather_redaction_stats(MockStorage(), None)
        assert report.total_documents == 0
        assert report.total_redactions == 0

    def test_gather_stats_no_redaction_data(self) -> None:
        """Gather stats handles chunks without redaction data."""

        class MockChunk:
            metadata = {"document_id": "doc-1", "source": "test.pdf"}

        class MockStorage:
            def get_all_chunks(self):
                return [MockChunk()]

        cmd = IngestReportCommand()
        report = cmd._gather_redaction_stats(MockStorage(), None)
        assert report.total_documents == 0

    def test_gather_stats_with_redaction_data(self) -> None:
        """Gather stats extracts redaction data."""

        class MockChunk:
            metadata = {
                "document_id": "doc-1",
                "source": "test.pdf",
                "redaction_stats": {
                    "total": 5,
                    "by_type": {"email": 3, "phone": 2},
                    "skipped": 1,
                },
            }

        class MockStorage:
            def get_all_chunks(self):
                return [MockChunk()]

        cmd = IngestReportCommand()
        report = cmd._gather_redaction_stats(MockStorage(), None)
        assert report.total_documents == 1
        assert report.total_redactions == 5
        assert report.by_type["email"] == 3

    def test_gather_stats_aggregates_chunks(self) -> None:
        """Gather stats aggregates multiple chunks."""

        class MockChunk1:
            metadata = {
                "document_id": "doc-1",
                "source": "test.pdf",
                "redaction_stats": {"total": 3, "by_type": {"email": 3}},
            }

        class MockChunk2:
            metadata = {
                "document_id": "doc-1",
                "source": "test.pdf",
                "redaction_stats": {"total": 2, "by_type": {"phone": 2}},
            }

        class MockStorage:
            def get_all_chunks(self):
                return [MockChunk1(), MockChunk2()]

        cmd = IngestReportCommand()
        report = cmd._gather_redaction_stats(MockStorage(), None)
        assert report.total_documents == 1
        assert report.total_redactions == 5
        assert report.by_type["email"] == 3
        assert report.by_type["phone"] == 2

    def test_gather_stats_filters_by_document(self) -> None:
        """Gather stats filters by document_id."""

        class MockChunk1:
            metadata = {
                "document_id": "doc-1",
                "source": "a.pdf",
                "redaction_stats": {"total": 10},
            }

        class MockChunk2:
            metadata = {
                "document_id": "doc-2",
                "source": "b.pdf",
                "redaction_stats": {"total": 20},
            }

        class MockStorage:
            def get_all_chunks(self):
                return [MockChunk1(), MockChunk2()]

        cmd = IngestReportCommand()
        report = cmd._gather_redaction_stats(MockStorage(), "doc-1")
        assert report.total_documents == 1
        assert report.total_redactions == 10

    def test_gather_stats_handles_missing_get_all_chunks(self) -> None:
        """Gather stats handles storage without get_all_chunks."""

        class MockStorage:
            pass

        cmd = IngestReportCommand()
        report = cmd._gather_redaction_stats(MockStorage(), None)
        assert report.total_documents == 0

    def test_gather_stats_handles_exception(self) -> None:
        """Gather stats handles storage exceptions."""

        class MockStorage:
            def get_all_chunks(self):
                raise RuntimeError("Storage error")

        cmd = IngestReportCommand()
        report = cmd._gather_redaction_stats(MockStorage(), None)
        assert report.total_documents == 0
