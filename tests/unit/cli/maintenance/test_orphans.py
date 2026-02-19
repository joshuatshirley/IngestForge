"""Tests for Orphan Node Cleanup.

Orphan Node Cleanup
Tests orphan detection and cleanup functionality.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


from ingestforge.cli.maintenance.orphans import (
    OrphanReport,
    OrphansCommand,
    MAX_ORPHANS_REPORTED,
    MAX_SCAN_ITEMS,
)


def create_test_project(project: Path) -> None:
    """Create minimal project structure for testing.

    Args:
        project: Project directory path.
    """
    # Create .ingestforge directory
    ingestforge_dir = project / ".ingestforge"
    ingestforge_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal config file
    config = {
        "name": "test-project",
        "version": "1.0.0",
    }
    (project / "ingestforge.yaml").write_text(
        f"name: {config['name']}\nversion: {config['version']}\n",
        encoding="utf-8",
    )


class TestOrphanReport:
    """Tests for OrphanReport dataclass."""

    def test_empty_report(self) -> None:
        """Test empty report has zero orphans."""
        report = OrphanReport()

        assert report.total_orphans == 0
        assert not report.is_truncated
        assert report.orphaned_chunks == []
        assert report.orphaned_links == []
        assert report.orphaned_bookmarks == []

    def test_report_with_orphans(self) -> None:
        """Test report counts orphans correctly."""
        report = OrphanReport(
            orphaned_chunks=[{"chunk_id": "c1"}],
            orphaned_links=[{"link_id": "l1"}, {"link_id": "l2"}],
            orphaned_bookmarks=[{"bookmark_id": "b1"}],
        )

        assert report.total_orphans == 4
        assert len(report.orphaned_chunks) == 1
        assert len(report.orphaned_links) == 2
        assert len(report.orphaned_bookmarks) == 1

    def test_report_truncation(self) -> None:
        """Test truncation flag when at limit."""
        # Create report at max limit
        report = OrphanReport(
            orphaned_chunks=[{"id": f"c{i}"} for i in range(MAX_ORPHANS_REPORTED)],
        )

        assert report.is_truncated

    def test_report_to_dict(self) -> None:
        """Test conversion to dictionary."""
        report = OrphanReport(
            orphaned_chunks=[{"chunk_id": "c1"}],
            scan_stats={"chunks_scanned": 100},
        )

        data = report.to_dict()

        assert data["orphaned_chunks"] == 1
        assert data["total_orphans"] == 1
        assert data["scan_stats"]["chunks_scanned"] == 100


class TestOrphansCommand:
    """Tests for OrphansCommand class."""

    def test_scan_empty_project(self) -> None:
        """Test scanning empty project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            create_test_project(project)

            cmd = OrphansCommand()
            report = cmd.scan(project)

            assert report.total_orphans == 0
            assert report.errors == []

    def test_scan_no_data_dir(self) -> None:
        """Test scanning project without .data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            create_test_project(project)

            cmd = OrphansCommand()
            report = cmd.scan(project)

            assert report.total_orphans == 0

    def test_scan_finds_orphaned_chunks(self) -> None:
        """Test detection of orphaned chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            create_test_project(project)
            data_dir = project / ".data"
            data_dir.mkdir(parents=True)

            # Create documents file with one document
            docs = [{"document_id": "doc-001", "title": "Valid Doc"}]
            (data_dir / "documents.jsonl").write_text(
                "\n".join(json.dumps(d) for d in docs),
                encoding="utf-8",
            )

            # Create chunks file with orphaned reference
            chunks = [
                {"chunk_id": "chunk-001", "document_id": "doc-001"},  # Valid
                {"chunk_id": "chunk-002", "document_id": "doc-999"},  # Orphan
            ]
            (data_dir / "chunks.jsonl").write_text(
                "\n".join(json.dumps(c) for c in chunks),
                encoding="utf-8",
            )

            cmd = OrphansCommand()
            report = cmd.scan(project)

            assert len(report.orphaned_chunks) == 1
            assert report.orphaned_chunks[0]["chunk_id"] == "chunk-002"
            assert report.orphaned_chunks[0]["document_id"] == "doc-999"

    def test_scan_finds_orphaned_links(self) -> None:
        """Test detection of orphaned manual links."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            create_test_project(project)
            data_dir = project / ".data"
            data_dir.mkdir(parents=True)

            # Create entities file
            entities = [
                {"name": "EntityA"},
                {"name": "EntityB"},
            ]
            (data_dir / "entities.jsonl").write_text(
                "\n".join(json.dumps(e) for e in entities),
                encoding="utf-8",
            )

            # Create links file with orphaned reference
            links_data = {
                "version": 1,
                "links": [
                    {
                        "link_id": "link-001",
                        "source_entity": "EntityA",
                        "target_entity": "EntityB",
                    },  # Valid
                    {
                        "link_id": "link-002",
                        "source_entity": "EntityA",
                        "target_entity": "EntityZ",
                    },  # Orphan
                ],
            }
            (data_dir / "manual_links.json").write_text(
                json.dumps(links_data),
                encoding="utf-8",
            )

            cmd = OrphansCommand()
            report = cmd.scan(project)

            assert len(report.orphaned_links) == 1
            assert report.orphaned_links[0]["link_id"] == "link-002"
            assert report.orphaned_links[0]["reason"] == "target_not_found"

    def test_scan_finds_orphaned_bookmarks(self) -> None:
        """Test detection of orphaned bookmarks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            create_test_project(project)
            data_dir = project / ".data"
            data_dir.mkdir(parents=True)

            # Create chunks file
            chunks = [{"chunk_id": "chunk-001"}]
            (data_dir / "chunks.jsonl").write_text(
                "\n".join(json.dumps(c) for c in chunks),
                encoding="utf-8",
            )

            # Create bookmarks file with orphaned reference
            bookmarks_data = {
                "version": 1,
                "bookmarks": [
                    {"bookmark_id": "bm-001", "chunk_id": "chunk-001"},  # Valid
                    {"bookmark_id": "bm-002", "chunk_id": "chunk-999"},  # Orphan
                ],
            }
            (data_dir / "bookmarks.json").write_text(
                json.dumps(bookmarks_data),
                encoding="utf-8",
            )

            cmd = OrphansCommand()
            report = cmd.scan(project)

            assert len(report.orphaned_bookmarks) == 1
            assert report.orphaned_bookmarks[0]["bookmark_id"] == "bm-002"
            assert report.orphaned_bookmarks[0]["chunk_id"] == "chunk-999"


class TestCleanup:
    """Tests for orphan cleanup functionality."""

    def test_clean_dry_run(self) -> None:
        """Test dry run doesn't delete anything."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            create_test_project(project)
            data_dir = project / ".data"
            data_dir.mkdir(parents=True)

            # Create links file
            links_data = {
                "version": 1,
                "links": [
                    {"link_id": "link-001", "source_entity": "A", "target_entity": "B"},
                ],
            }
            links_file = data_dir / "manual_links.json"
            links_file.write_text(json.dumps(links_data), encoding="utf-8")

            cmd = OrphansCommand()
            results = cmd.clean(project, dry_run=True, force=True)

            assert results["dry_run"] is True
            # File should be unchanged
            assert links_file.exists()

    def test_clean_removes_orphaned_links(self) -> None:
        """Test actual cleanup of orphaned links."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            create_test_project(project)
            data_dir = project / ".data"
            data_dir.mkdir(parents=True)

            # Create entities file
            entities = [{"name": "EntityA"}]
            (data_dir / "entities.jsonl").write_text(
                "\n".join(json.dumps(e) for e in entities),
                encoding="utf-8",
            )

            # Create links file with orphan
            links_data = {
                "version": 1,
                "links": [
                    {
                        "link_id": "link-001",
                        "source_entity": "EntityA",
                        "target_entity": "EntityZ",
                    },  # Orphan
                ],
            }
            links_file = data_dir / "manual_links.json"
            links_file.write_text(json.dumps(links_data), encoding="utf-8")

            cmd = OrphansCommand()
            results = cmd.clean(project, dry_run=False, force=True)

            assert results["links_removed"] == 1

            # Verify file was updated
            updated_data = json.loads(links_file.read_text(encoding="utf-8"))
            assert len(updated_data["links"]) == 0

    def test_clean_removes_orphaned_bookmarks(self) -> None:
        """Test actual cleanup of orphaned bookmarks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            create_test_project(project)
            data_dir = project / ".data"
            data_dir.mkdir(parents=True)

            # Create chunks file with one valid chunk (needed for orphan detection)
            chunks = [{"chunk_id": "chunk-001"}]
            (data_dir / "chunks.jsonl").write_text(
                json.dumps(chunks[0]), encoding="utf-8"
            )

            # Create bookmarks with orphan
            bookmarks_data = {
                "version": 1,
                "bookmarks": [
                    {"bookmark_id": "bm-001", "chunk_id": "chunk-999"},  # Orphan
                ],
            }
            bookmarks_file = data_dir / "bookmarks.json"
            bookmarks_file.write_text(json.dumps(bookmarks_data), encoding="utf-8")

            cmd = OrphansCommand()
            results = cmd.clean(project, dry_run=False, force=True)

            assert results["bookmarks_removed"] == 1

            # Verify file was updated
            updated_data = json.loads(bookmarks_file.read_text(encoding="utf-8"))
            assert len(updated_data["bookmarks"]) == 0


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_bounds_constants_exist(self) -> None:
        """Test that bound constants are defined."""
        assert MAX_ORPHANS_REPORTED > 0
        assert MAX_SCAN_ITEMS > 0

    def test_rule_9_type_hints(self) -> None:
        """Test that key methods have type hints."""
        import inspect

        cmd = OrphansCommand()

        # Check scan method has type hints (forward refs resolve to strings in 3.12+)
        sig = inspect.signature(cmd.scan)
        assert sig.return_annotation is not None
        # Verify it references OrphanReport (as string or class)
        return_hint = str(sig.return_annotation)
        assert "OrphanReport" in return_hint

        # Check clean method
        sig = inspect.signature(cmd.clean)
        # Return type should be Dict[str, Any]
        assert sig.return_annotation is not None

    def test_report_dataclass_immutable_fields(self) -> None:
        """Test OrphanReport has proper defaults."""
        report = OrphanReport()

        # Lists should be independent instances
        report.orphaned_chunks.append({"id": "test"})
        new_report = OrphanReport()

        assert len(new_report.orphaned_chunks) == 0
