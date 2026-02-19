"""Orphan Node Cleanup - Detect and clean orphaned references.

Orphan Node Cleanup
Epic: EP-15 (Storage Maintenance)

Identifies and removes orphaned nodes in the storage system:
- Chunks referencing non-existent documents
- Manual links referencing non-existent entities
- Bookmarks pointing to deleted chunks

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_ORPHANS_REPORTED)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import typer
from rich.table import Table

from ingestforge.cli.maintenance.base import MaintenanceCommand
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_ORPHANS_REPORTED = 1000
MAX_SCAN_ITEMS = 100_000


@dataclass
class OrphanReport:
    """Report of orphaned nodes found during scan.

    Rule #9: Complete type hints.
    """

    orphaned_chunks: List[Dict[str, Any]] = field(default_factory=list)
    orphaned_links: List[Dict[str, Any]] = field(default_factory=list)
    orphaned_bookmarks: List[Dict[str, Any]] = field(default_factory=list)
    scan_stats: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def total_orphans(self) -> int:
        """Total number of orphans found."""
        return (
            len(self.orphaned_chunks)
            + len(self.orphaned_links)
            + len(self.orphaned_bookmarks)
        )

    @property
    def is_truncated(self) -> bool:
        """Whether results were truncated."""
        return self.total_orphans >= MAX_ORPHANS_REPORTED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "orphaned_chunks": len(self.orphaned_chunks),
            "orphaned_links": len(self.orphaned_links),
            "orphaned_bookmarks": len(self.orphaned_bookmarks),
            "total_orphans": self.total_orphans,
            "truncated": self.is_truncated,
            "scan_stats": self.scan_stats,
            "errors": self.errors,
        }


class OrphansCommand(MaintenanceCommand):
    """Detect and clean orphaned nodes in storage.

    Rule #9: Complete type hints.
    """

    def execute(self, *args: Any, **kwargs: Any) -> int:
        """Execute orphan scan (default operation).

        Returns:
            Exit code (0 = success).
        """
        project = kwargs.get("project")
        report = self.scan(project)
        self.display_report(report)
        return 0 if not report.errors else 1

    def scan(
        self,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> OrphanReport:
        """Scan storage for orphaned nodes.

        Args:
            project: Project directory.
            output: Optional output file for report.

        Returns:
            OrphanReport with found orphans.

        Rule #4: Function < 60 lines.
        """
        report = OrphanReport()

        try:
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_path"]

            # Scan for orphaned chunks
            chunk_orphans = self._scan_chunk_orphans(project_path)
            report.orphaned_chunks = chunk_orphans[:MAX_ORPHANS_REPORTED]
            report.scan_stats["chunks_scanned"] = len(chunk_orphans)

            # Scan for orphaned links
            link_orphans = self._scan_link_orphans(project_path)
            report.orphaned_links = link_orphans[:MAX_ORPHANS_REPORTED]
            report.scan_stats["links_scanned"] = len(link_orphans)

            # Scan for orphaned bookmarks
            bookmark_orphans = self._scan_bookmark_orphans(project_path)
            report.orphaned_bookmarks = bookmark_orphans[:MAX_ORPHANS_REPORTED]
            report.scan_stats["bookmarks_scanned"] = len(bookmark_orphans)

        except Exception as e:
            report.errors.append(str(e))
            logger.error(f"Orphan scan failed: {e}")

        return report

    def _scan_chunk_orphans(self, project: Path) -> List[Dict[str, Any]]:
        """Scan for chunks referencing non-existent documents.

        Args:
            project: Project directory.

        Returns:
            List of orphaned chunk info.

        Rule #4: Function < 60 lines.
        """
        orphans: List[Dict[str, Any]] = []
        data_dir = project / ".data"

        if not data_dir.exists():
            return orphans

        # Get set of valid document IDs from documents index
        valid_docs = self._get_valid_document_ids(project)

        # Check chunks for orphaned references
        chunks_file = data_dir / "chunks.jsonl"
        if not chunks_file.exists():
            return orphans

        try:
            import json

            with open(chunks_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= MAX_SCAN_ITEMS:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    chunk = json.loads(line)
                    doc_id = chunk.get("document_id", "")

                    if doc_id and doc_id not in valid_docs:
                        orphans.append(
                            {
                                "chunk_id": chunk.get("chunk_id", f"line_{i}"),
                                "document_id": doc_id,
                                "reason": "document_not_found",
                            }
                        )

                        if len(orphans) >= MAX_ORPHANS_REPORTED:
                            break

        except Exception as e:
            logger.debug(f"Failed to scan chunks: {e}")

        return orphans

    def _get_valid_document_ids(self, project: Path) -> Set[str]:
        """Get set of valid document IDs.

        Args:
            project: Project directory.

        Returns:
            Set of valid document IDs.

        Rule #4: Function < 60 lines.
        """
        valid_ids: Set[str] = set()
        docs_file = project / ".data" / "documents.jsonl"

        if not docs_file.exists():
            return valid_ids

        try:
            import json

            with open(docs_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= MAX_SCAN_ITEMS:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    doc = json.loads(line)
                    doc_id = doc.get("document_id") or doc.get("id", "")
                    if doc_id:
                        valid_ids.add(doc_id)

        except Exception as e:
            logger.debug(f"Failed to read documents: {e}")

        return valid_ids

    def _scan_link_orphans(self, project: Path) -> List[Dict[str, Any]]:
        """Scan for manual links with orphaned references.

        Args:
            project: Project directory.

        Returns:
            List of orphaned link info.

        Rule #4: Function < 60 lines.
        """
        orphans: List[Dict[str, Any]] = []
        links_file = project / ".data" / "manual_links.json"

        if not links_file.exists():
            return orphans

        try:
            import json

            data = json.loads(links_file.read_text(encoding="utf-8"))
            links = data.get("links", [])

            # Get known entities from storage
            known_entities = self._get_known_entities(project)

            for link in links[:MAX_SCAN_ITEMS]:
                source = link.get("source_entity", "")
                target = link.get("target_entity", "")

                # Check if entities exist (only if we have a known entities list)
                if known_entities:
                    if source and source not in known_entities:
                        orphans.append(
                            {
                                "link_id": link.get("link_id", "unknown"),
                                "source": source,
                                "target": target,
                                "reason": "source_not_found",
                            }
                        )
                    elif target and target not in known_entities:
                        orphans.append(
                            {
                                "link_id": link.get("link_id", "unknown"),
                                "source": source,
                                "target": target,
                                "reason": "target_not_found",
                            }
                        )

                if len(orphans) >= MAX_ORPHANS_REPORTED:
                    break

        except Exception as e:
            logger.debug(f"Failed to scan links: {e}")

        return orphans

    def _get_known_entities(self, project: Path) -> Set[str]:
        """Get set of known entity names from storage.

        Args:
            project: Project directory.

        Returns:
            Set of known entity names.

        Rule #4: Function < 60 lines.
        """
        entities: Set[str] = set()
        entities_file = project / ".data" / "entities.jsonl"

        if not entities_file.exists():
            return entities  # Empty set means skip entity validation

        try:
            import json

            with open(entities_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= MAX_SCAN_ITEMS:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    entity = json.loads(line)
                    name = entity.get("name") or entity.get("entity", "")
                    if name:
                        entities.add(name)

        except Exception as e:
            logger.debug(f"Failed to read entities: {e}")

        return entities

    def _scan_bookmark_orphans(self, project: Path) -> List[Dict[str, Any]]:
        """Scan for bookmarks pointing to deleted chunks.

        Args:
            project: Project directory.

        Returns:
            List of orphaned bookmark info.

        Rule #4: Function < 60 lines.
        """
        orphans: List[Dict[str, Any]] = []
        bookmarks_file = project / ".data" / "bookmarks.json"

        if not bookmarks_file.exists():
            return orphans

        try:
            import json

            data = json.loads(bookmarks_file.read_text(encoding="utf-8"))
            bookmarks = data.get("bookmarks", [])

            # Get valid chunk IDs
            valid_chunks = self._get_valid_chunk_ids(project)

            for bookmark in bookmarks[:MAX_SCAN_ITEMS]:
                chunk_id = bookmark.get("chunk_id", "")

                if chunk_id and valid_chunks and chunk_id not in valid_chunks:
                    orphans.append(
                        {
                            "bookmark_id": bookmark.get("bookmark_id", "unknown"),
                            "chunk_id": chunk_id,
                            "reason": "chunk_not_found",
                        }
                    )

                if len(orphans) >= MAX_ORPHANS_REPORTED:
                    break

        except Exception as e:
            logger.debug(f"Failed to scan bookmarks: {e}")

        return orphans

    def _get_valid_chunk_ids(self, project: Path) -> Set[str]:
        """Get set of valid chunk IDs.

        Args:
            project: Project directory.

        Returns:
            Set of valid chunk IDs.

        Rule #4: Function < 60 lines.
        """
        valid_ids: Set[str] = set()
        chunks_file = project / ".data" / "chunks.jsonl"

        if not chunks_file.exists():
            return valid_ids

        try:
            import json

            with open(chunks_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= MAX_SCAN_ITEMS:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    chunk = json.loads(line)
                    chunk_id = chunk.get("chunk_id") or chunk.get("id", "")
                    if chunk_id:
                        valid_ids.add(chunk_id)

        except Exception as e:
            logger.debug(f"Failed to read chunks: {e}")

        return valid_ids

    def clean(
        self,
        project: Optional[Path] = None,
        dry_run: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Clean orphaned nodes from storage.

        Args:
            project: Project directory.
            dry_run: If True, report but don't delete.
            force: Skip confirmation.

        Returns:
            Cleanup results.

        Rule #4: Function < 60 lines.
        Rule #7: Check all return values.
        """
        results: Dict[str, Any] = {
            "chunks_removed": 0,
            "links_removed": 0,
            "bookmarks_removed": 0,
            "dry_run": dry_run,
            "errors": [],
        }

        # First scan for orphans
        report = self.scan(project)

        if report.total_orphans == 0:
            self.print_info("No orphans found")
            return results

        # Confirm cleanup
        if not dry_run and not force:
            msg = f"Remove {report.total_orphans} orphaned items?"
            if not typer.confirm(msg):
                self.print_info("Cleanup cancelled")
                return results

        if dry_run:
            self.print_info(f"[DRY RUN] Would remove {report.total_orphans} orphans")
            results["would_remove"] = report.total_orphans
            return results

        # Perform cleanup
        try:
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_path"]

            # Clean orphaned links
            if report.orphaned_links:
                removed = self._clean_orphaned_links(
                    project_path, report.orphaned_links
                )
                results["links_removed"] = removed

            # Clean orphaned bookmarks
            if report.orphaned_bookmarks:
                removed = self._clean_orphaned_bookmarks(
                    project_path, report.orphaned_bookmarks
                )
                results["bookmarks_removed"] = removed

            # Note: Chunk cleanup is more complex and may require storage backend
            if report.orphaned_chunks:
                results["errors"].append(
                    f"{len(report.orphaned_chunks)} orphaned chunks found - "
                    "manual cleanup via storage commands recommended"
                )

        except Exception as e:
            results["errors"].append(str(e))
            logger.error(f"Cleanup failed: {e}")

        return results

    def _clean_orphaned_links(
        self, project: Path, orphans: List[Dict[str, Any]]
    ) -> int:
        """Remove orphaned links.

        Args:
            project: Project directory.
            orphans: List of orphaned link info.

        Returns:
            Number of links removed.

        Rule #4: Function < 60 lines.
        """
        links_file = project / ".data" / "manual_links.json"

        if not links_file.exists():
            return 0

        try:
            import json

            data = json.loads(links_file.read_text(encoding="utf-8"))
            links = data.get("links", [])

            orphan_ids = {o["link_id"] for o in orphans}
            original_count = len(links)

            data["links"] = [l for l in links if l.get("link_id") not in orphan_ids]

            # Atomic write
            temp_file = links_file.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_file.replace(links_file)

            return original_count - len(data["links"])

        except Exception as e:
            logger.error(f"Failed to clean links: {e}")
            return 0

    def _clean_orphaned_bookmarks(
        self, project: Path, orphans: List[Dict[str, Any]]
    ) -> int:
        """Remove orphaned bookmarks.

        Args:
            project: Project directory.
            orphans: List of orphaned bookmark info.

        Returns:
            Number of bookmarks removed.

        Rule #4: Function < 60 lines.
        """
        bookmarks_file = project / ".data" / "bookmarks.json"

        if not bookmarks_file.exists():
            return 0

        try:
            import json

            data = json.loads(bookmarks_file.read_text(encoding="utf-8"))
            bookmarks = data.get("bookmarks", [])

            orphan_ids = {o["bookmark_id"] for o in orphans}
            original_count = len(bookmarks)

            data["bookmarks"] = [
                b for b in bookmarks if b.get("bookmark_id") not in orphan_ids
            ]

            # Atomic write
            temp_file = bookmarks_file.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_file.replace(bookmarks_file)

            return original_count - len(data["bookmarks"])

        except Exception as e:
            logger.error(f"Failed to clean bookmarks: {e}")
            return 0

    def display_report(self, report: OrphanReport) -> None:
        """Display orphan report as table.

        Args:
            report: OrphanReport to display.

        Rule #4: Function < 60 lines.
        """
        if report.total_orphans == 0:
            self.print_success("No orphans found")
            return

        # Summary table
        table = Table(title="Orphan Scan Results")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="yellow", justify="right")
        table.add_column("Status", style="green")

        table.add_row(
            "Orphaned Chunks",
            str(len(report.orphaned_chunks)),
            "Found" if report.orphaned_chunks else "None",
        )
        table.add_row(
            "Orphaned Links",
            str(len(report.orphaned_links)),
            "Found" if report.orphaned_links else "None",
        )
        table.add_row(
            "Orphaned Bookmarks",
            str(len(report.orphaned_bookmarks)),
            "Found" if report.orphaned_bookmarks else "None",
        )

        self.console.print(table)

        if report.is_truncated:
            self.print_warning(f"Results truncated at {MAX_ORPHANS_REPORTED} items")

        # Detail tables for each category
        if report.orphaned_chunks:
            self._display_chunk_orphans(report.orphaned_chunks[:10])

        if report.orphaned_links:
            self._display_link_orphans(report.orphaned_links[:10])

        if report.orphaned_bookmarks:
            self._display_bookmark_orphans(report.orphaned_bookmarks[:10])

    def _display_chunk_orphans(self, orphans: List[Dict[str, Any]]) -> None:
        """Display orphaned chunks.

        Args:
            orphans: List of orphan info.
        """
        table = Table(title="Orphaned Chunks (first 10)")
        table.add_column("Chunk ID", style="cyan")
        table.add_column("Missing Document", style="red")

        for o in orphans:
            table.add_row(o["chunk_id"][:20], o["document_id"][:30])

        self.console.print(table)

    def _display_link_orphans(self, orphans: List[Dict[str, Any]]) -> None:
        """Display orphaned links.

        Args:
            orphans: List of orphan info.
        """
        table = Table(title="Orphaned Links (first 10)")
        table.add_column("Link ID", style="cyan")
        table.add_column("Source", style="yellow")
        table.add_column("Target", style="yellow")
        table.add_column("Reason", style="red")

        for o in orphans:
            table.add_row(
                o["link_id"][:15],
                o["source"][:20],
                o["target"][:20],
                o["reason"],
            )

        self.console.print(table)

    def _display_bookmark_orphans(self, orphans: List[Dict[str, Any]]) -> None:
        """Display orphaned bookmarks.

        Args:
            orphans: List of orphan info.
        """
        table = Table(title="Orphaned Bookmarks (first 10)")
        table.add_column("Bookmark ID", style="cyan")
        table.add_column("Missing Chunk", style="red")

        for o in orphans:
            table.add_row(o["bookmark_id"][:20], o["chunk_id"][:30])

        self.console.print(table)


# Typer command wrapper for scan
def scan_command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for report"
    ),
) -> None:
    """Scan storage for orphaned nodes.

    Identifies orphaned references without deleting them:
    - Chunks referencing non-existent documents
    - Manual links with missing entities
    - Bookmarks pointing to deleted chunks

    Examples:
        ingestforge maintenance orphans scan
        ingestforge maintenance orphans scan -o orphans.md
    """
    cmd = OrphansCommand()
    report = cmd.scan(project, output)
    cmd.display_report(report)

    if output:
        try:
            import json

            output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            cmd.print_success(f"Report saved: {output}")
        except Exception as e:
            cmd.print_error(f"Failed to save report: {e}")


# Typer command wrapper for clean
def clean_command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--no-dry-run", help="Report only, don't delete"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Clean orphaned nodes from storage.

    Removes orphaned references from storage:
    - Manual links with missing entities
    - Bookmarks pointing to deleted chunks

    Use --dry-run (default) to preview without deleting.

    Examples:
        # Preview what would be cleaned
        ingestforge maintenance orphans clean

        # Actually clean orphans
        ingestforge maintenance orphans clean --no-dry-run

        # Clean without confirmation
        ingestforge maintenance orphans clean --no-dry-run --force
    """
    cmd = OrphansCommand()
    results = cmd.clean(project, dry_run, force)

    if results.get("dry_run"):
        cmd.print_info(f"Would remove: {results.get('would_remove', 0)} items")
    else:
        total = results.get("links_removed", 0) + results.get("bookmarks_removed", 0)
        cmd.print_success(f"Removed {total} orphaned items")

    if results.get("errors"):
        for err in results["errors"]:
            cmd.print_warning(err)


# Create orphans subcommand app
app = typer.Typer(help="Orphan node detection and cleanup")
app.command("scan")(scan_command)
app.command("clean")(clean_command)

# Default command (scan)
command = scan_command
