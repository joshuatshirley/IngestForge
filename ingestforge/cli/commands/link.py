"""Link command - Create manual relationships between entities.

Manual Graph Linker
Allows users to manually create edges between entities in the knowledge graph.

NASA JPL Power of Ten Rules:
- Rule #4: Functions <60 lines
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class LinkCommand(IngestForgeCommand):
    """Create a manual link between two entities."""

    def execute(
        self,
        source: str,
        target: str,
        relation: str = "related_to",
        notes: str = "",
        project: Optional[Path] = None,
    ) -> int:
        """Execute link command.

        Create manual edges between entities.
        Rule #4: <60 lines.
        Rule #7: Parameter validation.

        Args:
            source: Source entity name.
            target: Target entity name.
            relation: Relationship type (default: related_to).
            notes: Optional notes about the relationship.
            project: Project directory.

        Returns:
            0 on success, 1 on error.
        """
        try:
            self.validate_non_empty_string(source, "source")
            self.validate_non_empty_string(target, "target")

            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)

            # Get link manager
            from ingestforge.storage.manual_links import get_link_manager

            data_dir = ctx.get("data_dir", Path.cwd() / ".data")
            manager = get_link_manager(data_dir)

            # Create link
            link = ProgressManager.run_with_spinner(
                lambda: manager.add(source, target, relation, notes),
                "Creating link...",
                "Link created",
            )

            self.print_success(f"Created link: {source} --[{relation}]--> {target}")
            self.console.print(f"  [dim]ID: {link.link_id}[/dim]")

            if notes:
                self.console.print(f"  [dim]Notes: {notes}[/dim]")

            return 0

        except ValueError as e:
            self.print_error(str(e))
            return 1
        except Exception as e:
            return self.handle_error(e, "Link command failed")


class ListLinksCommand(IngestForgeCommand):
    """List all manual links in the knowledge base."""

    def execute(
        self,
        entity: Optional[str] = None,
        limit: int = 50,
        project: Optional[Path] = None,
    ) -> int:
        """Execute list links command.

        Rule #4: <60 lines.

        Args:
            entity: Optional entity to filter links for.
            limit: Maximum number of links to show.
            project: Project directory.

        Returns:
            0 on success, 1 on error.
        """
        try:
            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)

            # Get link manager
            from ingestforge.storage.manual_links import get_link_manager

            data_dir = ctx.get("data_dir", Path.cwd() / ".data")
            manager = get_link_manager(data_dir)

            # Get links
            if entity:
                links = manager.get_for_entity(entity)
                title = f"Links for '{entity}'"
            else:
                links = manager.list_all(limit=limit)
                title = "Manual Links"

            if not links:
                self.print_info("No manual links found")
                if entity:
                    self.print_info(f"No links involve entity '{entity}'")
                return 0

            # Display links
            self._display_links(links, title)
            return 0

        except Exception as e:
            return self.handle_error(e, "List links failed")

    def _display_links(self, links: list, title: str) -> None:
        """Display links in a table.

        Rule #4: <60 lines.

        Args:
            links: List of ManualLink objects.
            title: Table title.
        """
        from rich.table import Table

        table = Table(title=title, show_lines=False)
        table.add_column("ID", style="dim", width=16)
        table.add_column("Source", style="cyan")
        table.add_column("Relation", style="yellow")
        table.add_column("Target", style="green")
        table.add_column("Notes", style="dim", max_width=30)

        for link in links:
            notes_preview = (
                link.notes[:27] + "..." if len(link.notes) > 30 else link.notes
            )
            table.add_row(
                link.link_id,
                link.source_entity,
                link.relation,
                link.target_entity,
                notes_preview,
            )

        self.console.print(table)
        self.console.print(f"\n[dim]Total: {len(links)} link(s)[/dim]")


class DeleteLinkCommand(IngestForgeCommand):
    """Delete a manual link."""

    def execute(
        self,
        link_id: str,
        project: Optional[Path] = None,
    ) -> int:
        """Execute delete link command.

        Rule #4: <60 lines.

        Args:
            link_id: ID of the link to delete.
            project: Project directory.

        Returns:
            0 on success, 1 on error.
        """
        try:
            self.validate_non_empty_string(link_id, "link_id")

            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)

            # Get link manager
            from ingestforge.storage.manual_links import get_link_manager

            data_dir = ctx.get("data_dir", Path.cwd() / ".data")
            manager = get_link_manager(data_dir)

            # Delete link
            deleted = manager.delete(link_id)

            if deleted:
                self.print_success(f"Deleted link {link_id}")
                return 0
            else:
                self.print_error(f"Link '{link_id}' not found")
                return 1

        except Exception as e:
            return self.handle_error(e, "Delete link failed")


# =============================================================================
# Typer command wrappers
# =============================================================================


def command(
    source: str = typer.Argument(..., help="Source entity name"),
    target: str = typer.Argument(..., help="Target entity name"),
    relation: str = typer.Option(
        "related_to",
        "--relation",
        "-r",
        help="Relationship type (e.g., related_to, causes, contains)",
    ),
    notes: str = typer.Option(
        "",
        "--notes",
        "-n",
        help="Optional notes about the relationship",
    ),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Project directory",
    ),
) -> None:
    """Create a manual link between two entities.

    Manually connect entities in the knowledge graph with a specified
    relationship type. Links persist across sessions.

    Examples:
        ingestforge link "Einstein" "Relativity" -r "discovered"
        ingestforge link "Chapter 1" "Chapter 2" -r "precedes" -n "Narrative flow"
    """
    cmd = LinkCommand()
    result = cmd.execute(source, target, relation, notes, project)
    raise typer.Exit(result)


def list_links_command(
    entity: Optional[str] = typer.Option(
        None,
        "--entity",
        "-e",
        help="Filter links by entity name",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        "-l",
        help="Maximum number of links to show",
    ),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Project directory",
    ),
) -> None:
    """List all manual links in the knowledge graph.

    Shows manually created relationships between entities.

    Examples:
        ingestforge links
        ingestforge links --entity "Einstein"
    """
    cmd = ListLinksCommand()
    result = cmd.execute(entity, limit, project)
    raise typer.Exit(result)


def delete_link_command(
    link_id: str = typer.Argument(..., help="ID of the link to delete"),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Project directory",
    ),
) -> None:
    """Delete a manual link from the knowledge graph.

    Examples:
        ingestforge unlink link_abc123
    """
    cmd = DeleteLinkCommand()
    result = cmd.execute(link_id, project)
    raise typer.Exit(result)
