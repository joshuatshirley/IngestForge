"""Entity analysis and search command.

Search for entities across documents, view entity profiles,
and find related entities.

Usage:
    ingestforge analyze entities <entity_name>
    ingestforge analyze entities --list
    ingestforge analyze entities --statistics
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json

import typer

from ingestforge.cli.analyze.base import AnalyzeCommand
from ingestforge.enrichment.entity_linker import EntityLinker, link_entities
from ingestforge.enrichment.entities import EntityExtractor
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class EntityAnalysisCommand(AnalyzeCommand):
    """Analyze entities in corpus."""

    def execute(
        self,
        entity_name: Optional[str] = None,
        list_all: bool = False,
        statistics: bool = False,
        show_cooccurrences: bool = False,
        output_json: Optional[Path] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Execute entity analysis.

        Rule #1: Reduced nesting via early returns
        Rule #4: Function <60 lines (refactored from 82)
        Rule #9: Full type hints

        Args:
            entity_name: Entity to search for
            list_all: List all entities
            statistics: Show entity statistics
            show_cooccurrences: Show co-occurring entities
            output_json: Save results to JSON file
            ctx: Click context

        Returns:
            Exit code (0 = success)
        """
        try:
            # Load chunks and build entity index
            linker = self._load_and_build_index(ctx or {})
            if not linker:
                return 1

            # Route to appropriate operation
            return self._route_operation(
                linker,
                entity_name,
                list_all,
                statistics,
                show_cooccurrences,
                output_json,
            )

        except Exception as e:
            logger.error(f"Entity analysis failed: {e}")
            typer.echo(f"Error: {e}", err=True)
            return 1

    def _load_and_build_index(self, ctx: Dict[str, Any]) -> Optional[EntityLinker]:
        """
        Load chunks from storage and build entity index.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            ctx: Click context

        Returns:
            EntityLinker if successful, None otherwise
        """
        storage = ctx.get("storage") if ctx else None
        if not storage:
            typer.echo("Error: No storage backend available", err=True)
            return None

        # Load chunks
        all_chunks = self._load_chunks_from_storage(storage)
        if not all_chunks:
            return None

        typer.echo(f"Loaded {len(all_chunks)} chunks")

        # Build entity index
        typer.echo("Building entity index...")
        extractor = EntityExtractor(use_spacy=True)
        linker, entity_index = link_entities(all_chunks, extractor)

        typer.echo(f"Found {len(entity_index)} unique entities\n")
        return linker

    def _load_chunks_from_storage(self, storage: Any) -> list:
        """
        Load all chunks from storage backend.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            storage: Storage backend

        Returns:
            List of chunks, empty list if failed
        """
        typer.echo("Loading chunks from storage...")

        try:
            # Try storage-specific method first
            if hasattr(storage, "get_all_chunks"):
                return storage.get_all_chunks()

            # Fallback: search for empty query
            results = storage.search_semantic("", limit=10000)
            return results.get("chunks", [])

        except Exception as e:
            logger.warning(f"Could not load chunks: {e}")
            typer.echo(f"Warning: Could not load chunks: {e}", err=True)
            return []

    def _route_operation(
        self,
        linker: EntityLinker,
        entity_name: Optional[str],
        list_all: bool,
        statistics: bool,
        show_cooccurrences: bool,
        output_json: Optional[Path],
    ) -> int:
        """
        Route to appropriate analysis operation.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            linker: Entity linker with built index
            entity_name: Entity to search for
            list_all: List all entities
            statistics: Show entity statistics
            show_cooccurrences: Show co-occurring entities
            output_json: Save results to JSON file

        Returns:
            Exit code (0 = success)
        """
        if statistics:
            return self._show_statistics(linker, output_json)

        if list_all:
            return self._list_entities(linker, output_json)

        if entity_name:
            return self._search_entity(
                linker, entity_name, show_cooccurrences, output_json
            )

        # Default: show statistics
        return self._show_statistics(linker, output_json)

    def _show_statistics(
        self, linker: EntityLinker, output_json: Optional[Path]
    ) -> int:
        """Show entity index statistics."""
        stats = linker.get_statistics()

        typer.echo("Entity Index Statistics")
        typer.echo("=" * 50)
        typer.echo(f"Total unique entities: {stats['total_entities']}")
        typer.echo(f"Total mentions: {stats['total_mentions']}")
        typer.echo(f"Avg mentions per entity: {stats['avg_mentions_per_entity']:.1f}")
        typer.echo("\nEntity Types:")

        for entity_type, count in sorted(
            stats["entity_types"].items(), key=lambda x: x[1], reverse=True
        ):
            typer.echo(f"  {entity_type}: {count}")

        if output_json:
            with open(output_json, "w") as f:
                json.dump(stats, f, indent=2)
            typer.echo(f"\nStatistics saved to: {output_json}")

        return 0

    def _list_entities(self, linker: EntityLinker, output_json: Optional[Path]) -> int:
        """List all entities."""
        exported = linker.export_index()

        # Sort by mention count (descending)
        exported.sort(key=lambda x: x["mention_count"], reverse=True)

        typer.echo("All Entities")
        typer.echo("=" * 50)

        for entity in exported[:50]:  # Limit to top 50
            name = entity["canonical_name"]
            etype = entity["entity_type"]
            count = entity["mention_count"]
            docs = len(entity["documents"])

            typer.echo(f"{name} ({etype})")
            typer.echo(f"  Mentions: {count} | Documents: {docs}")

            if len(entity["variations"]) > 1:
                variations = ", ".join(entity["variations"][:3])
                typer.echo(f"  Variations: {variations}")

            typer.echo()

        if len(exported) > 50:
            typer.echo(f"... and {len(exported) - 50} more entities")

        if output_json:
            with open(output_json, "w") as f:
                json.dump(exported, f, indent=2)
            typer.echo(f"\nFull list saved to: {output_json}")

        return 0

    def _search_entity(
        self,
        linker: EntityLinker,
        entity_name: str,
        show_cooccurrences: bool,
        output_json: Optional[Path],
    ) -> int:
        """Search for specific entity."""
        profile = linker.search_by_entity(entity_name)

        if not profile:
            typer.echo(f"Entity not found: {entity_name}", err=True)
            return 1

        typer.echo(f"Entity: {profile.canonical_name}")
        typer.echo("=" * 50)
        typer.echo(f"Type: {profile.entity_type}")
        typer.echo(f"Total mentions: {profile.mention_count}")
        typer.echo(f"Documents: {len(profile.documents)}")
        typer.echo(f"Chunks: {len(profile.chunks)}")

        if len(profile.variations) > 1:
            typer.echo("\nVariations found:")
            for variation in sorted(profile.variations):
                typer.echo(f"  - {variation}")

        typer.echo("\nDocuments containing this entity:")
        for doc_id in sorted(profile.documents)[:20]:
            typer.echo(f"  - {doc_id}")

        if len(profile.documents) > 20:
            typer.echo(f"  ... and {len(profile.documents) - 20} more")

        if show_cooccurrences:
            typer.echo("\nRelated entities (co-occurring):")
            cooccurrences = linker.get_entity_cooccurrences(entity_name)

            # Sort by count
            sorted_cooccurrences = sorted(
                cooccurrences.items(), key=lambda x: x[1], reverse=True
            )

            for related_entity, count in sorted_cooccurrences[:10]:
                typer.echo(f"  - {related_entity} ({count} shared chunks)")

        if output_json:
            result = profile.to_dict()
            if show_cooccurrences:
                result["cooccurrences"] = linker.get_entity_cooccurrences(entity_name)

            with open(output_json, "w") as f:
                json.dump(result, f, indent=2)
            typer.echo(f"\nResults saved to: {output_json}")

        return 0


def command(
    entity_name: Optional[str] = typer.Argument(None, help="Entity name to search for"),
    list_all: bool = typer.Option(False, "--list", help="List all entities in corpus"),
    statistics: bool = typer.Option(
        False, "--statistics", help="Show entity index statistics"
    ),
    show_cooccurrences: bool = typer.Option(
        False, "--related", help="Show related entities (co-occurrences)"
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output-json", help="Save results to JSON file"
    ),
) -> None:
    """
    Analyze entities in corpus.

    Search for entities, view profiles, and find related entities.

    Examples:

        # Show entity statistics
        ingestforge analyze entities --statistics

        # List all entities
        ingestforge analyze entities --list

        # Search for specific entity
        ingestforge analyze entities "Microsoft"

        # Find related entities
        ingestforge analyze entities "Microsoft" --related

        # Export to JSON
        ingestforge analyze entities --list --output-json entities.json
    """
    cmd = EntityAnalysisCommand()
    exit_code = cmd.execute(
        entity_name=entity_name,
        list_all=list_all,
        statistics=statistics,
        show_cooccurrences=show_cooccurrences,
        output_json=output_json,
        ctx=None,  # Typer handles context automatically
    )
    raise typer.Exit(code=exit_code)
