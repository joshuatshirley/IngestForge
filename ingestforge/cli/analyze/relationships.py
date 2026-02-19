"""Relationship and Knowledge Graph analysis commands.

Extract and visualize relationships between entities,
build knowledge graphs, and query entity connections.

Usage:
    ingestforge analyze relationships [OPTIONS]
    ingestforge analyze knowledge-graph [OPTIONS]"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import json

import typer

from ingestforge.cli.analyze.base import AnalyzeCommand
from ingestforge.enrichment.relationships import (
    SpacyRelationshipExtractor,
    Relationship,
    get_relationship_types,
    extract_by_type,
)
from ingestforge.enrichment.knowledge_graph import (
    KnowledgeGraphBuilder,
    build_graph_from_text,
    export_to_mermaid_file,
)
from ingestforge.enrichment.ner import NERExtractor
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class RelationshipAnalysisCommand(AnalyzeCommand):
    """Analyze relationships between entities."""

    def execute(
        self,
        relationship_type: Optional[str] = None,
        list_types: bool = False,
        input_file: Optional[Path] = None,
        output_json: Optional[Path] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Execute relationship analysis.

        Args:
            relationship_type: Filter by relationship type
            list_types: List available relationship types
            input_file: Input text file to analyze
            output_json: Save results to JSON
            ctx: Click context

        Returns:
            Exit code (0 = success)

        Rule #1: Early returns for routing
        Rule #4: Function <60 lines
        """
        try:
            if list_types:
                return self._list_relationship_types()

            if input_file:
                return self._analyze_file(input_file, relationship_type, output_json)

            return self._analyze_corpus(relationship_type, output_json, ctx)

        except Exception as e:
            logger.error(f"Relationship analysis failed: {e}")
            typer.echo(f"Error: {e}", err=True)
            return 1

    def _list_relationship_types(self) -> int:
        """List available relationship types.

        Returns:
            Exit code (0 = success)

        Rule #4: Function <60 lines
        """
        types = get_relationship_types()

        typer.echo("Available Relationship Types")
        typer.echo("=" * 40)

        for rel_type in sorted(types):
            typer.echo(f"  - {rel_type}")

        typer.echo(f"\nTotal: {len(types)} types")
        return 0

    def _analyze_file(
        self,
        input_file: Path,
        relationship_type: Optional[str],
        output_json: Optional[Path],
    ) -> int:
        """Analyze relationships in a file.

        Args:
            input_file: Path to input file
            relationship_type: Optional type filter
            output_json: Optional output path

        Returns:
            Exit code

        Rule #4: Function <60 lines
        """
        if not input_file.exists():
            typer.echo(f"File not found: {input_file}", err=True)
            return 1

        text = input_file.read_text(encoding="utf-8")
        typer.echo(f"Analyzing: {input_file}")

        # Extract relationships
        if relationship_type:
            relationships = extract_by_type(text, relationship_type)
        else:
            extractor = SpacyRelationshipExtractor(use_spacy=True)
            relationships = extractor.extract(text)

        self._display_relationships(relationships)

        if output_json:
            self._save_json(relationships, output_json)

        return 0

    def _analyze_corpus(
        self,
        relationship_type: Optional[str],
        output_json: Optional[Path],
        ctx: Optional[Dict[str, Any]],
    ) -> int:
        """Analyze relationships in corpus.

        Args:
            relationship_type: Optional type filter
            output_json: Optional output path
            ctx: Context with storage

        Returns:
            Exit code

        Rule #4: Function <60 lines
        """
        storage = ctx.get("storage") if ctx else None
        if not storage:
            typer.echo("Error: No storage backend available", err=True)
            return 1

        # Load chunks
        typer.echo("Loading chunks from storage...")
        all_chunks = self._load_chunks(storage)
        if not all_chunks:
            typer.echo("No chunks found in storage", err=True)
            return 1

        typer.echo(f"Loaded {len(all_chunks)} chunks")

        # Extract relationships from all chunks
        extractor = SpacyRelationshipExtractor(use_spacy=True)
        all_relationships: List[Relationship] = []

        for chunk in all_chunks:
            text = (
                chunk.get("content", "")
                if isinstance(chunk, dict)
                else getattr(chunk, "content", "")
            )
            rels = extractor.extract(text)
            all_relationships.extend(rels)

        # Filter by type if specified
        if relationship_type:
            all_relationships = [
                r for r in all_relationships if r.predicate == relationship_type
            ]

        self._display_relationships(all_relationships)

        if output_json:
            self._save_json(all_relationships, output_json)

        return 0

    def _load_chunks(self, storage: Any) -> list:
        """Load chunks from storage.

        Args:
            storage: Storage backend

        Returns:
            List of chunks

        Rule #4: Function <60 lines
        """
        try:
            if hasattr(storage, "get_all_chunks"):
                return storage.get_all_chunks()
            results = storage.search_semantic("", limit=10000)
            return results.get("chunks", [])
        except Exception as e:
            logger.warning(f"Could not load chunks: {e}")
            return []

    def _display_relationships(self, relationships: List[Relationship]) -> None:
        """Display relationships.

        Args:
            relationships: List of relationships to display

        Rule #4: Function <60 lines
        """
        typer.echo(f"\nFound {len(relationships)} relationships")
        typer.echo("=" * 50)

        # Group by predicate
        by_type: Dict[str, List[Relationship]] = {}
        for rel in relationships:
            if rel.predicate not in by_type:
                by_type[rel.predicate] = []
            by_type[rel.predicate].append(rel)

        for pred_type, rels in sorted(by_type.items()):
            typer.echo(f"\n{pred_type.upper()} ({len(rels)})")
            for rel in rels[:10]:  # Limit display
                typer.echo(f"  {rel.subject} -> {rel.object}")
            if len(rels) > 10:
                typer.echo(f"  ... and {len(rels) - 10} more")

    def _save_json(self, relationships: List[Relationship], output_path: Path) -> None:
        """Save relationships to JSON.

        Args:
            relationships: List of relationships
            output_path: Output file path

        Rule #4: Function <60 lines
        """
        data = [rel.to_dict() for rel in relationships]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        typer.echo(f"\nResults saved to: {output_path}")


class KnowledgeGraphCommand(AnalyzeCommand):
    """Build and query knowledge graphs."""

    def execute(
        self,
        query_entity: Optional[str] = None,
        depth: int = 2,
        visualize: bool = False,
        output: Optional[Path] = None,
        output_json: Optional[Path] = None,
        input_file: Optional[Path] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Execute knowledge graph analysis.

        Args:
            query_entity: Entity to center query on
            depth: Query depth (hops)
            visualize: Generate Mermaid visualization
            output: Output file for Mermaid
            output_json: Output file for JSON
            input_file: Input text file
            ctx: Click context

        Returns:
            Exit code (0 = success)

        Rule #1: Early returns for routing
        Rule #4: Function <60 lines
        """
        try:
            # Build graph from input
            if input_file:
                builder = self._build_from_file(input_file)
            else:
                builder = self._build_from_corpus(ctx)

            if builder is None:
                return 1

            # Query if entity specified
            if query_entity:
                builder = builder.query(query_entity, depth)
                typer.echo(f"Queried subgraph around: {query_entity}")

            # Display or export
            return self._output_graph(builder, visualize, output, output_json)

        except Exception as e:
            logger.error(f"Knowledge graph analysis failed: {e}")
            typer.echo(f"Error: {e}", err=True)
            return 1

    def _build_from_file(self, input_file: Path) -> Optional[KnowledgeGraphBuilder]:
        """Build graph from input file.

        Args:
            input_file: Path to text file

        Returns:
            KnowledgeGraphBuilder or None on error

        Rule #4: Function <60 lines
        """
        if not input_file.exists():
            typer.echo(f"File not found: {input_file}", err=True)
            return None

        typer.echo(f"Building graph from: {input_file}")
        text = input_file.read_text(encoding="utf-8")

        return build_graph_from_text(text)

    def _build_from_corpus(
        self, ctx: Optional[Dict[str, Any]]
    ) -> Optional[KnowledgeGraphBuilder]:
        """Build graph from corpus.

        Args:
            ctx: Context with storage

        Returns:
            KnowledgeGraphBuilder or None on error

        Rule #4: Function <60 lines
        """
        storage = ctx.get("storage") if ctx else None
        if not storage:
            typer.echo("Error: No storage backend available", err=True)
            return None

        typer.echo("Loading chunks from storage...")

        try:
            if hasattr(storage, "get_all_chunks"):
                chunks = storage.get_all_chunks()
            else:
                results = storage.search_semantic("", limit=10000)
                chunks = results.get("chunks", [])
        except Exception as e:
            logger.warning(f"Could not load chunks: {e}")
            typer.echo(f"Error loading chunks: {e}", err=True)
            return None

        if not chunks:
            typer.echo("No chunks found in storage", err=True)
            return None

        typer.echo(f"Building graph from {len(chunks)} chunks...")

        # Build graph
        builder = KnowledgeGraphBuilder()
        ner = NERExtractor()
        rel_extractor = SpacyRelationshipExtractor(use_spacy=True)

        for chunk in chunks:
            text = (
                chunk.get("content", "")
                if isinstance(chunk, dict)
                else getattr(chunk, "content", "")
            )
            entities = ner.extract(text)
            relationships = rel_extractor.extract(text)

            builder.add_entities(entities)
            builder.add_relationships(relationships)

        return builder

    def _output_graph(
        self,
        builder: KnowledgeGraphBuilder,
        visualize: bool,
        output: Optional[Path],
        output_json: Optional[Path],
    ) -> int:
        """Output graph in requested format.

        Args:
            builder: Graph builder
            visualize: Generate Mermaid
            output: Mermaid output path
            output_json: JSON output path

        Returns:
            Exit code

        Rule #4: Function <60 lines
        """
        graph = builder.build()

        typer.echo("\nKnowledge Graph Statistics")
        typer.echo("=" * 40)
        typer.echo(f"Nodes: {graph.get_node_count()}")
        typer.echo(f"Edges: {graph.get_edge_count()}")

        # Mermaid output
        if visualize or output:
            mermaid = builder.to_mermaid()

            if output:
                export_to_mermaid_file(builder, output)
                typer.echo(f"\nMermaid diagram saved to: {output}")
            else:
                typer.echo("\nMermaid Diagram:")
                typer.echo("```mermaid")
                typer.echo(mermaid)
                typer.echo("```")

        # JSON output
        if output_json:
            data = builder.to_json()
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            typer.echo(f"\nJSON exported to: {output_json}")

        return 0


# =============================================================================
# CLI Commands
# =============================================================================


def relationships_command(
    relationship_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by relationship type"
    ),
    list_types: bool = typer.Option(
        False, "--list-types", help="List available relationship types"
    ),
    input_file: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input text file to analyze"
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output-json", "-o", help="Save results to JSON"
    ),
) -> None:
    """Analyze relationships between entities.

    Extract and display relationships from text or corpus.

    Examples:

        # List relationship types
        ingestforge analyze relationships --list-types

        # Analyze a file
        ingestforge analyze relationships --input document.txt

        # Filter by type
        ingestforge analyze relationships --type works_at

        # Export to JSON
        ingestforge analyze relationships --output-json rels.json
    """
    cmd = RelationshipAnalysisCommand()

    # Initialize context with storage if analyzing corpus
    ctx = None
    if not list_types and not input_file:
        ctx = cmd.initialize_context(require_storage=True)

    exit_code = cmd.execute(
        relationship_type=relationship_type,
        list_types=list_types,
        input_file=input_file,
        output_json=output_json,
        ctx=ctx,
    )
    raise typer.Exit(code=exit_code)


def knowledge_graph_command(
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Entity to query around"
    ),
    depth: int = typer.Option(2, "--depth", "-d", help="Query depth (number of hops)"),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="Generate Mermaid visualization"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for Mermaid diagram"
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output-json", help="Export graph to JSON"
    ),
    input_file: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input text file to analyze"
    ),
) -> None:
    """Build and query knowledge graphs.

    Create knowledge graphs from entities and relationships,
    query subgraphs, and export visualizations.

    Examples:

        # Build graph from file and visualize
        ingestforge analyze knowledge-graph --input doc.txt --visualize

        # Query around an entity
        ingestforge analyze knowledge-graph --query "Einstein" --depth 2

        # Export to Mermaid file
        ingestforge analyze knowledge-graph --output graph.md

        # Export to JSON
        ingestforge analyze knowledge-graph --output-json graph.json
    """
    cmd = KnowledgeGraphCommand()

    # Initialize context with storage if analyzing corpus (not a specific file)
    ctx = None
    if not input_file:
        ctx = cmd.initialize_context(require_storage=True)

    exit_code = cmd.execute(
        query_entity=query,
        depth=depth,
        visualize=visualize,
        output=output,
        output_json=output_json,
        input_file=input_file,
        ctx=ctx,
    )
    raise typer.Exit(code=exit_code)
