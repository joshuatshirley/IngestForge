"""Knowledge Graph export command.

Exports knowledge graph visualization from chunk metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Set
import typer

from ingestforge.cli.export.base import ExportCommand
from ingestforge.viz.graph_export import (
    GraphData,
    GraphNode,
    GraphEdge,
    GraphExporter,
    NodeType,
    EdgeType,
)
from ingestforge.viz.d3_renderer import (
    D3Renderer,
    render_with_template,
    open_in_browser,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_NODES = 500
MAX_EDGES = 2000
MAX_CONCEPTS_PER_CHUNK = 20
MAX_ENTITIES_PER_CHUNK = 20


class KnowledgeGraphExportCommand(ExportCommand):
    """Export knowledge graph visualization from chunks."""

    def execute(
        self,
        output: Path,
        project: Optional[Path] = None,
        query: Optional[str] = None,
        format: str = "json",
        max_nodes: int = 500,
        include_relationships: bool = True,
        use_template: bool = False,
        open_browser: bool = False,
    ) -> int:
        """Export knowledge graph visualization.

        Args:
            output: Output file path
            project: Project directory
            query: Optional search query to filter chunks
            format: Output format (json or html)
            max_nodes: Maximum nodes to include
            include_relationships: Include relationship edges
            use_template: Use enhanced D3 template
            open_browser: Open result in browser

        Returns:
            0 on success, 1 on error
        """
        try:
            self._validate_parameters(output, format, max_nodes)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Retrieve chunks
            chunks = self.search_filtered_chunks(ctx["storage"], query, limit=None)
            if not chunks:
                self._handle_no_chunks(query)
                return 0

            # Build graph from chunks
            graph_data = self._build_graph_from_chunks(
                chunks, max_nodes, include_relationships
            )

            # Export based on format
            if use_template and format == "html":
                success = self._export_with_template(graph_data, output)
            else:
                success = self._export_graph(graph_data, output, format)

            if success:
                self._display_summary(graph_data, output, format)

                # Open in browser if requested
                if open_browser and format == "html":
                    open_in_browser(output)

                return 0
            else:
                self.print_error("Failed to export graph")
                return 1

        except Exception as e:
            return self.handle_error(e, "Knowledge graph export failed")

    def _validate_parameters(self, output: Path, format: str, max_nodes: int) -> None:
        """Validate command parameters.

        Args:
            output: Output file path
            format: Output format
            max_nodes: Maximum nodes

        Raises:
            typer.BadParameter: If invalid
        """
        self.validate_output_path(output)

        if format not in ["json", "html"]:
            raise typer.BadParameter(
                f"Invalid format: {format}. Must be 'json' or 'html'"
            )

        if max_nodes < 1 or max_nodes > MAX_NODES:
            raise typer.BadParameter(f"max_nodes must be between 1 and {MAX_NODES}")

    def _handle_no_chunks(self, query: Optional[str]) -> None:
        """Handle case where no chunks found.

        Args:
            query: Optional search query
        """
        if query:
            self.print_warning(f"No chunks found matching: '{query}'")
        else:
            self.print_warning("Knowledge base is empty")

        self.print_info("Try ingesting some documents first")

    def _build_graph_from_chunks(
        self,
        chunks: List[Any],
        max_nodes: int,
        include_relationships: bool,
    ) -> GraphData:
        """Build graph data from chunks.

        Args:
            chunks: List of chunks
            max_nodes: Maximum nodes to include
            include_relationships: Include relationship edges

        Returns:
            GraphData structure
        """
        graph_data = GraphData(
            title="Knowledge Graph", description=f"Extracted from {len(chunks)} chunks"
        )

        # Track nodes by ID to avoid duplicates
        node_ids: Set[str] = set()
        document_nodes: Dict[str, str] = {}
        if not chunks:
            return graph_data

        # Extract nodes and edges from chunks
        for chunk in chunks[:max_nodes]:
            if len(graph_data.nodes) >= max_nodes:
                break

            self._process_chunk_graph(
                chunk,
                graph_data,
                node_ids,
                document_nodes,
                include_relationships,
            )

        return graph_data

    def _process_chunk_graph(
        self,
        chunk: Any,
        graph_data: GraphData,
        node_ids: Set[str],
        document_nodes: Dict[str, str],
        include_relationships: bool,
    ) -> None:
        """Process a single chunk into graph nodes/edges.

        Args:
            chunk: Chunk to process
            graph_data: GraphData to populate
            node_ids: Set of existing node IDs
            document_nodes: Map of document IDs to node IDs
            include_relationships: Include relationship edges
        """
        metadata = self.extract_chunk_metadata(chunk)

        # Get document info
        doc_id = metadata.get("source", "unknown")
        chunk_id = getattr(chunk, "chunk_id", str(id(chunk)))

        # Add document node if not exists
        if doc_id not in document_nodes:
            self._add_document_node(doc_id, graph_data, node_ids, document_nodes)

        # Extract and add entity nodes
        entities = metadata.get("entities", [])
        self._add_entity_nodes(
            entities,
            chunk_id,
            doc_id,
            graph_data,
            node_ids,
            document_nodes,
            include_relationships,
        )

        # Extract and add concept nodes
        concepts = metadata.get("concepts", [])
        self._add_concept_nodes(
            concepts,
            chunk_id,
            doc_id,
            graph_data,
            node_ids,
            document_nodes,
            include_relationships,
        )

    def _add_document_node(
        self,
        doc_id: str,
        graph_data: GraphData,
        node_ids: Set[str],
        document_nodes: Dict[str, str],
    ) -> None:
        """Add document node to graph.

        Args:
            doc_id: Document ID
            graph_data: GraphData to populate
            node_ids: Set of existing node IDs
            document_nodes: Map of document IDs to node IDs
        """
        if doc_id in node_ids:
            return

        node_id = f"doc_{doc_id}"
        label = Path(doc_id).stem if doc_id else "Document"

        graph_data.nodes.append(
            GraphNode(
                id=node_id,
                label=label,
                node_type=NodeType.DOCUMENT,
                size=1.5,
                group=0,
            )
        )

        node_ids.add(node_id)
        document_nodes[doc_id] = node_id

    def _should_add_edge(
        self,
        graph_data: GraphData,
        include_relationships: bool,
        doc_id: str,
        document_nodes: Dict[str, str],
    ) -> bool:
        """Check if edge should be added.

        Args:
            graph_data: GraphData to check
            include_relationships: Whether relationships enabled
            doc_id: Document ID
            document_nodes: Map of document nodes

        Returns:
            True if edge should be added
        """
        if not include_relationships:
            return False
        if doc_id not in document_nodes:
            return False
        if len(graph_data.edges) >= MAX_EDGES:
            return False
        return True

    def _add_edge_to_graph(
        self,
        graph_data: GraphData,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
    ) -> None:
        """Add edge to graph.

        Args:
            graph_data: GraphData to populate
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge
        """
        graph_data.edges.append(
            GraphEdge(
                source=source_id,
                target=target_id,
                edge_type=edge_type,
                weight=1.0,
            )
        )

    def _add_entity_nodes(
        self,
        entities: List[str],
        chunk_id: str,
        doc_id: str,
        graph_data: GraphData,
        node_ids: Set[str],
        document_nodes: Dict[str, str],
        include_relationships: bool,
    ) -> None:
        """Add entity nodes and edges.

        Args:
            entities: List of entity strings
            chunk_id: Source chunk ID
            doc_id: Source document ID
            graph_data: GraphData to populate
            node_ids: Set of existing node IDs
            document_nodes: Map of document IDs to node IDs
            include_relationships: Include relationship edges
        """
        for entity in entities[:MAX_ENTITIES_PER_CHUNK]:
            if len(graph_data.nodes) >= MAX_NODES:
                break

            entity_id = f"entity_{entity}"

            # Add entity node if not exists
            if entity_id not in node_ids:
                graph_data.nodes.append(
                    GraphNode(
                        id=entity_id,
                        label=str(entity),
                        node_type=NodeType.ENTITY,
                        size=1.0,
                        group=2,
                    )
                )
                node_ids.add(entity_id)
            if self._should_add_edge(
                graph_data, include_relationships, doc_id, document_nodes
            ):
                self._add_edge_to_graph(
                    graph_data,
                    document_nodes[doc_id],
                    entity_id,
                    EdgeType.CONTAINS,
                )

    def _add_concept_nodes(
        self,
        concepts: List[str],
        chunk_id: str,
        doc_id: str,
        graph_data: GraphData,
        node_ids: Set[str],
        document_nodes: Dict[str, str],
        include_relationships: bool,
    ) -> None:
        """Add concept nodes and edges.

        Args:
            concepts: List of concept strings
            chunk_id: Source chunk ID
            doc_id: Source document ID
            graph_data: GraphData to populate
            node_ids: Set of existing node IDs
            document_nodes: Map of document IDs to node IDs
            include_relationships: Include relationship edges
        """
        for concept in concepts[:MAX_CONCEPTS_PER_CHUNK]:
            if len(graph_data.nodes) >= MAX_NODES:
                break

            concept_id = f"concept_{concept}"

            # Add concept node if not exists
            if concept_id not in node_ids:
                graph_data.nodes.append(
                    GraphNode(
                        id=concept_id,
                        label=str(concept),
                        node_type=NodeType.CONCEPT,
                        size=1.0,
                        group=1,
                    )
                )
                node_ids.add(concept_id)
            if self._should_add_edge(
                graph_data, include_relationships, doc_id, document_nodes
            ):
                self._add_edge_to_graph(
                    graph_data,
                    document_nodes[doc_id],
                    concept_id,
                    EdgeType.CONTAINS,
                )

    def _export_graph(
        self,
        graph_data: GraphData,
        output: Path,
        format: str,
    ) -> bool:
        """Export graph to file.

        Args:
            graph_data: Graph data to export
            output: Output file path
            format: Output format

        Returns:
            True if successful
        """
        if format == "json":
            return self._export_json(graph_data, output)
        elif format == "html":
            return self._export_html(graph_data, output)
        else:
            logger.error(f"Unknown format: {format}")
            return False

    def _export_json(self, graph_data: GraphData, output: Path) -> bool:
        """Export as JSON.

        Args:
            graph_data: Graph data
            output: Output path

        Returns:
            True if successful
        """
        exporter = GraphExporter()
        return exporter.to_file(graph_data, output)

    def _export_html(self, graph_data: GraphData, output: Path) -> bool:
        """Export as HTML visualization.

        Args:
            graph_data: Graph data
            output: Output path

        Returns:
            True if successful
        """
        renderer = D3Renderer()
        return renderer.render(graph_data, output)

    def _export_with_template(self, graph_data: GraphData, output: Path) -> bool:
        """Export using enhanced D3 template.

        Args:
            graph_data: Graph data
            output: Output path

        Returns:
            True if successful
        """
        return render_with_template(graph_data, output)

    def _display_summary(
        self,
        graph_data: GraphData,
        output: Path,
        format: str,
    ) -> None:
        """Display export summary.

        Args:
            graph_data: Exported graph data
            output: Output file path
            format: Output format
        """
        self.console.print()
        self.print_success(f"Knowledge graph exported to: {output}")
        self.print_info(f"Format: {format.upper()}")
        self.print_info(f"Nodes: {graph_data.node_count}")
        self.print_info(f"Edges: {graph_data.edge_count}")

        # Show file size
        try:
            file_size = output.stat().st_size
            size_kb = file_size / 1024
            self.print_info(f"File size: {size_kb:.2f} KB")
        except Exception as e:
            logger.debug(f"Failed to get file size: {e}")


# Typer command wrapper
def command(
    output: Path = typer.Argument(..., help="Output file path"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Filter chunks by search query"
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Output format: json or html"
    ),
    max_nodes: int = typer.Option(500, "--max-nodes", help="Maximum number of nodes"),
    no_relationships: bool = typer.Option(
        False, "--no-relationships", help="Exclude relationship edges"
    ),
    enhanced: bool = typer.Option(
        False,
        "--enhanced",
        "-e",
        help="Use enhanced dark-themed D3 template with search/filter",
    ),
    open_browser: bool = typer.Option(
        False, "--open", "-o", help="Open the graph in browser after export (HTML only)"
    ),
) -> None:
    """Export knowledge graph visualization.

    Creates an interactive knowledge graph from your chunks,
    visualizing entities, concepts, and their relationships.

    Formats:
    - json: D3-compatible JSON for custom visualization
    - html: Interactive browser-based visualization

    The graph extracts:
    - Document nodes (from chunk sources)
    - Entity nodes (from chunk metadata)
    - Concept nodes (from chunk metadata)
    - Relationship edges (CONTAINS, MENTIONS)

    Examples:
        # Export as interactive HTML
        ingestforge export knowledge-graph graph.html --format html

        # Export with enhanced D3 template (dark theme)
        ingestforge export knowledge-graph graph.html -f html --enhanced

        # Export and open in browser
        ingestforge export knowledge-graph graph.html -f html --open

        # Export as JSON data
        ingestforge export knowledge-graph graph.json --format json

        # Export with query filter
        ingestforge export knowledge-graph ml_graph.html -q "machine learning"

        # Limit graph size
        ingestforge export knowledge-graph graph.html --max-nodes 100

        # Export without relationship edges
        ingestforge export knowledge-graph graph.json --no-relationships

        # Specific project
        ingestforge export knowledge-graph graph.html -p /path/to/project
    """
    cmd = KnowledgeGraphExportCommand()
    include_relationships = not no_relationships
    exit_code = cmd.execute(
        output,
        project,
        query,
        format,
        max_nodes,
        include_relationships,
        use_template=enhanced,
        open_browser=open_browser,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
