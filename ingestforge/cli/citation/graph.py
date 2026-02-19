"""Citation graph command - Visualize citation networks."""

from pathlib import Path
from typing import Optional, Dict, Any, List
import typer
from rich.tree import Tree
from ingestforge.cli.citation.base import CitationCommand


class GraphCommand(CitationCommand):
    """Generate citation network graphs."""

    def execute(
        self,
        topic: Optional[str] = None,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """Generate citation graph."""
        try:
            ctx = self.initialize_context(project, require_storage=True)

            # Get all chunks
            all_chunks = self.get_all_chunks_from_storage(ctx["storage"])

            if not all_chunks:
                self.print_warning("No content in knowledge base")
                return 0

            # Extract citations
            citations = self._extract_citations(all_chunks)

            if not citations:
                self.print_warning("No citations found")
                return 0

            # Build citation network
            network = self._build_network(citations)

            # Display graph
            self._display_graph(network)

            # Save if requested
            if output:
                self.save_json_output(
                    output, network, f"Citation graph saved to: {output}"
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Citation graph generation failed")

    def _extract_citations(self, chunks: list) -> List[Dict[str, Any]]:
        """Extract citations from chunks."""
        citations = []

        for chunk in chunks:
            metadata = self.extract_chunk_metadata(chunk)
            source = metadata.get("source", "Unknown")

            # Check for citation metadata
            if "citations" in metadata:
                for cite in metadata["citations"]:
                    citations.append(
                        {"source": source, "target": cite, "type": "cites"}
                    )

        return citations

    def _build_network(self, citations: List[Dict]) -> Dict:
        """Build citation network."""
        nodes = set()
        edges = []

        for cite in citations:
            nodes.add(cite["source"])
            nodes.add(cite["target"])
            edges.append(cite)

        return {
            "nodes": list(nodes),
            "edges": edges,
            "stats": {"total_nodes": len(nodes), "total_edges": len(edges)},
        }

    def _display_graph(self, network: Dict) -> None:
        """Display citation graph."""
        self.console.print()
        self.console.print("[bold cyan]Citation Network[/bold cyan]\n")

        stats = network.get("stats", {})
        self.print_info(f"Nodes: {stats.get('total_nodes', 0)}")
        self.print_info(f"Citations: {stats.get('total_edges', 0)}")

        # Display as tree (simplified visualization)
        nodes = network.get("nodes", [])
        edges = network.get("edges", [])

        if nodes:
            self.console.print()
            tree = Tree("[bold]Citation Network[/bold]")

            # Group by source
            by_source = {}
            for edge in edges:
                source = edge["source"]
                target = edge["target"]

                if source not in by_source:
                    by_source[source] = []

                by_source[source].append(target)

            # Add to tree
            for source in list(by_source.keys())[:10]:  # Limit display
                branch = tree.add(f"[yellow]{source}[/yellow]")
                for target in by_source[source][:5]:  # Limit citations per source
                    branch.add(f"[cyan]→ {target}[/cyan]")

            self.console.print(tree)


def command(
    topic: Optional[str] = typer.Argument(None, help="Optional topic to focus on"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
) -> None:
    """Generate citation network graph.

    Examples:
        ingestforge citation graph
        ingestforge citation graph "machine learning"
        ingestforge citation graph -o network.json
    """
    cmd = GraphCommand()
    exit_code = cmd.execute(topic, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
