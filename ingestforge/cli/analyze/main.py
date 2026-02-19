"""Content analysis subcommands.

Provides tools for analyzing content patterns:
- topics: Extract and analyze main topics
- similarity: Find similar documents and duplicates
- entities: Named entity recognition and linking
- relationships: Relationship extraction between entities
- knowledge-graph: Build and query knowledge graphs
- timeline: Build security incident timelines (CYBER-004)
- contradictions: Detect contradicting claims (P3-AI-002.1)
- evidence: Link claims to supporting/refuting evidence (P3-AI-002.2)

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.analyze import topics, similarity, duplicates, entities
from ingestforge.cli.analyze import relationships, timeline, contradictions, evidence
from ingestforge.cli.analyze import connections

# Create analyze subcommand application
app = typer.Typer(
    name="analyze",
    help="Content analysis tools",
    add_completion=False,
)

# Register analysis commands
app.command("topics")(topics.command)
app.command("similarity")(similarity.command)
app.command("duplicates")(duplicates.command)
app.command("entities")(entities.command)
app.command("relationships")(relationships.relationships_command)
app.command("knowledge-graph")(relationships.knowledge_graph_command)
app.command("timeline")(timeline.command)
app.command("contradictions")(contradictions.command)
app.command("evidence")(evidence.command)
app.command("connections")(connections.command)


@app.callback()
def main() -> None:
    """Content analysis tools for IngestForge.

    Analyze patterns, topics, and relationships in your
    knowledge base content.

    Features:
    - Topic extraction and analysis
    - Similarity detection
    - Duplicate identification
    - Entity analysis and linking
    - Relationship extraction
    - Knowledge graph building
    - Security incident timelines (Cyber Vertical)
    - Contradiction detection (AI-powered)
    - Evidence linking for fact-checking
    - Lateral connections across domain silos

    Use cases:
    - Content organization
    - Quality assurance
    - Duplicate removal
    - Entity search and profiling
    - Topic modeling
    - Knowledge graph visualization
    - Security incident investigation
    - Fact-checking and verification
    - Cross-domain intelligence discovery

    Examples:
        # Analyze main topics
        ingestforge analyze topics --topics 20

        # Scan for lateral connections across domains
        ingestforge analyze connections

        # Find similar content
        ingestforge analyze similarity --threshold 0.8

        # Search for entities
        ingestforge analyze entities "Microsoft" --related

        # List all entities
        ingestforge analyze entities --list

        # Analyze relationships
        ingestforge analyze relationships --type works_at

        # Build knowledge graph
        ingestforge analyze knowledge-graph --visualize

        # Query knowledge graph
        ingestforge analyze knowledge-graph --query "Einstein" --depth 2

        # Export to Mermaid
        ingestforge analyze knowledge-graph --output graph.md

        # Build security timeline
        ingestforge analyze timeline --start 2024-01-01 --format md

        # Timeline with correlations
        ingestforge analyze timeline --correlate --format json

        # Detect contradictions in content about a topic
        ingestforge analyze contradictions "climate change"

        # Find evidence for/against a claim
        ingestforge analyze evidence "The Earth is round"

    For help on specific commands:
        ingestforge analyze <command> --help
    """
    pass
