"""Transform subcommands.

Provides tools for data transformation and processing:
- split: Split documents into chunks / partition collections
- merge: Merge and deduplicate chunks
- filter: Filter chunks by criteria
- clean: Clean and normalize text
- enrich: Enrich with metadata

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.transform import split, merge, clean, enrich, filter

# Create transform subcommand application
app = typer.Typer(
    name="transform",
    help="Data transformation tools",
    add_completion=False,
)

# Register transform commands
app.command("split")(split.command)
app.command("merge")(merge.command)
app.command("filter")(filter.command)
app.command("clean")(clean.command)
app.command("enrich")(enrich.command)


@app.callback()
def main() -> None:
    """Data transformation tools for IngestForge.

    Transform and process documents for RAG pipelines:
    - Split documents into chunks or partition collections
    - Merge and deduplicate chunks
    - Filter chunks by various criteria
    - Clean and normalize text
    - Enrich with metadata

    Features:
    - Configurable chunk sizes and partitioning
    - Deduplication (exact, fuzzy, semantic)
    - Flexible filtering (length, entity, topic)
    - Text cleaning and normalization
    - Metadata extraction

    Use cases:
    - Prepare documents for embedding
    - Clean scraped web content
    - Split large documents
    - Merge fragmented content
    - Deduplicate overlapping collections
    - Filter chunks by quality or topic

    Examples:
        # Split document into chunks
        ingestforge transform split document.txt -c 1000 -o chunks.json

        # Partition by document
        ingestforge transform split docs --by document -o partitions/

        # Merge and deduplicate libraries
        ingestforge transform merge lib1 lib2 --deduplicate -o merged.jsonl

        # Filter by criteria
        ingestforge transform filter docs --min-length 100 --has-entities -o filtered.json

        # Clean text
        ingestforge transform clean scraped.html --remove-urls -o clean.txt

        # Enrich with metadata
        ingestforge transform enrich document.txt -o enriched.json

    For help on specific commands:
        ingestforge transform <command> --help
    """
    pass
