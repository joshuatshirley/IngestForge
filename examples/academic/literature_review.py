"""
Example: Literature Review Generator

Description:
    Analyzes multiple research papers and generates a comprehensive
    literature review with citation graphs, thematic clustering, and
    synthesis of findings across papers.

Usage:
    python examples/academic/literature_review.py \
        --papers-dir papers/ \
        --output review.md

Expected output:
    - Markdown literature review document
    - Citation graph JSON
    - Thematic clusters
    - Synthesis and recommendations

Requirements:
    - Multiple PDF research papers in a directory
    - Optional: LLM for synthesis (OPENAI_API_KEY)
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

from ingestforge.ingest.processor import DocumentProcessor
from ingestforge.chunking.semantic_chunker import SemanticChunker
from ingestforge.enrichment.entities import EntityExtractor


def analyze_papers(papers_dir: str) -> dict:
    """
    Analyze multiple papers to extract common themes.

    Args:
        papers_dir: Directory containing PDF papers

    Returns:
        Analysis dictionary with themes, citations, etc.
    """
    papers_dir = Path(papers_dir)

    if not papers_dir.exists():
        print(f"Error: Directory not found: {papers_dir}")
        return {}

    pdf_files = list(papers_dir.glob("**/*.pdf"))
    if not pdf_files:
        print(f"[!] No PDF files found in {papers_dir}")
        return {}

    print(f"\n[*] Analyzing {len(pdf_files)} paper(s)...")

    processor = DocumentProcessor()
    chunker = SemanticChunker(target_size=512, overlap=50)
    entity_extractor = EntityExtractor()

    papers = []
    all_entities = defaultdict(int)
    all_chunks = []

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"    [{i}/{len(pdf_files)}] {pdf_file.name}...", end="", flush=True)

        try:
            # Process paper
            text = processor.process(pdf_file)

            # Extract metadata
            title = extract_title(text)
            authors = extract_authors(text)

            # Chunk it
            chunks = chunker.chunk(text)

            # Extract entities
            all_paper_entities = set()
            for chunk in chunks:
                try:
                    entities = entity_extractor.extract(chunk["text"])
                    for entity in entities:
                        entity_text = entity.get("text", "")
                        if entity_text:
                            all_entities[entity_text] += 1
                            all_paper_entities.add(entity_text)
                except:
                    pass

            # Store paper info
            papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "path": str(pdf_file),
                    "filename": pdf_file.name,
                    "word_count": len(text.split()),
                    "chunks": len(chunks),
                    "entities": list(all_paper_entities),
                }
            )

            all_chunks.extend(chunks)
            print(" OK")

        except Exception as e:
            print(f" ERROR: {e}")

    # Analyze
    print("\n[*] Analyzing themes...")

    # Find common entities (themes)
    common_themes = sorted(
        [(k, v) for k, v in all_entities.items() if v > 1],
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    print("[✓] Analysis complete!")
    print(f"    Papers: {len(papers)}")
    print(f"    Chunks: {len(all_chunks)}")
    print(f"    Unique entities: {len(all_entities)}")
    print(f"    Common themes: {len(common_themes)}")

    return {
        "papers": papers,
        "chunks": all_chunks,
        "themes": common_themes,
        "entities": all_entities,
    }


def extract_title(text: str) -> str:
    """Extract likely title from text (usually first line)."""
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line and len(line) > 10 and len(line) < 200:
            return line
    return "Unknown Title"


def extract_authors(text: str) -> list[str]:
    """Extract likely authors from text."""
    # Simple heuristic: look for common author patterns
    import re

    # Look for patterns like "Smith, J." or "John Smith"
    pattern = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z]\.)*)"

    matches = re.findall(pattern, text[:500])  # Only first 500 chars
    authors = [
        m for m in matches if m not in ["Abstract", "Introduction", "Conclusion"]
    ]
    return authors[:5] if authors else []


def generate_literature_review(
    analysis: dict,
    output_path: str = "literature_review.md",
) -> None:
    """
    Generate markdown literature review.

    Args:
        analysis: Analysis dictionary from analyze_papers()
        output_path: Output markdown file
    """
    papers = analysis.get("papers", [])
    themes = analysis.get("themes", [])
    entities = analysis.get("entities", {})

    print("\n[*] Generating literature review...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build markdown
    lines = [
        "# Literature Review",
        "",
        f"**Generated from {len(papers)} papers**",
        "",
        "## Executive Summary",
        "",
        f"This review analyzes {len(papers)} research papers to identify common",
        f"themes, key concepts, and research directions. {len(themes)} significant",
        "themes were identified across the papers.",
        "",
        "## Papers Analyzed",
        "",
    ]

    # Papers section
    for i, paper in enumerate(papers, 1):
        lines.append(f"### {i}. {paper['title']}")
        lines.append("")

        if paper["authors"]:
            lines.append(f"**Authors:** {', '.join(paper['authors'])}")
        lines.append(f"**Word Count:** {paper['word_count']:,} words")
        lines.append(f"**Chunks:** {paper['chunks']}")
        lines.append("")

    # Themes section
    lines.extend(
        [
            "## Common Themes",
            "",
            "The following themes appear across multiple papers:",
            "",
        ]
    )

    for i, (theme, count) in enumerate(themes[:15], 1):
        lines.append(f"{i}. **{theme}** (mentioned in {count} papers)")

    lines.append("")

    # Key concepts section
    lines.extend(
        [
            "## Key Concepts and Entities",
            "",
            "Most frequently mentioned concepts:",
            "",
        ]
    )

    top_entities = sorted(
        [(k, v) for k, v in entities.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    for i, (entity, count) in enumerate(top_entities, 1):
        if len(entity) > 5:  # Skip very short terms
            lines.append(f"- {entity} ({count} mentions)")

    lines.append("")

    # Synthesis section
    lines.extend(
        [
            "## Synthesis and Observations",
            "",
            "Based on the analysis of the papers:",
            "",
            f"1. **Scope**: The {len(papers)} papers cover {len(set(e for e in entities.keys() if len(e) > 5))} unique concepts",
            f"2. **Common Themes**: {', '.join([t[0] for t in themes[:3]])} are the most frequently discussed topics",
            f"3. **Research Density**: Papers range from {min(p['word_count'] for p in papers):,} to {max(p['word_count'] for p in papers):,} words",
            "",
            "## Recommendations for Further Study",
            "",
            "Based on this literature review:",
            "",
            "1. Focus on the most frequently discussed themes",
            "2. Explore connections between papers with common entities",
            "3. Investigate gaps where themes are underrepresented",
            "4. Consider chronological analysis if publication dates are available",
            "",
        ]
    )

    # Write file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[✓] Review generated: {output_path}")


def generate_citation_graph(
    analysis: dict,
    output_path: str = "citation_graph.json",
) -> None:
    """
    Generate citation graph in JSON format.

    Args:
        analysis: Analysis dictionary
        output_path: Output JSON file
    """
    papers = analysis.get("papers", [])
    themes = analysis.get("themes", [])

    print("[*] Generating citation graph...")

    graph = {
        "nodes": [
            {
                "id": i,
                "label": paper["title"][:50],
                "title": paper["title"],
                "type": "paper",
            }
            for i, paper in enumerate(papers)
        ],
        "edges": [],
        "themes": [{"name": t[0], "frequency": t[1]} for t in themes[:10]],
    }

    # Add edges between papers (simplified: papers with common themes)
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            # Check if papers share entities
            paper_i_entities = set(papers[i]["entities"])
            paper_j_entities = set(papers[j]["entities"])

            shared = paper_i_entities & paper_j_entities
            if shared:
                graph["edges"].append(
                    {
                        "from": i,
                        "to": j,
                        "shared_entities": len(shared),
                    }
                )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    print(f"[✓] Citation graph saved: {output_path}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate literature review from multiple papers"
    )
    parser.add_argument(
        "--papers-dir",
        "-d",
        default="papers",
        help="Directory containing PDF papers (default: papers)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="literature_review.md",
        help="Output markdown file (default: literature_review.md)",
    )
    parser.add_argument("--graph", "-g", help="Save citation graph to JSON file")

    args = parser.parse_args()

    # Analyze papers
    analysis = analyze_papers(args.papers_dir)

    if not analysis.get("papers"):
        print("[!] No papers to analyze")
        return

    # Generate review
    generate_literature_review(analysis, args.output)

    # Generate graph if requested
    if args.graph:
        generate_citation_graph(analysis, args.graph)

    # Print summary
    print(f"\n{'='*70}")
    print("LITERATURE REVIEW SUMMARY")
    print(f"{'='*70}\n")

    print(f"Papers analyzed:     {len(analysis['papers'])}")
    print(f"Common themes:       {len(analysis['themes'])}")
    print(f"Total entities:      {len(analysis['entities'])}")
    print(f"Review file:         {args.output}")

    if args.graph:
        print(f"Citation graph:      {args.graph}")


if __name__ == "__main__":
    main()
