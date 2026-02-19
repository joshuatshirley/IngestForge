"""
Example: ArXiv Research Assistant

Description:
    Demonstrates searching ArXiv for research papers, downloading them,
    ingesting them into IngestForge, and performing analysis on them.
    Perfect for literature reviews and research project kickoff.

Usage:
    python examples/academic/arxiv_research_assistant.py \
        --query "transformer architecture" \
        --max-results 10

    python examples/academic/arxiv_research_assistant.py \
        --query "attention mechanisms" \
        --max-results 20 \
        --output papers.db

Expected output:
    - Downloaded PDF files
    - Indexed knowledge base
    - Search results with citations
    - Bibliography export

Requirements:
    - pip install arxiv requests
    - ArXiv is free and requires no API key
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class Paper:
    """Represents a research paper."""

    title: str
    authors: list[str]
    url: str
    published: str
    summary: str
    arxiv_id: str
    local_path: Optional[Path] = None


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
) -> list[Paper]:
    """
    Search ArXiv for papers.

    Args:
        query: Search query
        max_results: Maximum number of results
        sort_by: Sort order (relevance, lastUpdatedDate, submittedDate)

    Returns:
        List of Paper objects
    """
    try:
        import arxiv
    except ImportError:
        print("[!] arxiv library not installed")
        print("    Run: pip install arxiv")
        return []

    print(f"\n[*] Searching ArXiv for: {query}")
    print(f"    Max results: {max_results}")

    papers = []
    client = arxiv.Client()

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for result in client.results(search):
            paper = Paper(
                title=result.title,
                authors=[author.name for author in result.authors],
                url=result.entry_id,
                published=result.published.isoformat(),
                summary=result.summary,
                arxiv_id=result.arxiv_id,
            )
            papers.append(paper)

        print(f"[✓] Found {len(papers)} papers")
        return papers

    except Exception as e:
        print(f"[!] Error searching ArXiv: {e}")
        return []


def download_papers(
    papers: list[Paper],
    output_dir: str = "papers",
) -> list[Paper]:
    """
    Download papers from ArXiv.

    Args:
        papers: List of Paper objects
        output_dir: Where to save PDFs

    Returns:
        Updated list with local paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Downloading {len(papers)} papers...")
    print(f"    Output: {output_dir}")

    try:
        import arxiv
    except ImportError:
        print("[!] arxiv library not installed")
        return papers

    client = arxiv.Client()

    for i, paper in enumerate(papers, 1):
        print(f"    [{i}/{len(papers)}] {paper.title[:60]}...", end="", flush=True)

        try:
            pdf_path = output_dir / f"{paper.arxiv_id.split('/')[1]}.pdf"

            if pdf_path.exists():
                print(" (cached)")
                paper.local_path = pdf_path
            else:
                client.download_pdf(
                    next(arxiv.Search(arxiv_id=paper.arxiv_id).results()),
                    dirpath=str(output_dir),
                )
                paper.local_path = pdf_path
                print(" OK")

        except Exception as e:
            print(f" ERROR: {e}")

    return papers


def index_papers(
    papers: list[Paper],
    output_db: str = "papers.jsonl",
) -> None:
    """
    Index papers into knowledge base.

    Args:
        papers: List of Paper objects
        output_db: Path to save index
    """
    from ingestforge.ingest.processor import DocumentProcessor
    from ingestforge.chunking.semantic_chunker import SemanticChunker
    from ingestforge.storage.jsonl import JSONLStorage

    print("\n[*] Indexing papers...")
    processor = DocumentProcessor()
    chunker = SemanticChunker(target_size=512, overlap=50)
    storage = JSONLStorage(output_db)

    all_chunks = []

    for i, paper in enumerate(papers, 1):
        if not paper.local_path or not paper.local_path.exists():
            continue

        print(f"    [{i}/{len(papers)}] {paper.title[:60]}...", end="", flush=True)

        try:
            # Process paper
            text = processor.process(paper.local_path)

            # Chunk it
            chunks = chunker.chunk(text)

            # Add metadata
            for chunk in chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}

                chunk["metadata"].update(
                    {
                        "title": paper.title,
                        "authors": paper.authors,
                        "arxiv_id": paper.arxiv_id,
                        "published": paper.published,
                        "source": paper.local_path.name,
                    }
                )

            all_chunks.extend(chunks)
            print(f" OK ({len(chunks)} chunks)")

        except Exception as e:
            print(f" ERROR: {e}")

    # Save index
    print(f"\n[*] Saving {len(all_chunks)} chunks to {output_db}...")
    storage.save(all_chunks)

    print("[✓] Indexing complete!")


def print_papers(papers: list[Paper]) -> None:
    """Print paper summary."""
    print(f"\n{'='*70}")
    print(f"PAPERS FOUND ({len(papers)})")
    print(f"{'='*70}\n")

    for i, paper in enumerate(papers, 1):
        print(f"[{i}] {paper.title}")
        print(f"    Authors: {', '.join(paper.authors[:3])}")
        if len(paper.authors) > 3:
            print(f"            and {len(paper.authors) - 3} more")

        print(f"    Published: {paper.published}")
        print(f"    ArXiv ID: {paper.arxiv_id}")
        print(f"    URL: {paper.url}")

        # Print first 200 chars of summary
        summary = paper.summary.replace("\n", " ")[:200]
        if len(paper.summary) > 200:
            summary += "..."
        print(f"    Summary: {summary}")
        print()


def save_bibliography(papers: list[Paper], output_path: str) -> None:
    """
    Save bibliography in BibTeX format.

    Args:
        papers: List of papers
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Saving bibliography to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for paper in papers:
            # Convert ArXiv ID to BibTeX
            arxiv_num = paper.arxiv_id.split("/")[-1].replace(".", "")

            bibtex = f"""@article{{{arxiv_num},
  title={{{paper.title}}},
  author={{{' and '.join(paper.authors)}}},
  journal={{arXiv preprint arXiv:{paper.arxiv_id}}},
  year={{{paper.published[:4]}}},
  url={{{paper.url}}}
}}

"""
            f.write(bibtex)

    print("[✓] Bibliography saved!")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Search and analyze research papers from ArXiv"
    )
    parser.add_argument(
        "--query", required=True, help="Search query (e.g. 'transformer architecture')"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum papers to retrieve (default: 10)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="papers.jsonl",
        help="Output knowledge base path (default: papers.jsonl)",
    )
    parser.add_argument(
        "--papers-dir",
        default="papers",
        help="Directory to save PDFs (default: papers)",
    )
    parser.add_argument(
        "--bibliography", "-b", help="Save bibliography to file (BibTeX format)"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download papers (default: search only)"
    )
    parser.add_argument(
        "--index", action="store_true", help="Index papers into knowledge base"
    )

    args = parser.parse_args()

    # Search papers
    papers = search_arxiv(args.query, max_results=args.max_results)

    if not papers:
        print("[!] No papers found")
        return

    # Print results
    print_papers(papers)

    # Download if requested
    if args.download:
        papers = download_papers(papers, args.papers_dir)

    # Index if requested
    if args.index or args.download:
        index_papers(papers, args.output)

    # Save bibliography if requested
    if args.bibliography:
        save_bibliography(papers, args.bibliography)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    print(f"Papers found:        {len(papers)}")

    if args.download:
        downloaded = sum(1 for p in papers if p.local_path)
        print(f"Papers downloaded:   {downloaded}/{len(papers)}")

    if args.index:
        print(f"Knowledge base:      {args.output}")

    if args.bibliography:
        print(f"Bibliography:        {args.bibliography}")


if __name__ == "__main__":
    main()
