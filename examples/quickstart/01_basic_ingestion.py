"""
Example: Basic Document Ingestion

Description:
    Demonstrates how to load a PDF document, extract text, chunk it,
    and print the chunks with metadata. This is the simplest way to
    get started with IngestForge.

Usage:
    python examples/quickstart/01_basic_ingestion.py path/to/document.pdf

Expected output:
    - Number of chunks extracted
    - Chunk text samples
    - Metadata for each chunk (page numbers, word count, etc.)

Requirements:
    - pip install pdfplumber (for PDF processing)
"""

from __future__ import annotations

from pathlib import Path
import json

from ingestforge.ingest.processor import DocumentProcessor
from ingestforge.chunking.semantic_chunker import SemanticChunker


def ingest_document(
    file_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """
    Ingest a document and print extracted chunks.

    Args:
        file_path: Path to the document (PDF, TXT, DOCX, etc.)
        chunk_size: Target size for each chunk
        chunk_overlap: Overlap between chunks for context
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    print(f"\n[*] Ingesting: {file_path}")
    print(f"    Size: {file_path.stat().st_size / (1024*1024):.2f} MB")

    try:
        # Step 1: Process document
        print("\n[*] Processing document...")
        processor = DocumentProcessor()
        text = processor.process(file_path)

        print(f"    Extracted {len(text)} characters")
        print(f"    Approximately {len(text.split())} words")

        # Step 2: Chunk the text
        print("\n[*] Chunking text...")
        chunker = SemanticChunker(
            target_size=chunk_size,
            overlap=chunk_overlap,
        )
        chunks = chunker.chunk(text)

        print(f"    Created {len(chunks)} chunks")

        # Step 3: Print chunks
        print(f"\n{'='*70}")
        print("CHUNKS (showing first 5)")
        print(f"{'='*70}\n")

        for i, chunk in enumerate(chunks[:5], 1):
            print(f"[Chunk {i}]")
            print(f"Size: {len(chunk['text'])} chars")

            # Print first 200 characters
            preview = chunk["text"][:200].replace("\n", " ")
            if len(chunk["text"]) > 200:
                preview += "..."
            print(f"Text: {preview}")

            if "metadata" in chunk:
                metadata = chunk["metadata"]
                print(f"Metadata: {json.dumps(metadata, indent=2)}")

            print()

        # Step 4: Summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}\n")

        chunk_sizes = [len(c["text"]) for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunks else 0

        print(f"Total chunks:        {len(chunks)}")
        print(f"Average chunk size:  {avg_size:.0f} characters")
        print(f"Min chunk size:      {min(chunk_sizes) if chunks else 0} characters")
        print(f"Max chunk size:      {max(chunk_sizes) if chunks else 0} characters")
        print(f"Total text length:   {sum(chunk_sizes)} characters")

        print("\n[✓] Ingestion complete!")

    except Exception as e:
        print(f"\n[!] Error during ingestion: {e}")
        import traceback

        traceback.print_exc()


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a document and extract chunks")
    parser.add_argument(
        "file",
        nargs="?",
        default="examples/data/sample.pdf",
        help="Path to document file (default: examples/data/sample.pdf)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in characters (default: 512)",
    )
    parser.add_argument(
        "--overlap", type=int, default=50, help="Overlap between chunks (default: 50)"
    )

    args = parser.parse_args()

    ingest_document(
        args.file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
