"""
Example: Simple Document Search

Description:
    Demonstrates storing documents in a knowledge base and performing
    semantic search queries. Shows how to build a basic RAG system
    for question-answering.

Usage:
    python examples/quickstart/02_simple_search.py

    Then interactively query:
    > What is the main topic?
    > Tell me about...
    > exit

Expected output:
    - Interactive search prompts
    - Ranked retrieval results
    - Similarity scores

Requirements:
    - Embeddings model (downloads automatically)
    - Sample documents
"""

from __future__ import annotations

from pathlib import Path

from ingestforge.ingest.processor import DocumentProcessor
from ingestforge.chunking.semantic_chunker import SemanticChunker
from ingestforge.retrieval.semantic import SemanticRetriever
from ingestforge.storage.jsonl import JSONLStorage


def setup_knowledge_base(
    documents_dir: str,
    db_path: str = "example_kb.jsonl",
) -> JSONLStorage:
    """
    Create a knowledge base from documents.

    Args:
        documents_dir: Directory containing documents
        db_path: Where to save the knowledge base

    Returns:
        JSONLStorage instance with indexed documents
    """
    documents_dir = Path(documents_dir)
    db_path = Path(db_path)

    print("\n[*] Setting up knowledge base")
    print(f"    Documents: {documents_dir}")
    print(f"    Database: {db_path}")

    # Create storage
    storage = JSONLStorage(str(db_path))

    # Collect documents
    doc_files = (
        list(documents_dir.glob("**/*.txt"))
        + list(documents_dir.glob("**/*.pdf"))
        + list(documents_dir.glob("**/*.md"))
    )

    if not doc_files:
        print(f"\n[!] No documents found in {documents_dir}")
        return storage

    print(f"\n[*] Processing {len(doc_files)} document(s)...")

    processor = DocumentProcessor()
    chunker = SemanticChunker(target_size=512, overlap=50)

    all_chunks = []

    for i, doc_file in enumerate(doc_files, 1):
        try:
            print(f"    [{i}/{len(doc_files)}] {doc_file.name}...", end="", flush=True)

            # Process document
            text = processor.process(doc_file)

            # Chunk it
            chunks = chunker.chunk(text)

            # Add source metadata
            for chunk in chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                chunk["metadata"]["source"] = str(doc_file)
                chunk["metadata"]["source_name"] = doc_file.name

            all_chunks.extend(chunks)
            print(f" OK ({len(chunks)} chunks)")

        except Exception as e:
            print(f" ERROR: {e}")

    # Store all chunks
    print(f"\n[*] Storing {len(all_chunks)} chunks...")
    storage.save(all_chunks)

    print("[✓] Knowledge base ready!")
    return storage


def search_knowledge_base(
    storage: JSONLStorage,
    query: str,
    k: int = 5,
) -> None:
    """
    Search the knowledge base and print results.

    Args:
        storage: JSONLStorage instance
        query: Search query
        k: Number of results to return
    """
    print(f"\n[*] Searching for: {query}")

    # Load chunks
    chunks = storage.load()
    if not chunks:
        print("[!] Knowledge base is empty")
        return

    # Create retriever and search
    retriever = SemanticRetriever(chunks)
    results = retriever.retrieve(query, k=k)

    if not results:
        print("[!] No results found")
        return

    print(f"\n[*] Found {len(results)} result(s)\n")

    for i, result in enumerate(results, 1):
        print(f"[Result {i}] (Score: {result.get('score', 0):.3f})")

        text = result.get("text", "")[:300]
        if len(result.get("text", "")) > 300:
            text += "..."

        print(f"Text: {text}")

        metadata = result.get("metadata", {})
        if "source" in metadata:
            print(f"Source: {metadata['source']}")

        print()


def interactive_search(storage: JSONLStorage) -> None:
    """
    Interactive search mode.

    Args:
        storage: JSONLStorage instance
    """
    print("\n" + "=" * 70)
    print("INTERACTIVE SEARCH")
    print("=" * 70)
    print("\nType your questions below. Type 'exit' or 'quit' to exit.\n")

    while True:
        try:
            query = input("Query> ").strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit"):
                print("\n[*] Goodbye!")
                break

            search_knowledge_base(storage, query, k=3)

        except KeyboardInterrupt:
            print("\n\n[*] Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[!] Error: {e}\n")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple document search example")
    parser.add_argument(
        "--docs-dir",
        default="examples/data",
        help="Directory containing documents (default: examples/data)",
    )
    parser.add_argument(
        "--db-path",
        default="example_kb.jsonl",
        help="Path to knowledge base (default: example_kb.jsonl)",
    )
    parser.add_argument(
        "--query", help="Single query (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild knowledge base from documents"
    )

    args = parser.parse_args()

    # Setup knowledge base
    db_path = Path(args.db_path)

    if args.rebuild or not db_path.exists():
        storage = setup_knowledge_base(args.docs_dir, args.db_path)
    else:
        storage = JSONLStorage(args.db_path)
        print(f"\n[*] Loaded existing knowledge base: {args.db_path}")

    # Search
    if args.query:
        search_knowledge_base(storage, args.query)
    else:
        interactive_search(storage)


if __name__ == "__main__":
    main()
