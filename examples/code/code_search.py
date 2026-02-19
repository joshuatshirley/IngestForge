"""
Example: Semantic Code Search

Description:
    Demonstrates semantic search across a codebase. Allows finding
    relevant code snippets by describing what you're looking for in
    natural language, rather than exact keywords.

Usage:
    python examples/code/code_search.py \
        --codebase src/ \
        --query "authentication logic" \
        --language python

    python examples/code/code_search.py \
        --index my_project.jsonl \
        --query "error handling"

Expected output:
    - Relevant code snippets
    - File locations
    - Similarity scores
    - Function/class context

Requirements:
    - Source code files
    - Embeddings model
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ingestforge.ingest.processor import DocumentProcessor
from ingestforge.chunking.code_chunker import CodeChunker
from ingestforge.retrieval.semantic import SemanticRetriever
from ingestforge.storage.jsonl import JSONLStorage


def index_codebase(
    codebase_dir: str,
    output_index: str = "code_index.jsonl",
    language: str = "python",
) -> None:
    """
    Index a codebase for semantic search.

    Args:
        codebase_dir: Directory containing source code
        output_index: Where to save the index
        language: Programming language (python, javascript, java, etc.)
    """
    codebase_dir = Path(codebase_dir)

    if not codebase_dir.exists():
        print(f"Error: Directory not found: {codebase_dir}")
        return

    print(f"\n[*] Indexing codebase: {codebase_dir}")
    print(f"    Language: {language}")
    print(f"    Output: {output_index}")

    # Find source files
    extensions = {
        "python": ["**/*.py"],
        "javascript": ["**/*.js", "**/*.ts"],
        "java": ["**/*.java"],
        "csharp": ["**/*.cs"],
        "cpp": ["**/*.cpp", "**/*.h"],
        "go": ["**/*.go"],
    }

    patterns = extensions.get(language, ["**/*"])

    source_files = []
    for pattern in patterns:
        source_files.extend(codebase_dir.glob(pattern))

    # Exclude common non-source directories
    exclude_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}
    source_files = [
        f for f in source_files if not any(ex in f.parts for ex in exclude_dirs)
    ]

    if not source_files:
        print(f"[!] No {language} source files found")
        return

    print(f"\n[*] Found {len(source_files)} source file(s)")

    # Index files
    print("\n[*] Indexing code...")

    processor = DocumentProcessor()
    code_chunker = CodeChunker(language=language)
    storage = JSONLStorage(output_index)

    all_chunks = []

    for i, source_file in enumerate(sorted(source_files), 1):
        try:
            print(
                f"    [{i}/{len(source_files)}] {source_file.relative_to(codebase_dir)}...",
                end="",
                flush=True,
            )

            # Read source
            with open(source_file, "r", encoding="utf-8", errors="ignore") as f:
                code_text = f.read()

            # Chunk by function/class
            chunks = code_chunker.chunk(code_text)

            # Add metadata
            for chunk in chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}

                chunk["metadata"].update(
                    {
                        "file": str(source_file),
                        "file_relative": str(source_file.relative_to(codebase_dir)),
                        "language": language,
                        "size": len(code_text),
                    }
                )

            all_chunks.extend(chunks)
            print(f" OK ({len(chunks)} chunks)")

        except Exception as e:
            print(f" ERROR: {e}")

    # Save index
    print(f"\n[*] Saving {len(all_chunks)} chunks to {output_index}...")
    storage.save(all_chunks)

    print(f"\n{'='*70}")
    print("CODE INDEX SUMMARY")
    print(f"{'='*70}\n")

    print(f"Source files:        {len(source_files)}")
    print(f"Code chunks:         {len(all_chunks)}")
    print(f"Language:            {language}")
    print(f"Index file:          {output_index}")

    print("\n[✓] Indexing complete!")
    print(
        f"\nTo search: python code_search.py --index {output_index} --query YOUR_QUERY"
    )


def search_codebase(
    query: str,
    codebase_dir: Optional[str] = None,
    index_path: Optional[str] = None,
    language: str = "python",
    k: int = 5,
) -> None:
    """
    Search the codebase semantically.

    Args:
        query: Natural language search query
        codebase_dir: Codebase to search (creates index if needed)
        index_path: Path to pre-built index
        language: Programming language
        k: Number of results to return
    """
    print(f"\n[*] Searching: {query}")
    print(f"    Results: top {k}\n")

    # Load or create index
    if index_path and Path(index_path).exists():
        storage = JSONLStorage(index_path)
        chunks = storage.load()
    elif codebase_dir:
        # Create temporary index
        temp_index = ".tmp_code_index.jsonl"
        index_codebase(codebase_dir, temp_index, language)
        storage = JSONLStorage(temp_index)
        chunks = storage.load()
    else:
        print("[!] Provide either --index or --codebase")
        return

    if not chunks:
        print("[!] No code chunks found")
        return

    # Search
    retriever = SemanticRetriever(chunks)
    results = retriever.retrieve(query, k=k)

    if not results:
        print("[!] No matching code found")
        return

    # Print results
    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        score = result.get("score", 0)

        print(f"[Result {i}] (Score: {score:.3f})")

        # File and function/class info
        file_rel = metadata.get("file_relative", metadata.get("file", "unknown"))
        print(f"File: {file_rel}")

        # Code preview
        code_text = result.get("text", "")[:400]
        if len(result.get("text", "")) > 400:
            code_text += "\n    ..."

        print("Code:")
        for line in code_text.split("\n"):
            print(f"    {line}")

        print()


def interactive_search(
    codebase_dir: Optional[str] = None,
    index_path: Optional[str] = None,
    language: str = "python",
) -> None:
    """
    Interactive search mode.

    Args:
        codebase_dir: Codebase to search
        index_path: Index file
        language: Programming language
    """
    print("\n" + "=" * 70)
    print("CODE SEMANTIC SEARCH - Interactive Mode")
    print("=" * 70)
    print("\nDescribe what you're looking for in natural language.")
    print("Type 'exit' or 'quit' to exit.\n")

    while True:
        try:
            query = input("Search> ").strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit"):
                print("\n[*] Goodbye!")
                break

            search_codebase(query, codebase_dir, index_path, language, k=3)

        except KeyboardInterrupt:
            print("\n\n[*] Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[!] Error: {e}\n")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic code search")
    parser.add_argument("--codebase", "-c", help="Codebase directory to index")
    parser.add_argument("--index", "-i", help="Pre-built index file to search")
    parser.add_argument(
        "--language",
        "-l",
        default="python",
        choices=["python", "javascript", "java", "csharp", "cpp", "go"],
        help="Programming language (default: python)",
    )
    parser.add_argument(
        "--query", "-q", help="Search query (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--results", "-k", type=int, default=5, help="Number of results (default: 5)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.codebase and not args.index:
        print("Error: Provide either --codebase or --index")
        print("Usage: python code_search.py --codebase src/ [--query QUERY]")
        return

    if args.codebase and not Path(args.codebase).exists():
        print(f"Error: Directory not found: {args.codebase}")
        return

    # Index codebase if needed
    if args.codebase and not args.index:
        index_path = ".tmp_code_index.jsonl"
        if not Path(index_path).exists():
            index_codebase(args.codebase, index_path, args.language)
        args.index = index_path

    # Search
    if args.query:
        search_codebase(
            args.query,
            codebase_dir=args.codebase,
            index_path=args.index,
            language=args.language,
            k=args.results,
        )
    else:
        interactive_search(args.codebase, args.index, args.language)


if __name__ == "__main__":
    main()
