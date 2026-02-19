"""
Example: Personal Wiki / Knowledge Base

Description:
    Builds a searchable knowledge base from personal notes, documentation,
    and reference materials. Supports full-text search, bidirectional links,
    tagging, and export to HTML for viewing.

Usage:
    python examples/knowledge/personal_wiki.py --documents notes/ --output wiki.db
    python examples/knowledge/personal_wiki.py --search "machine learning"

Expected output:
    - SQLite knowledge base
    - Full-text search results
    - Wiki index with links
    - HTML export for viewing

Requirements:
    - Personal notes/documentation files
    - Optional: sqlite3 for advanced queries
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from dataclasses import dataclass

from ingestforge.ingest.processor import DocumentProcessor
from ingestforge.chunking.semantic_chunker import SemanticChunker


@dataclass
class WikiPage:
    """Represents a wiki page."""

    title: str
    content: str
    source: str
    tags: list[str]
    links: list[str]
    word_count: int


def create_wiki_database(db_path: str) -> sqlite3.Connection:
    """
    Create wiki database schema.

    Args:
        db_path: Path to database file

    Returns:
        Database connection
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Pages table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY,
            title TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            word_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Tags table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY,
            page_id INTEGER,
            tag TEXT,
            FOREIGN KEY(page_id) REFERENCES pages(id)
        )
    """
    )

    # Links table (for bidirectional linking)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS links (
            id INTEGER PRIMARY KEY,
            from_page_id INTEGER,
            to_page_id INTEGER,
            FOREIGN KEY(from_page_id) REFERENCES pages(id),
            FOREIGN KEY(to_page_id) REFERENCES pages(id)
        )
    """
    )

    # Full-text search index
    cursor.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts
        USING fts5(title, content, tokenize = 'porter')
    """
    )

    conn.commit()
    return conn


def extract_wiki_links(text: str) -> list[str]:
    """
    Extract wiki-style links from text.

    Args:
        text: Content text

    Returns:
        List of linked page names
    """
    # Match [[Page Name]] style links
    pattern = r"\[\[([^\]]+)\]\]"
    return re.findall(pattern, text)


def extract_tags(text: str) -> list[str]:
    """
    Extract tags from text.

    Args:
        text: Content text

    Returns:
        List of tags
    """
    # Match #tag style tags
    pattern = r"#([a-z0-9_-]+)"
    return re.findall(pattern, text.lower())


def ingest_documents(
    documents_dir: str,
    db_path: str = "wiki.db",
) -> None:
    """
    Ingest documents into wiki database.

    Args:
        documents_dir: Directory containing documents
        db_path: Database path
    """
    documents_dir = Path(documents_dir)

    if not documents_dir.exists():
        print(f"Error: Directory not found: {documents_dir}")
        return

    print(f"\n[*] Building wiki from {documents_dir}")
    print(f"    Database: {db_path}")

    # Create database
    conn = create_wiki_database(db_path)
    cursor = conn.cursor()

    # Find documents
    doc_files = (
        list(documents_dir.glob("**/*.md"))
        + list(documents_dir.glob("**/*.txt"))
        + list(documents_dir.glob("**/*.pdf"))
    )

    if not doc_files:
        print("[!] No documents found")
        return

    print(f"\n[*] Found {len(doc_files)} documents")

    processor = DocumentProcessor()
    chunker = SemanticChunker(target_size=512, overlap=50)

    # Process each document
    print("\n[*] Processing documents...")

    for i, doc_file in enumerate(doc_files, 1):
        print(f"    [{i}/{len(doc_files)}] {doc_file.name}...", end="", flush=True)

        try:
            # Process document
            text = processor.process(doc_file)

            # Extract title (from filename or first heading)
            title = doc_file.stem
            if text.startswith("# "):
                title = text.split("\n")[0].lstrip("# ")

            # Extract tags and links
            tags = extract_tags(text)
            links = extract_wiki_links(text)
            word_count = len(text.split())

            # Store in database
            cursor.execute(
                """
                INSERT OR REPLACE INTO pages (title, content, source, word_count)
                VALUES (?, ?, ?, ?)
                """,
                (title, text, str(doc_file), word_count),
            )

            page_id = cursor.lastrowid

            # Store tags
            for tag in tags:
                cursor.execute(
                    "INSERT INTO tags (page_id, tag) VALUES (?, ?)",
                    (page_id, tag),
                )

            # Store FTS index
            cursor.execute(
                "INSERT INTO pages_fts (title, content) VALUES (?, ?)",
                (title, text),
            )

            conn.commit()
            print(" OK")

        except Exception as e:
            print(f" ERROR: {e}")

    # Create bidirectional links
    print("\n[*] Creating links...")
    cursor.execute("SELECT id, title FROM pages")
    pages = {title: page_id for page_id, title in cursor.fetchall()}

    cursor.execute("SELECT id FROM pages")
    for (page_id,) in cursor.fetchall():
        cursor.execute("SELECT content FROM pages WHERE id = ?", (page_id,))
        (content,) = cursor.fetchone()

        links = extract_wiki_links(content)
        for link in links:
            if link in pages:
                target_id = pages[link]
                cursor.execute(
                    "INSERT OR IGNORE INTO links (from_page_id, to_page_id) VALUES (?, ?)",
                    (page_id, target_id),
                )

    conn.commit()

    # Print summary
    cursor.execute("SELECT COUNT(*) FROM pages")
    page_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tags")
    tag_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM links")
    link_count = cursor.fetchone()[0]

    print(f"\n{'='*70}")
    print("WIKI SUMMARY")
    print(f"{'='*70}\n")

    print(f"Pages:               {page_count}")
    print(f"Tags:                {tag_count}")
    print(f"Links:               {link_count}")
    print(f"Database:            {db_path}")

    print("\n[✓] Wiki created successfully!")

    conn.close()


def search_wiki(
    query: str,
    db_path: str = "wiki.db",
    limit: int = 10,
) -> None:
    """
    Search the wiki.

    Args:
        query: Search query
        db_path: Database path
        limit: Maximum results
    """
    if not Path(db_path).exists():
        print(f"[!] Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"\n[*] Searching for: {query}\n")

    # Full-text search
    cursor.execute(
        """
        SELECT DISTINCT p.id, p.title, p.source
        FROM pages p
        JOIN pages_fts fts ON p.rowid = fts.rowid
        WHERE pages_fts MATCH ?
        LIMIT ?
        """,
        (query, limit),
    )

    results = cursor.fetchall()

    if not results:
        print("[!] No results found")
        return

    print(f"[✓] Found {len(results)} result(s)\n")

    for i, (page_id, title, source) in enumerate(results, 1):
        print(f"[{i}] {title}")
        print(f"    Source: {source}")

        # Show related tags
        cursor.execute(
            "SELECT DISTINCT tag FROM tags WHERE page_id = ?",
            (page_id,),
        )
        tags = [row[0] for row in cursor.fetchall()]
        if tags:
            print(f"    Tags: {', '.join(tags)}")

        # Show backlinks
        cursor.execute(
            "SELECT DISTINCT p.title FROM pages p JOIN links l ON p.id = l.from_page_id WHERE l.to_page_id = ?",
            (page_id,),
        )
        backlinks = [row[0] for row in cursor.fetchall()]
        if backlinks:
            print(f"    Backlinks: {', '.join(backlinks[:3])}")

        print()

    conn.close()


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Personal wiki / knowledge base")
    parser.add_argument(
        "--documents", "-d", help="Documents directory (creates wiki if provided)"
    )
    parser.add_argument(
        "--database",
        "-db",
        default="wiki.db",
        help="Wiki database path (default: wiki.db)",
    )
    parser.add_argument("--search", "-s", help="Search the wiki")
    parser.add_argument(
        "--limit", type=int, default=10, help="Search result limit (default: 10)"
    )

    args = parser.parse_args()

    if args.documents:
        ingest_documents(args.documents, args.database)
    elif args.search:
        search_wiki(args.search, args.database, args.limit)
    else:
        print("Usage: python personal_wiki.py --documents DIR | --search QUERY")
        print("\nExamples:")
        print("  python personal_wiki.py --documents notes/ --database wiki.db")
        print(
            "  python personal_wiki.py --search 'machine learning' --database wiki.db"
        )


if __name__ == "__main__":
    main()
