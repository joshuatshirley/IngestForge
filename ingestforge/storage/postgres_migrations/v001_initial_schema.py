"""Initial schema migration.

Creates the base tables and indexes for IngestForge PostgreSQL storage.
"""

from typing import Any


def up(conn: Any, table_name: str, embedding_dim: int) -> None:
    """Apply migration.

    Args:
        conn: Database connection
        table_name: Main chunks table name
        embedding_dim: Embedding vector dimension
    """
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Main chunks table
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                chunk_id VARCHAR(255) PRIMARY KEY,
                document_id VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                section_title VARCHAR(512) DEFAULT '',
                chunk_type VARCHAR(64) DEFAULT 'content',
                source_file VARCHAR(512) DEFAULT '',
                word_count INTEGER DEFAULT 0,
                char_count INTEGER DEFAULT 0,
                chunk_index INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 1,
                page_start INTEGER,
                page_end INTEGER,
                library VARCHAR(255) DEFAULT 'default',
                embedding vector({embedding_dim}),
                entities JSONB DEFAULT '[]',
                concepts JSONB DEFAULT '[]',
                quality_score REAL DEFAULT 0.0,
                metadata JSONB DEFAULT '{{}}',
                source_location JSONB,
                ingested_at TIMESTAMP DEFAULT NOW(),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """
        )

        # Parent-child mapping table
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name}_parent_map (
                child_chunk_id VARCHAR(255) PRIMARY KEY,
                parent_chunk_id VARCHAR(255) NOT NULL,
                document_id VARCHAR(255) NOT NULL,
                child_position INTEGER DEFAULT 0,
                total_children INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """
        )

        # Create indexes
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_document
            ON {table_name}(document_id)
        """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_library
            ON {table_name}(library)
        """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_source
            ON {table_name}(source_file)
        """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_metadata
            ON {table_name} USING GIN (metadata)
        """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_content_fts
            ON {table_name}
            USING GIN (to_tsvector('english', content))
        """
        )


def down(conn: Any, table_name: str) -> None:
    """Rollback migration.

    Args:
        conn: Database connection
        table_name: Main chunks table name
    """
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}_parent_map")
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
