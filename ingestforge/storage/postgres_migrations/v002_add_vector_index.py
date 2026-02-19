"""Add IVFFlat vector index migration.

Creates the pgvector IVFFlat index for efficient similarity search.
Note: IVFFlat requires data in the table for optimal index creation.
"""

from typing import Any


def up(conn: Any, table_name: str, embedding_dim: int) -> None:
    """Apply migration.

    Args:
        conn: Database connection
        table_name: Main chunks table name
        embedding_dim: Embedding vector dimension (unused but kept for interface)
    """
    with conn.cursor() as cur:
        # Check if we have enough data for IVFFlat
        cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE embedding IS NOT NULL")
        count = cur.fetchone()[0]

        # IVFFlat with lists parameter - use sqrt(n) as a rule of thumb
        # Minimum 1 list, max 1000 lists for very large datasets
        lists = max(1, min(int(count**0.5), 1000)) if count > 0 else 100

        # Drop existing index if present
        cur.execute(
            f"""
            DROP INDEX IF EXISTS idx_{table_name}_embedding
        """
        )

        # Create IVFFlat index
        # Note: For empty tables, we create with default lists=100
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding
            ON {table_name}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists})
        """
        )


def down(conn: Any, table_name: str) -> None:
    """Rollback migration.

    Args:
        conn: Database connection
        table_name: Main chunks table name
    """
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS idx_{table_name}_embedding")
