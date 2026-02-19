"""PostgreSQL storage backend with pgvector.

Production-grade storage using PostgreSQL with pgvector extension
for vector similarity search.

TASK-018: Migrated to IFChunkArtifact + Fixed SQL Injection (CRITICAL-002)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.base import (
    ChunkRepository,
    SearchResult,
    normalize_to_chunk_record,
    ChunkInput,
)

if TYPE_CHECKING:
    pass

# Lazy import check for psycopg2
try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import RealDictCursor, execute_values

    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    psycopg2 = None
    pool = None
    sql = None
    RealDictCursor = None
    execute_values = None


class _Logger:
    """Lazy logger holder."""

    _instance = None

    @classmethod
    def get(cls):
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


# Schema version for migrations
SCHEMA_VERSION = 2

# Default embedding dimension
DEFAULT_EMBEDDING_DIM = 384

# Security: Allowed table names (CRITICAL-002 mitigation)
ALLOWED_TABLES = frozenset(
    {
        "chunks",
        "documents",
        "metadata",
        "embeddings",
        "chunks_parent_map",
    }
)


def _validate_table_name(table_name: str) -> None:
    """
    Validate table name against whitelist.

    CRITICAL-002: SQL injection mitigation.
    Rule #7: Input validation.
    Rule #1: Early return on error.

    Args:
        table_name: Table name to validate

    Raises:
        ValueError: If table name is not in whitelist
    """
    if table_name not in ALLOWED_TABLES:
        raise ValueError(
            f"Invalid table name: {table_name!r}. "
            f"Must be one of: {', '.join(sorted(ALLOWED_TABLES))}"
        )


class MigrationRunner:
    """Runs and tracks PostgreSQL schema migrations.

        - Rule #4: Functions <60 lines
    - Rule #5: Log all errors
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "chunks",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        """Initialize migration runner.

        Args:
            connection_string: PostgreSQL connection string
            table_name: Main chunks table name
            embedding_dim: Embedding vector dimension
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim

    def get_current_version(self, conn: Any) -> int:
        """Get current schema version.

        Args:
            conn: Database connection

        Returns:
            Current version number (0 if no migrations applied)
        """
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'schema_version'
                    )
                """
                )
                if not cur.fetchone()[0]:
                    return 0

                cur.execute("SELECT MAX(version) FROM schema_version")
                result = cur.fetchone()[0]
                return result or 0
        except Exception as e:
            _Logger.get().debug(f"Could not get schema version: {e}")
            return 0

    def run_migrations(self, conn: Any) -> int:
        """Run pending migrations.

        Args:
            conn: Database connection

        Returns:
            Number of migrations applied
        """
        self._ensure_schema_version_table(conn)
        current_version = self.get_current_version(conn)
        applied = 0

        from ingestforge.storage.postgres_migrations import get_available_migrations

        for version, description, path in get_available_migrations():
            if version <= current_version:
                continue

            _Logger.get().info(f"Applying migration v{version}: {description}")
            self._apply_migration(conn, version, path)
            applied += 1

        return applied

    def _ensure_schema_version_table(self, conn: Any) -> None:
        """Ensure schema_version table exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    description VARCHAR(255),
                    applied_at TIMESTAMP DEFAULT NOW()
                )
            """
            )

    def _apply_migration(self, conn: Any, version: int, path: Any) -> None:
        """Apply a single migration."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(f"migration_v{version}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Call the up function
        module.up(conn, self.table_name, self.embedding_dim)

        # Record migration
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO schema_version (version, description) VALUES (%s, %s)",
                (version, path.stem),
            )

        conn.commit()
        _Logger.get().info(f"Migration v{version} applied successfully")


class PostgresRepository(ChunkRepository):
    """PostgreSQL storage with pgvector for vector search.

    Implements ChunkRepository interface for PostgreSQL with pgvector extension.
    Supports semantic search, library management, and parent-child chunk mappings.
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "chunks",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        """Initialize PostgreSQL storage.

        Args:
            connection_string: PostgreSQL connection string
            table_name: Table name for chunks
            embedding_dim: Embedding vector dimension
            min_pool_size: Minimum pool connections
            max_pool_size: Maximum pool connections

        Raises:
            ImportError: If psycopg2 not installed
            ValueError: If table_name is invalid (CRITICAL-002)
        """
        if not HAS_POSTGRES:
            raise ImportError(
                "psycopg2 required for PostgreSQL. "
                "Install: pip install psycopg2-binary"
            )

        # CRITICAL-002: Validate table name before use
        _validate_table_name(table_name)

        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self._pool = self._create_pool(min_pool_size, max_pool_size)
        self._ensure_schema()

    def _create_pool(self, min_size: int, max_size: int) -> pool.ThreadedConnectionPool:
        """Create threaded connection pool.

        Args:
            min_size: Minimum connections
            max_size: Maximum connections

        Returns:
            Connection pool
        """
        _Logger.get().info(
            f"Creating PostgreSQL connection pool (min={min_size}, max={max_size})"
        )
        return pool.ThreadedConnectionPool(min_size, max_size, self.connection_string)

    def _get_conn(self) -> Any:
        """Get connection from pool."""
        return self._pool.getconn()

    def _put_conn(self, conn: Any) -> None:
        """Return connection to pool."""
        self._pool.putconn(conn)

    def _ensure_schema(self) -> None:
        """Ensure database schema exists using migrations."""
        conn = self._get_conn()
        try:
            # First ensure pgvector extension exists
            self._create_extension(conn)
            conn.commit()

            # Run migrations
            runner = MigrationRunner(
                self.connection_string,
                self.table_name,
                self.embedding_dim,
            )
            applied = runner.run_migrations(conn)

            if applied > 0:
                _Logger.get().info(f"Applied {applied} schema migration(s)")
            else:
                _Logger.get().debug("Schema is up to date")

        except Exception as e:
            conn.rollback()
            _Logger.get().error(f"Failed to initialize schema: {e}")
            raise
        finally:
            self._put_conn(conn)

    def get_schema_version(self) -> int:
        """Get current schema version.

        Returns:
            Current schema version number
        """
        conn = self._get_conn()
        try:
            runner = MigrationRunner(
                self.connection_string,
                self.table_name,
                self.embedding_dim,
            )
            return runner.get_current_version(conn)
        finally:
            self._put_conn(conn)

    def _create_extension(self, conn: Any) -> None:
        """Create pgvector extension if not exists."""
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # =========================================================================
    # ChunkRepository Interface Implementation
    # =========================================================================

    def add_chunk(self, chunk: ChunkInput) -> bool:
        """Add a single chunk to storage.

        TASK-018: Now accepts ChunkRecord or IFChunkArtifact.

        Args:
            chunk: Chunk to add (ChunkRecord or IFChunkArtifact)

        Returns:
            True if successful
        """
        # TASK-018: Normalize to ChunkRecord for storage
        chunk_record = normalize_to_chunk_record(chunk)

        conn = self._get_conn()
        try:
            self._insert_chunk(conn, chunk_record)
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            _Logger.get().error(f"Failed to add chunk {chunk_record.chunk_id}: {e}")
            return False
        finally:
            self._put_conn(conn)

    def add_chunks(self, chunks: List[ChunkInput]) -> int:
        """Add multiple chunks to storage.

        TASK-018: Now accepts list of ChunkRecord or IFChunkArtifact.

        Args:
            chunks: Chunks to add (list of ChunkRecord or IFChunkArtifact)

        Returns:
            Number of chunks successfully added
        """
        if not chunks:
            return 0

        # TASK-018: Normalize all to ChunkRecord
        chunk_records = [normalize_to_chunk_record(c) for c in chunks]

        conn = self._get_conn()
        try:
            count = self._batch_insert_chunks(conn, chunk_records)
            conn.commit()
            _Logger.get().info(f"Added {count} chunks to PostgreSQL")
            return count
        except Exception as e:
            conn.rollback()
            _Logger.get().error(f"Failed to add chunks: {e}")
            return 0
        finally:
            self._put_conn(conn)

    def _insert_chunk(self, conn: Any, chunk: ChunkRecord) -> None:
        """Insert single chunk into database.

        CRITICAL-002: Uses psycopg2.sql.Identifier for safe SQL construction.
        Rule #7: All user inputs are parameterized.
        """
        embedding_str = self._format_embedding(chunk.embedding)
        source_loc = self._serialize_source_location(chunk.source_location)

        # CRITICAL-002: Safe SQL construction using psycopg2.sql
        query = sql.SQL(
            """
            INSERT INTO {} (
                chunk_id, document_id, content, section_title, chunk_type,
                source_file, word_count, char_count, chunk_index, total_chunks,
                page_start, page_end, library, embedding, entities, concepts,
                quality_score, metadata, source_location
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s::vector, %s, %s, %s, %s, %s
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                entities = EXCLUDED.entities,
                concepts = EXCLUDED.concepts,
                quality_score = EXCLUDED.quality_score
        """
        ).format(sql.Identifier(self.table_name))

        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.content,
                    chunk.section_title,
                    chunk.chunk_type,
                    chunk.source_file,
                    chunk.word_count,
                    chunk.char_count,
                    chunk.chunk_index,
                    chunk.total_chunks,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.library,
                    embedding_str,
                    json.dumps(chunk.entities),
                    json.dumps(chunk.concepts),
                    chunk.quality_score,
                    json.dumps(chunk.metadata),
                    source_loc,
                ),
            )

    def _batch_insert_chunks(self, conn: Any, chunks: List[ChunkRecord]) -> int:
        """Batch insert chunks for efficiency.

        CRITICAL-002: Uses psycopg2.sql.Identifier for safe SQL construction.
        """
        values = []
        for chunk in chunks:
            embedding_str = self._format_embedding(chunk.embedding)
            source_loc = self._serialize_source_location(chunk.source_location)
            values.append(
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.content,
                    chunk.section_title,
                    chunk.chunk_type,
                    chunk.source_file,
                    chunk.word_count,
                    chunk.char_count,
                    chunk.chunk_index,
                    chunk.total_chunks,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.library,
                    embedding_str,
                    json.dumps(chunk.entities),
                    json.dumps(chunk.concepts),
                    chunk.quality_score,
                    json.dumps(chunk.metadata),
                    source_loc,
                )
            )

        # CRITICAL-002: Safe SQL construction
        query = sql.SQL(
            """
            INSERT INTO {} (
                chunk_id, document_id, content, section_title, chunk_type,
                source_file, word_count, char_count, chunk_index, total_chunks,
                page_start, page_end, library, embedding, entities, concepts,
                quality_score, metadata, source_location
            ) VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
        """
        ).format(sql.Identifier(self.table_name))

        with conn.cursor() as cur:
            execute_values(
                cur,
                query,
                values,
                template="""(
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s::vector, %s, %s, %s, %s, %s
                )""",
            )
        return len(chunks)

    def _format_embedding(self, embedding: Optional[List[float]]) -> Optional[str]:
        """Format embedding for PostgreSQL vector type."""
        if not embedding:
            return None
        return f"[{','.join(map(str, embedding))}]"

    def _serialize_source_location(self, source_loc: Optional[Any]) -> Optional[str]:
        """Serialize source location to JSON."""
        if source_loc is None:
            return None
        if hasattr(source_loc, "to_dict"):
            return json.dumps(source_loc.to_dict())
        if isinstance(source_loc, dict):
            return json.dumps(source_loc)
        return None

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        """Get a chunk by ID.

        CRITICAL-002: Safe SQL using psycopg2.sql.Identifier.

        Args:
            chunk_id: Chunk identifier

        Returns:
            ChunkRecord or None if not found
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # CRITICAL-002: Safe SQL construction
                query = sql.SQL("SELECT * FROM {} WHERE chunk_id = %s").format(
                    sql.Identifier(self.table_name)
                )
                cur.execute(query, (chunk_id,))
                row = cur.fetchone()
                if row:
                    return self._row_to_chunk(row)
                return None
        finally:
            self._put_conn(conn)

    def get_chunks_by_document(self, document_id: str) -> List[ChunkRecord]:
        """Get all chunks for a document.

        CRITICAL-002: Safe SQL using psycopg2.sql.Identifier.

        Args:
            document_id: Document identifier

        Returns:
            List of chunks
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # CRITICAL-002: Safe SQL construction
                query = sql.SQL(
                    """
                    SELECT * FROM {}
                    WHERE document_id = %s
                    ORDER BY chunk_index
                """
                ).format(sql.Identifier(self.table_name))
                cur.execute(query, (document_id,))
                return [self._row_to_chunk(row) for row in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def _row_to_chunk(self, row: Dict[str, Any]) -> ChunkRecord:
        """Convert database row to ChunkRecord."""
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            content=row["content"],
            section_title=row.get("section_title", ""),
            chunk_type=row.get("chunk_type", "content"),
            source_file=row.get("source_file", ""),
            word_count=row.get("word_count", 0),
            char_count=row.get("char_count", 0),
            chunk_index=row.get("chunk_index", 0),
            total_chunks=row.get("total_chunks", 1),
            page_start=row.get("page_start"),
            page_end=row.get("page_end"),
            library=row.get("library", "default"),
            embedding=None,  # Don't load large embeddings by default
            entities=row.get("entities") or [],
            concepts=row.get("concepts") or [],
            quality_score=row.get("quality_score", 0.0),
            metadata=row.get("metadata") or {},
            source_location=row.get("source_location"),
            ingested_at=str(row.get("ingested_at")) if row.get("ingested_at") else None,
        )

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk.

        CRITICAL-002: Safe SQL using psycopg2.sql.Identifier.

        Args:
            chunk_id: Chunk to delete

        Returns:
            True if deleted
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # CRITICAL-002: Safe SQL construction
                query = sql.SQL("DELETE FROM {} WHERE chunk_id = %s").format(
                    sql.Identifier(self.table_name)
                )
                cur.execute(query, (chunk_id,))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted
        except Exception as e:
            conn.rollback()
            _Logger.get().error(f"Failed to delete chunk {chunk_id}: {e}")
            return False
        finally:
            self._put_conn(conn)

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        CRITICAL-002: Safe SQL using psycopg2.sql.Identifier.

        Args:
            document_id: Document to delete

        Returns:
            Number of chunks deleted
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # CRITICAL-002: Safe SQL construction
                query = sql.SQL("DELETE FROM {} WHERE document_id = %s").format(
                    sql.Identifier(self.table_name)
                )
                cur.execute(query, (document_id,))
                deleted = cur.rowcount
                conn.commit()
                _Logger.get().info(
                    f"Deleted {deleted} chunks for document {document_id}"
                )
                return deleted
        except Exception as e:
            conn.rollback()
            _Logger.get().error(f"Failed to delete document {document_id}: {e}")
            return 0
        finally:
            self._put_conn(conn)

    def search(
        self,
        query: str,
        top_k: int = 10,
        library_filter: Optional[str] = None,
        document_filter: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """Search for relevant chunks using full-text search.

        Args:
            query: Search query
            top_k: Number of results
            library_filter: Filter by library
            document_filter: Filter by document

        Returns:
            List of SearchResult
        """
        if not query.strip():
            return []

        conn = self._get_conn()
        try:
            return self._execute_fts_search(
                conn, query, top_k, library_filter, document_filter
            )
        finally:
            self._put_conn(conn)

    def _execute_fts_search(
        self,
        conn: Any,
        query: str,
        top_k: int,
        library_filter: Optional[str],
        document_filter: Optional[str],
    ) -> List[SearchResult]:
        """Execute full-text search query.

        CRITICAL-002: Safe SQL using psycopg2.sql.Identifier.
        Rule #4: Function < 60 lines.
        """
        where_clauses = []
        params: List[Any] = []

        if library_filter:
            where_clauses.append("library = %s")
            params.append(library_filter)

        if document_filter:
            where_clauses.append("document_id = %s")
            params.append(document_filter)

        where_sql = ""
        if where_clauses:
            where_sql = "AND " + " AND ".join(where_clauses)

        # Build tsquery from terms
        terms = query.split()
        tsquery = " | ".join(terms)

        params = [tsquery] + params + [top_k]

        # CRITICAL-002: Safe SQL construction
        query_template = sql.SQL(
            """
            SELECT *,
                ts_rank(to_tsvector('english', content), plainto_tsquery(%s)) as score
            FROM {}
            WHERE to_tsvector('english', content) @@ plainto_tsquery(%s)
            {where_clause}
            ORDER BY score DESC
            LIMIT %s
        """
        ).format(
            sql.Identifier(self.table_name),
            where_clause=sql.SQL(where_sql),
        )

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query_template, [tsquery, tsquery] + params[1:])

            results = []
            for row in cur.fetchall():
                chunk = self._row_to_chunk(row)
                results.append(SearchResult.from_chunk(chunk, float(row["score"])))
            return results

    def search_semantic(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        library_filter: Optional[str] = None,
        document_filter: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """Search using embedding vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            library_filter: Filter by library
            document_filter: Filter by document

        Returns:
            List of SearchResult
        """
        conn = self._get_conn()
        try:
            return self._execute_vector_search(
                conn, query_embedding, top_k, library_filter, document_filter
            )
        finally:
            self._put_conn(conn)

    def _execute_vector_search(
        self,
        conn: Any,
        query_embedding: List[float],
        top_k: int,
        library_filter: Optional[str],
        document_filter: Optional[str],
    ) -> List[SearchResult]:
        """Execute vector similarity search."""
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        where_clauses = ["embedding IS NOT NULL"]
        params: List[Any] = []

        if library_filter:
            where_clauses.append("library = %s")
            params.append(library_filter)

        if document_filter:
            where_clauses.append("document_id = %s")
            params.append(document_filter)

        where_sql = " AND ".join(where_clauses)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT *,
                    1 - (embedding <=> %s::vector) as score
                FROM {self.table_name}
                WHERE {where_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """,
                [embedding_str] + params + [embedding_str, top_k],
            )

            results = []
            for row in cur.fetchall():
                chunk = self._row_to_chunk(row)
                results.append(SearchResult.from_chunk(chunk, float(row["score"])))
            return results

    def count(self) -> int:
        """Get total number of chunks.

        Returns:
            Chunk count
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cur.fetchone()[0]
        finally:
            self._put_conn(conn)

    def clear(self) -> None:
        """Clear all stored chunks."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name}")
                cur.execute(f"TRUNCATE TABLE {self.table_name}_parent_map")
            conn.commit()
            _Logger.get().info("Cleared all chunks from PostgreSQL")
        except Exception as e:
            conn.rollback()
            _Logger.get().error(f"Failed to clear chunks: {e}")
            raise
        finally:
            self._put_conn(conn)

    # =========================================================================
    # Library Operations
    # =========================================================================

    def get_libraries(self) -> List[str]:
        """Get list of unique library names.

        Returns:
            List of library names
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT library FROM {self.table_name} ORDER BY library"
                )
                libraries = [row[0] for row in cur.fetchall()]
                if "default" not in libraries:
                    libraries.insert(0, "default")
                return libraries
        finally:
            self._put_conn(conn)

    def count_by_library(self, library_name: str) -> int:
        """Count chunks in a specific library.

        Args:
            library_name: Library to count

        Returns:
            Number of chunks
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {self.table_name} WHERE library = %s",
                    (library_name,),
                )
                return cur.fetchone()[0]
        finally:
            self._put_conn(conn)

    def delete_by_library(self, library_name: str) -> int:
        """Delete all chunks in a library.

        Args:
            library_name: Library to delete

        Returns:
            Number of chunks deleted
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE library = %s", (library_name,)
                )
                deleted = cur.rowcount
                conn.commit()
                return deleted
        except Exception as e:
            conn.rollback()
            _Logger.get().error(f"Failed to delete library {library_name}: {e}")
            return 0
        finally:
            self._put_conn(conn)

    def reassign_library(self, old_library: str, new_library: str) -> int:
        """Move all chunks from one library to another.

        Args:
            old_library: Source library
            new_library: Target library

        Returns:
            Number of chunks moved
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self.table_name} SET library = %s WHERE library = %s",
                    (new_library, old_library),
                )
                updated = cur.rowcount
                conn.commit()
                return updated
        except Exception as e:
            conn.rollback()
            _Logger.get().error(f"Failed to reassign library: {e}")
            return 0
        finally:
            self._put_conn(conn)

    def move_document_to_library(self, document_id: str, new_library: str) -> int:
        """Move document chunks to a different library.

        Args:
            document_id: Document to move
            new_library: Target library

        Returns:
            Number of chunks updated
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""UPDATE {self.table_name}
                        SET library = %s WHERE document_id = %s""",
                    (new_library, document_id),
                )
                updated = cur.rowcount
                conn.commit()
                return updated
        except Exception as e:
            conn.rollback()
            _Logger.get().error(f"Failed to move document: {e}")
            return 0
        finally:
            self._put_conn(conn)

    # =========================================================================
    # Parent-Child Mapping
    # =========================================================================

    def set_parent_mapping(
        self,
        child_chunk_id: str,
        parent_chunk_id: str,
    ) -> bool:
        """Map a child chunk to its parent chunk.

        Args:
            child_chunk_id: ID of child chunk
            parent_chunk_id: ID of parent chunk

        Returns:
            True if mapping was set
        """
        conn = self._get_conn()
        try:
            # Get document_id from child chunk
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT document_id FROM {self.table_name} WHERE chunk_id = %s",
                    (child_chunk_id,),
                )
                row = cur.fetchone()
                if not row:
                    return False
                document_id = row[0]

                cur.execute(
                    f"""
                    INSERT INTO {self.table_name}_parent_map
                    (child_chunk_id, parent_chunk_id, document_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (child_chunk_id) DO UPDATE SET
                        parent_chunk_id = EXCLUDED.parent_chunk_id
                """,
                    (child_chunk_id, parent_chunk_id, document_id),
                )

            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            _Logger.get().error(f"Failed to set parent mapping: {e}")
            return False
        finally:
            self._put_conn(conn)

    def get_parent_chunk(self, child_chunk_id: str) -> Optional[ChunkRecord]:
        """Get the parent chunk for a child chunk.

        Args:
            child_chunk_id: ID of child chunk

        Returns:
            Parent ChunkRecord or None
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT parent_chunk_id
                    FROM {self.table_name}_parent_map
                    WHERE child_chunk_id = %s
                """,
                    (child_chunk_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return self.get_chunk(row[0])
        finally:
            self._put_conn(conn)

    def get_child_chunks(self, parent_chunk_id: str) -> List[ChunkRecord]:
        """Get all child chunks for a parent chunk.

        Args:
            parent_chunk_id: ID of parent chunk

        Returns:
            List of child ChunkRecords
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT child_chunk_id
                    FROM {self.table_name}_parent_map
                    WHERE parent_chunk_id = %s
                    ORDER BY child_position
                """,
                    (parent_chunk_id,),
                )
                chunk_ids = [row[0] for row in cur.fetchall()]

            return [
                chunk
                for chunk_id in chunk_ids
                if (chunk := self.get_chunk(chunk_id)) is not None
            ]
        finally:
            self._put_conn(conn)

    # =========================================================================
    # Statistics and Admin
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary of statistics
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Total chunks
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                total_chunks = cur.fetchone()[0]

                # Total documents
                cur.execute(
                    f"SELECT COUNT(DISTINCT document_id) FROM {self.table_name}"
                )
                total_documents = cur.fetchone()[0]

                # Chunks with embeddings
                cur.execute(
                    f"SELECT COUNT(*) FROM {self.table_name} WHERE embedding IS NOT NULL"
                )
                chunks_with_embeddings = cur.fetchone()[0]

                # Libraries
                cur.execute(f"SELECT COUNT(DISTINCT library) FROM {self.table_name}")
                library_count = cur.fetchone()[0]

                # Table size
                cur.execute(
                    f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{self.table_name}'))
                """
                )
                table_size = cur.fetchone()[0]

            return {
                "total_chunks": total_chunks,
                "total_documents": total_documents,
                "chunks_with_embeddings": chunks_with_embeddings,
                "library_count": library_count,
                "table_size": table_size,
                "backend": "postgres",
                "embedding_dim": self.embedding_dim,
            }
        finally:
            self._put_conn(conn)

    def health_check(self) -> Tuple[bool, str]:
        """Check database health.

        Returns:
            Tuple of (healthy, message)
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            return True, "PostgreSQL connection healthy"
        except Exception as e:
            return False, f"PostgreSQL health check failed: {e}"
        finally:
            self._put_conn(conn)

    def vacuum(self) -> None:
        """Run VACUUM ANALYZE on chunks table."""
        conn = self._get_conn()
        try:
            # VACUUM cannot run in transaction
            old_isolation = conn.isolation_level
            conn.set_isolation_level(0)
            with conn.cursor() as cur:
                cur.execute(f"VACUUM ANALYZE {self.table_name}")
            conn.set_isolation_level(old_isolation)
            _Logger.get().info("VACUUM ANALYZE completed")
        finally:
            self._put_conn(conn)

    def reindex(self) -> None:
        """Reindex all indexes on chunks table."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"REINDEX TABLE {self.table_name}")
            conn.commit()
            _Logger.get().info("REINDEX completed")
        except Exception as e:
            conn.rollback()
            _Logger.get().error(f"REINDEX failed: {e}")
            raise
        finally:
            self._put_conn(conn)

    def close(self) -> None:
        """Close all connections."""
        if self._pool:
            self._pool.closeall()
            _Logger.get().info("PostgreSQL connection pool closed")


# Backward compatibility alias
PostgreSQLStorage = PostgresRepository


def create_postgres_repository(
    connection_string: str,
    table_name: str = "chunks",
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> PostgresRepository:
    """Create PostgreSQL repository instance.

    Args:
        connection_string: PostgreSQL connection string
        table_name: Table name
        embedding_dim: Embedding dimension

    Returns:
        PostgreSQL repository instance
    """
    return PostgresRepository(connection_string, table_name, embedding_dim)
