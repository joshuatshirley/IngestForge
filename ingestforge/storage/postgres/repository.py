"""PostgreSQL Storage Repository.

Provides relational storage for document chunks with multi-user support.
Follows NASA JPL Rule #4 (Modular) and Rule #9 (Type Hints).
"""

from __future__ import annotations
from typing import List, Optional
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.base import ChunkRepository, SearchResult
from ingestforge.storage.postgres.models import Base, PostgresChunk, DocumentShare
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class PostgresRepository(ChunkRepository):
    """
    PostgreSQL-based chunk storage.

    Handles persistence of document chunks in a relational database.
    Designed for scalability and multi-user collaboration.
    """

    def __init__(self, connection_string: str):
        """Initialize Postgres connection and session factory."""
        assert connection_string, "Connection string cannot be empty"
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create extension and schema."""
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        Base.metadata.create_all(bind=self.engine)

    def add_chunk(self, chunk: ChunkRecord) -> bool:
        """Persist a single chunk to Postgres."""
        with self.SessionLocal() as session:
            try:
                db_chunk = self._map_to_db(chunk)
                session.add(db_chunk)
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to add chunk {chunk.chunk_id}: {e}")
                return False

    def add_chunks(self, chunks: List[ChunkRecord]) -> int:
        """Bulk insert multiple chunks for performance."""
        with self.SessionLocal() as session:
            try:
                db_chunks = [self._map_to_db(c) for c in chunks]
                session.add_all(db_chunks)
                session.commit()
                return len(chunks)
            except Exception as e:
                session.rollback()
                logger.error(f"Bulk insert failed: {e}")
                return 0

    def get_chunk(
        self, chunk_id: str, user_id: Optional[int] = None
    ) -> Optional[ChunkRecord]:
        """Retrieve a chunk by its ID with ACL check."""
        with self.SessionLocal() as session:
            query = session.query(PostgresChunk).filter(
                PostgresChunk.chunk_id == chunk_id
            )
            if user_id is not None:
                # Filter for owned OR shared documents
                shared_docs = select(DocumentShare.document_id).where(
                    DocumentShare.user_id == user_id
                )
                query = query.filter(
                    (PostgresChunk.owner_id == user_id)
                    | (PostgresChunk.document_id.in_(shared_docs))
                )

            db_chunk = query.first()
            return self._map_to_record(db_chunk) if db_chunk else None

    def verify_chunk_exists(self, chunk_id: str) -> bool:
        """Fast existence check."""
        with self.SessionLocal() as session:
            return (
                session.query(PostgresChunk.id)
                .filter(PostgresChunk.chunk_id == chunk_id)
                .count()
                > 0
            )

    def get_chunks_by_document(
        self, document_id: str, user_id: Optional[int] = None
    ) -> List[ChunkRecord]:
        """Get all chunks associated with a document with ACL check."""
        with self.SessionLocal() as session:
            query = session.query(PostgresChunk).filter(
                PostgresChunk.document_id == document_id
            )
            if user_id is not None:
                shared_docs = select(DocumentShare.document_id).where(
                    DocumentShare.user_id == user_id
                )
                query = query.filter(
                    (PostgresChunk.owner_id == user_id)
                    | (PostgresChunk.document_id.in_(shared_docs))
                )

            db_chunks = query.all()
            return [self._map_to_record(c) for c in db_chunks]

    def delete_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from storage."""
        with self.SessionLocal() as session:
            result = (
                session.query(PostgresChunk)
                .filter(PostgresChunk.chunk_id == chunk_id)
                .delete()
            )
            session.commit()
            return result > 0

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a specific document."""
        with self.SessionLocal() as session:
            count = (
                session.query(PostgresChunk)
                .filter(PostgresChunk.document_id == document_id)
                .delete()
            )
            session.commit()
            return count

    def search(
        self, query: str, top_k: int = 10, user_id: Optional[int] = None, **kwargs
    ) -> List[SearchResult]:
        """Perform semantic search using pgvector with ACL filtering."""
        if not query:
            return []
        with self.SessionLocal() as session:
            try:
                query_vector = self._get_embedding(query)
                stmt = select(
                    PostgresChunk,
                    (1 - PostgresChunk.embedding.cosine_distance(query_vector)).label(
                        "score"
                    ),
                )
                if user_id is not None:
                    shared_docs = select(DocumentShare.document_id).where(
                        DocumentShare.user_id == user_id
                    )
                    stmt = stmt.where(
                        (PostgresChunk.owner_id == user_id)
                        | (PostgresChunk.document_id.in_(shared_docs))
                    )

                stmt = stmt.order_by(
                    PostgresChunk.embedding.cosine_distance(query_vector)
                ).limit(top_k)
                results = session.execute(stmt).all()
                return [
                    SearchResult.from_chunk(
                        self._map_to_record(r[0]), score=float(r[1])
                    )
                    for r in results
                ]
            except Exception as e:
                logger.error(f"Postgres ACL search failed: {e}")
                return []

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for query string locally."""
        if not hasattr(self, "_st_model") or self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._st_model.encode(text).tolist()

    def _map_to_db(self, record: ChunkRecord) -> PostgresChunk:
        """Convert ChunkRecord to SQLAlchemy model."""
        return PostgresChunk(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            content=record.content,
            embedding=record.embedding,
            chunk_type=record.chunk_type,
            section_title=record.section_title,
            source_file=record.source_file,
            word_count=record.word_count,
            page_start=record.page_start,
            page_end=record.page_end,
            library=record.library,
            extra_metadata=record.metadata,
        )

    def _map_to_record(self, db_chunk: PostgresChunk) -> ChunkRecord:
        """Convert SQLAlchemy model back to ChunkRecord."""
        return ChunkRecord(
            chunk_id=db_chunk.chunk_id,
            document_id=db_chunk.document_id,
            content=db_chunk.content,
            chunk_type=db_chunk.chunk_type,
            section_title=db_chunk.section_title,
            source_file=db_chunk.source_file,
            word_count=db_chunk.word_count,
            page_start=db_chunk.page_start,
            page_end=db_chunk.page_end,
            library=db_chunk.library,
            metadata=db_chunk.extra_metadata or {},
        )
