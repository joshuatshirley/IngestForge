"""Postgres SQLAlchemy Models.

Defines the database schema for multi-user shared corpora.
Follows NASA JPL Rule #9 (Type Hints) and Rule #10 (Static Structure).
"""

from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    Text,
    Float,
    JSON,
    ForeignKey,
    Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class User(Base):
    """User accounts for authentication and ownership."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DocumentShare(Base):
    """Many-to-many mapping for shared document access."""

    __tablename__ = "document_shares"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(255), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    shared_at = Column(DateTime, default=datetime.utcnow)


class PostgresChunk(Base):
    """A semantic chunk of document data stored in Postgres."""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(64), unique=True, nullable=False, index=True)
    document_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=True)  # Added for US-COLLAB.2

    # Metadata
    chunk_type = Column(String(50), default="content")
    section_title = Column(String(255), nullable=True)
    source_file = Column(String(512), nullable=True)
    word_count = Column(Integer, default=0)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    library = Column(String(100), default="default", index=True)

    # JSON metadata for extensible properties (Rule #10)
    extra_metadata = Column(JSON, default=dict)

    # Ownership
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)

    # Verification Status (Sprint 5 Meta)
    is_verified = Column(Boolean, default=False)
    verification_score = Column(Float, nullable=True)

    def to_dict(self) -> dict:
        """Convert model to dictionary for API response."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "section_title": self.section_title,
            "source_file": self.source_file,
            "page_start": self.page_start,
            "library": self.library,
            "is_verified": self.is_verified,
        }
