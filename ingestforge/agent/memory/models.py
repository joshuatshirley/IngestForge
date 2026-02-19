"""Agent Memory Models.

Defines the SQL structure for persistent fact storage.
Follows NASA JPL Rule #9 (Type Hints) and Rule #10 (Static Structure).
"""

from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class AgentFact(Base):
    """A consolidated finding learned by the agent during a mission."""

    __tablename__ = "agent_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fact_text = Column(Text, nullable=False)
    evidence_chunk_id = Column(String(64), nullable=True)  # ID from ChromaDB
    source_title = Column(String(255), nullable=True)

    # Metadata for retrieval
    category = Column(String(100), nullable=True)
    confidence_score = Column(Float, default=1.0)

    # Persistence audit trail
    mission_id = Column(String(64), nullable=False)
    learned_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert model to dictionary for API response."""
        return {
            "id": self.id,
            "fact": self.fact_text,
            "source": self.source_title,
            "chunk_id": self.evidence_chunk_id,
            "category": self.category,
            "learned_at": self.learned_at.isoformat(),
        }
