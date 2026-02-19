"""
Army Doctrine RAG API Client.

HTTP client for querying the Army Doctrine RAG system to retrieve
relevant regulations, policies, and guidance for feature implementations.

The Army Doctrine RAG maintains:
- 350+ documents with 10,801 indexed chunks
- AR 601-210 (Level 1 authority) - Core recruiting regulation
- 151 USAREC policy messages (2020-2026)
- 7-stage query pipeline with authority boosting

API Endpoints:
- POST /api/v1/retrieve - Retrieve relevant chunks
- POST /api/v1/query - Generate answer with sources
- GET /api/v1/health - Health check
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from ingestforge.core.logging import get_logger


logger = get_logger(__name__)


@dataclass
class DoctrineResult:
    """A single result from the Army Doctrine RAG API."""

    document: str  # Document name (e.g., "AR 601-210")
    section: str  # Section reference (e.g., "Chapter 4-7")
    content: str  # Text content
    authority_level: int  # 1=Core Reg, 2=Policy, 3=Guide
    relevance_score: float  # Retrieval relevance score (0-1)
    source_type: str = ""  # Type of source (regulation, message, guide)
    chunk_id: str = ""  # Unique identifier for the chunk

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document": self.document,
            "section": self.section,
            "content": self.content,
            "authority_level": self.authority_level,
            "relevance_score": self.relevance_score,
            "source_type": self.source_type,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DoctrineResult":
        """Create from API response data."""
        # Handle different API response formats
        return cls(
            document=data.get("document", data.get("source", "")),
            section=data.get("section", data.get("section_title", "")),
            content=data.get("content", data.get("text", "")),
            authority_level=data.get("authority_level", data.get("authority", 4)),
            relevance_score=data.get("relevance_score", data.get("score", 0.0)),
            source_type=data.get("source_type", data.get("doc_type", "")),
            chunk_id=data.get("chunk_id", data.get("id", "")),
        )


class DoctrineAPIClient:
    """
    HTTP client for Army Doctrine RAG API.

    Queries the deployed RAG system for regulations and policies
    that may apply to feature implementations.

    Example:
        client = DoctrineAPIClient("http://localhost:8000")

        # Check availability
        if await client.health_check():
            # Retrieve relevant regulations
            results = await client.retrieve(
                "medical waiver requirements for Army enlistment",
                top_k=5,
            )
            for r in results:
                print(f"{r.document} {r.section}: {r.content[:100]}...")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout_seconds: int = 30,
    ):
        """
        Initialize the Doctrine API client.

        Args:
            base_url: Base URL of the Army Doctrine RAG API
            timeout_seconds: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._http_client = None

    async def _get_client(self):
        """Get or create HTTP client (lazy initialization)."""
        if self._http_client is None:
            try:
                import httpx

                self._http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout_seconds),
                )
            except ImportError:
                raise ImportError(
                    "httpx is required for DoctrineAPIClient.\n"
                    "Install with: pip install httpx"
                )
        return self._http_client

    async def close(self):
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def health_check(self) -> bool:
        """
        Check if the Doctrine API is available.

        Returns:
            True if API is healthy and responding
        """
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/v1/health")
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Doctrine API health check failed: {e}")
            return False

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        authority_filter: Optional[int] = None,
        doc_type_filter: Optional[str] = None,
    ) -> List[DoctrineResult]:
        """
        Retrieve relevant regulations and policies.

        Args:
            query: Natural language query describing the feature/requirement
            top_k: Maximum number of results to return
            authority_filter: If set, only return docs with this authority level
            doc_type_filter: If set, filter by document type (regulation, message, etc.)

        Returns:
            List of DoctrineResult ordered by relevance

        Example:
            results = await client.retrieve(
                "medical waiver tracking for MIRS integration",
                top_k=5,
            )
        """
        try:
            client = await self._get_client()

            payload = {
                "question": query,
                "top_k": top_k,
            }

            # Add optional filters
            if authority_filter is not None:
                payload["authority_level"] = authority_filter
            if doc_type_filter:
                payload["doc_type"] = doc_type_filter

            response = await client.post(
                f"{self.base_url}/api/v1/retrieve",
                json=payload,
            )

            if response.status_code != 200:
                logger.warning(
                    f"Doctrine API returned status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return []

            data = response.json()

            # Parse results - handle different response formats
            results = []
            sources = data.get("sources", data.get("results", data.get("chunks", [])))

            for source in sources:
                try:
                    result = DoctrineResult.from_api_response(source)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to parse doctrine result: {e}")
                    continue

            logger.info(
                f"Doctrine API retrieved {len(results)} results for query",
                query=query[:50],
            )
            return results

        except Exception as e:
            logger.error(f"Doctrine API retrieve failed: {e}")
            return []

    async def query_with_answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Query the Doctrine RAG and get a generated answer with sources.

        This uses the full RAG pipeline which generates an answer
        using the LLM based on retrieved context.

        Args:
            question: Natural language question
            top_k: Number of sources to retrieve

        Returns:
            Dict with 'answer' and 'sources' keys
        """
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self.base_url}/api/v1/query",
                json={
                    "question": question,
                    "top_k": top_k,
                },
            )

            if response.status_code != 200:
                logger.warning(f"Doctrine API query failed: {response.status_code}")
                return {"answer": "", "sources": []}

            data = response.json()

            # Parse sources
            sources = []
            for source in data.get("sources", []):
                try:
                    sources.append(DoctrineResult.from_api_response(source))
                except Exception:
                    continue

            return {
                "answer": data.get("answer", ""),
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"Doctrine API query failed: {e}")
            return {"answer": "", "sources": []}

    def retrieve_sync(
        self,
        query: str,
        top_k: int = 5,
        authority_filter: Optional[int] = None,
    ) -> List[DoctrineResult]:
        """
        Synchronous wrapper for retrieve().

        Convenience method for non-async contexts.

        Args:
            query: Natural language query
            top_k: Maximum number of results
            authority_filter: Optional authority level filter

        Returns:
            List of DoctrineResult
        """
        return asyncio.run(self.retrieve(query, top_k, authority_filter))

    def health_check_sync(self) -> bool:
        """Synchronous wrapper for health_check()."""
        return asyncio.run(self.health_check())


# Authority level descriptions for display
AUTHORITY_LEVELS = {
    1: "Core Regulation (AR)",
    2: "Policy Message (USAREC Msg)",
    3: "Guidance/Manual",
    4: "Reference Material",
    5: "Supporting Document",
}


def format_authority_level(level: int) -> str:
    """Format authority level for display."""
    return AUTHORITY_LEVELS.get(level, f"Level {level}")
