"""Knowledge Base Tools for Autonomous Agents.

Provides tools that expose IngestForge core capabilities
to the ReAct agent for autonomous document research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ingestforge.agent.react_engine import ToolOutput, ToolResult
from ingestforge.agent.tool_registry import (
    ToolCategory,
    ToolParameter,
    ToolRegistry,
)
from ingestforge.core.logging import get_logger
from ingestforge.storage.base import ChunkRepository

logger = get_logger(__name__)
MAX_SEARCH_RESULTS = 50
MAX_CHUNK_CONTENT_LENGTH = 10000


def search_knowledge_base(
    storage: ChunkRepository,
    query: str,
    top_k: int = 5,
) -> ToolOutput:
    """Search the knowledge base for relevant chunks."""
    if not query or not query.strip():
        return ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message="Query cannot be empty",
        )

    if top_k < 1:
        top_k = 1
    if top_k > MAX_SEARCH_RESULTS:
        top_k = MAX_SEARCH_RESULTS

    try:
        results = storage.search(query, top_k=top_k)

        if not results:
            return ToolOutput(
                status=ToolResult.SUCCESS,
                data="No results found for query",
            )

        formatted = _format_search_results(results)

        logger.info(f"Search completed: {len(results)} results for query")
        return ToolOutput(
            status=ToolResult.SUCCESS,
            data=formatted,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message=f"Search error: {str(e)}",
        )


def ingest_document(
    pipeline: Any,
    path: str,
) -> ToolOutput:
    """Ingest a new document into the knowledge base."""
    if not path or not path.strip():
        return ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message="Path cannot be empty",
        )

    file_path = Path(path)

    if not file_path.exists():
        return ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message=f"File not found: {path}",
        )

    if not file_path.is_file():
        return ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message=f"Path is not a file: {path}",
        )

    try:
        result = pipeline.process_file(file_path)

        if not result.success:
            return ToolOutput(
                status=ToolResult.ERROR,
                data=None,
                error_message=result.error_message or "Processing failed",
            )

        message = _format_ingest_result(result)

        logger.info(f"Document ingested: {result.document_id}")
        return ToolOutput(
            status=ToolResult.SUCCESS,
            data=message,
        )

    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        return ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message=f"Ingest error: {str(e)}",
        )


def get_chunk_details(
    storage: ChunkRepository,
    chunk_id: str,
) -> ToolOutput:
    """Retrieve detailed information about a specific chunk."""
    if not chunk_id or not chunk_id.strip():
        return ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message="Chunk ID cannot be empty",
        )

    try:
        chunk = storage.get_chunk(chunk_id)

        if chunk is None:
            return ToolOutput(
                status=ToolResult.ERROR,
                data=None,
                error_message=f"Chunk not found: {chunk_id}",
            )

        formatted = _format_chunk_details(chunk)

        logger.debug(f"Retrieved chunk: {chunk_id}")
        return ToolOutput(
            status=ToolResult.SUCCESS,
            data=formatted,
        )

    except Exception as e:
        logger.error(f"Chunk retrieval failed: {e}")
        return ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message=f"Retrieval error: {str(e)}",
        )


def _format_search_results(results: list[Any]) -> str:
    """Format search results for display."""
    header = f"Found {len(results)} results:\n"
    lines = [header]

    for i, result in enumerate(results, 1):
        content = result.content[:200]
        if len(result.content) > 200:
            content += "..."

        lines.append(f"{i}. [{result.chunk_id}] (score: {result.score:.3f})")
        lines.append(f"   Source: {result.source_file}")
        lines.append(f"   Content: {content}\n")

    return "\n".join(lines)


def _format_ingest_result(result: Any) -> str:
    """Format pipeline result for display."""
    lines = [
        "Document ingested successfully:",
        f"  Document ID: {result.document_id}",
        f"  Chunks created: {result.chunks_created}",
        f"  Chunks indexed: {result.chunks_indexed}",
        f"  Processing time: {result.processing_time_sec:.2f}s",
    ]
    return "\n".join(lines)


def _format_chunk_details(chunk: Any) -> str:
    """Format chunk details for display."""
    content = chunk.content[:MAX_CHUNK_CONTENT_LENGTH]
    if len(chunk.content) > MAX_CHUNK_CONTENT_LENGTH:
        content += "..."

    lines = [
        f"Chunk ID: {chunk.chunk_id}",
        f"Document: {chunk.document_id}",
        f"Source: {chunk.source_file}",
        f"Section: {chunk.section_title}",
        f"Type: {chunk.chunk_type}",
        f"Words: {chunk.word_count}",
        f"Quality: {chunk.quality_score:.2f}",
        "",
        "Content:",
        content,
    ]

    if chunk.entities:
        lines.append("\nEntities: " + ", ".join(chunk.entities))

    if chunk.concepts:
        lines.append("Concepts: " + ", ".join(chunk.concepts))

    return "\n".join(lines)


def register_knowledge_tools(
    registry: ToolRegistry,
    storage: ChunkRepository,
    pipeline: Any,
) -> int:
    """Register all knowledge base tools with the registry."""
    count = 0

    # Use default="" to prevent lambda crash; validation inside function handles empty
    if registry.register(
        name="search_knowledge_base",
        fn=lambda query="", top_k=5, **kwargs: search_knowledge_base(
            storage, query, top_k
        ),
        description="Search the knowledge base for relevant information",
        category=ToolCategory.SEARCH,
        parameters=[
            ToolParameter(
                name="query",
                param_type="str",
                description="Search query text",
                required=True,
            ),
            ToolParameter(
                name="top_k",
                param_type="int",
                description="Number of results to return",
                required=False,
                default=5,
            ),
        ],
    ):
        count += 1

    if registry.register(
        name="ingest_document",
        fn=lambda path="", **kwargs: ingest_document(pipeline, path),
        description="Ingest a new document into the knowledge base",
        category=ToolCategory.UTILITY,
        parameters=[
            ToolParameter(
                name="path",
                param_type="str",
                description="Path to document file",
                required=True,
            ),
        ],
    ):
        count += 1

    if registry.register(
        name="get_chunk_details",
        fn=lambda chunk_id="", **kwargs: get_chunk_details(storage, chunk_id),
        description="Get detailed information about a specific chunk",
        category=ToolCategory.RETRIEVE,
        parameters=[
            ToolParameter(
                name="chunk_id",
                param_type="str",
                description="Unique chunk identifier",
                required=True,
            ),
        ],
    ):
        count += 1

    logger.info(f"Registered {count} knowledge base tools")
    return count
