"""Web Research Tools for Autonomous Agents.

Provides tools for searching the open web and extracting content
to be ingested into the IngestForge knowledge base."""

from __future__ import annotations

from typing import Any, List

from ingestforge.agent.react_engine import ToolOutput, ToolResult
from ingestforge.agent.tool_registry import (
    ToolCategory,
    ToolParameter,
    ToolRegistry,
)
from ingestforge.core.logging import get_logger
from ingestforge.ingest.web_search import WebSearcher

logger = get_logger(__name__)
MAX_WEB_RESULTS = 10


def search_web(
    query: str,
    num_results: int = 5,
) -> ToolOutput:
    """Search the web for information."""
    if not query or not query.strip():
        return ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message="Query cannot be empty",
        )

    if num_results < 1:
        num_results = 1
    if num_results > MAX_WEB_RESULTS:
        num_results = MAX_WEB_RESULTS

    try:
        searcher = WebSearcher()
        session = searcher.search(query, max_results=num_results)
        results = session.results

        if not results:
            return ToolOutput(
                status=ToolResult.SUCCESS,
                data="No web results found for query",
            )

        formatted = _format_web_results(results)

        logger.info(f"Web search completed: {len(results)} results for query")
        return ToolOutput(
            status=ToolResult.SUCCESS,
            data=formatted,
        )

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message=f"Web search error: {str(e)}",
        )


def _format_web_results(results: List[Any]) -> str:
    """Format web search results for display."""
    header = "Web Search Results:\n"
    lines = [header]

    for i, result in enumerate(results, 1):
        # Result typically has title, link, snippet
        title = getattr(result, "title", "No Title")
        link = getattr(result, "link", "No Link")
        snippet = getattr(result, "snippet", "No snippet available")

        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {link}")
        lines.append(f"   Summary: {snippet}\n")

    return "\n".join(lines)


def register_web_tools(registry: ToolRegistry) -> int:
    """Register all web research tools with the registry."""
    count = 0

    # Use default="" to prevent lambda crash; validation inside function handles empty
    if registry.register(
        name="search_web",
        fn=lambda query="", num_results=5, **kwargs: search_web(query, num_results),
        description="Search the internet for up-to-date information",
        category=ToolCategory.SEARCH,
        parameters=[
            ToolParameter(
                name="query",
                param_type="str",
                description="Search query text",
                required=True,
            ),
            ToolParameter(
                name="num_results",
                param_type="int",
                description="Number of results to return",
                required=False,
                default=5,
            ),
        ],
    ):
        count += 1

    logger.info(f"Registered {count} web research tools")
    return count
