"""Vision Research Tools for Autonomous Agents.

Provides tools that allow the agent to 'see' and analyze visual
elements like figures, charts, and diagrams.

Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

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
from ingestforge.ingest.ocr.vlm_processor import VLMProcessor
from ingestforge.storage.base import ChunkRepository

logger = get_logger(__name__)


def describe_figure(
    storage: ChunkRepository,
    vlm: VLMProcessor,
    chunk_id: str,
) -> ToolOutput:
    """Analyze a visual chunk and return a textual description."""
    if not chunk_id:
        return ToolOutput(
            status=ToolResult.ERROR, data=None, error_message="Missing chunk_id"
        )

    try:
        # 1. Retrieve the chunk metadata to find the image path
        chunk = storage.get_chunk(chunk_id)
        if not chunk:
            return ToolOutput(
                status=ToolResult.ERROR,
                data=None,
                error_message=f"Chunk {chunk_id} not found",
            )

        # Logic: Extract image path from metadata (Added in US-VISION.1)
        image_path_str = chunk.metadata.get("image_path")
        if not image_path_str:
            return ToolOutput(
                status=ToolResult.ERROR,
                data=None,
                error_message="No visual data associated with this chunk",
            )

        image_path = Path(image_path_str)

        # 2. Trigger local VLM analysis
        description = vlm.describe_image(image_path)

        logger.info(f"Visual Analysis Complete for {chunk_id}: {description[:30]}...")
        return ToolOutput(status=ToolResult.SUCCESS, data=description)

    except Exception as e:
        logger.error(f"Vision tool failed for {chunk_id}: {e}")
        return ToolOutput(status=ToolResult.ERROR, data=None, error_message=str(e))


def register_vision_tools(
    registry: ToolRegistry,
    storage: ChunkRepository,
    vlm: VLMProcessor,
) -> int:
    """Register all vision tools with the registry."""
    count = 0

    # Wrapper handles missing arguments gracefully
    def _describe_figure_wrapper(chunk_id: str = "", **kwargs: Any) -> ToolOutput:
        if not chunk_id:
            return ToolOutput(
                status=ToolResult.ERROR,
                data=None,
                error_message="Missing required argument: chunk_id",
            )
        return describe_figure(storage, vlm, chunk_id)

    if registry.register(
        name="describe_figure",
        fn=_describe_figure_wrapper,
        description="Extract a textual description from an image-based chunk (figure, chart, table).",
        category=ToolCategory.ANALYZE,
        parameters=[
            ToolParameter(
                name="chunk_id",
                param_type="str",
                description="The unique ID of the visual chunk to analyze.",
                required=True,
            )
        ],
    ):
        count += 1

    logger.info(f"Registered {count} vision research tools")
    return count
