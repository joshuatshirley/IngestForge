"""
Spatial Linkage Engine.

Spatial Linkage Engine
Links visual evidence (bounding boxes) to knowledge graph nodes.

JPL Compliance:
- Rule #2: Bounded loops (MAX_NODES_SEARCHED, MAX_VARIATIONS_CHECKED).
- Rule #4: All functions < 60 lines.
- Rule #9: 100% type hints.
"""

from typing import List, Optional
from ingestforge.enrichment.knowledge_graph import KnowledgeGraph
from ingestforge.core.pipeline.artifacts import IFArtifact
from ingestforge.processors.vision.vision_processor import IFImageArtifact, BoundingBox
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Safety limits
MAX_ARTIFACTS_PROCESSED = 1000
MAX_BOUNDING_BOXES_PER_ARTIFACT = 100
MAX_NODES_SEARCHED = 1000
MAX_VARIATIONS_CHECKED = 50


class SpatialLinkageEngine:
    """Bridges the gap between visual coordinates and semantic graph nodes."""

    def __init__(self, graph: KnowledgeGraph):
        """Initialize with a reference to the Knowledge Graph."""
        self.graph = graph

    def link_visual_artifacts(self, artifacts: List[IFArtifact]) -> int:
        """
        Links bounding boxes from image artifacts to existing graph nodes.

        Rule #2: Bounded iteration over artifacts and boxes.
        Rule #4: Orchestration logic.
        """
        link_count = 0

        # JPL Rule #2: Bound artifacts
        for artifact in artifacts[:MAX_ARTIFACTS_PROCESSED]:
            if not isinstance(artifact, IFImageArtifact):
                continue

            if not artifact.has_bounding_boxes:
                continue

            link_count += self._process_image_artifact(artifact)

        return link_count

    def _process_image_artifact(self, artifact: IFImageArtifact) -> int:
        """Processes a single image artifact and its boxes."""
        artifact_links = 0
        boxes = artifact.bounding_boxes

        # JPL Rule #2: Bound boxes
        for box in boxes[:MAX_BOUNDING_BOXES_PER_ARTIFACT]:
            if not box.label:
                continue

            # Try to find a matching node in the graph
            node_id = self._find_matching_node(box.label)
            if node_id:
                self._apply_link(node_id, artifact.artifact_id, box)
                artifact_links += 1

        return artifact_links

    def _find_matching_node(self, label: str) -> Optional[str]:
        """
        Finds a graph node that matches the given label.

        JPL Rule #2: Bounded graph searching.
        """
        if not self.graph.is_available():
            return None

        label_lower = label.lower()

        # JPL Rule #2: Strictly bound search depth
        for i, (node_id, attrs) in enumerate(self.graph._graph.nodes(data=True)):
            if i >= MAX_NODES_SEARCHED:
                break

            node_label = attrs.get("label", "").lower()
            if node_label == label_lower:
                return str(node_id)

            variations = attrs.get("variations", [])
            # JPL Rule #2: Bound inner variation loop
            for v in variations[:MAX_VARIATIONS_CHECKED]:
                if v.lower() == label_lower:
                    return str(node_id)

        return None

    def _apply_link(self, node_id: str, image_id: str, box: BoundingBox) -> None:
        """Persists the spatial link onto the graph node."""
        spatial_data = {
            "image_id": image_id,
            "x": box.x,
            "y": box.y,
            "width": box.width,
            "height": box.height,
            "confidence": box.confidence,
            "source_type": "vlm_extraction",
        }

        # Delegate persistence to KnowledgeGraph
        success = self.graph.add_spatial_link(node_id, spatial_data)
        if success:
            logger.debug(f"Linked node {node_id} to region in {image_id}")
        else:
            logger.warning(f"Failed to link node {node_id} (not found in graph)")
