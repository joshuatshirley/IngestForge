"""Layout Matcher.

Provides spatial join logic to associate text captions with visual elements.
Follows NASA JPL Rule #4 (Modular) and Rule #9 (Type Hints).
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
from dataclasses import dataclass


# Assuming OCRBlock definition from ingestforge.ocr.models (Task 5.2.1 Dependency)
@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


class LayoutMatcher:
    """Logic for linking text blocks to adjacent images/tables."""

    def __init__(self, proximity_threshold: int = 150):
        self.threshold = proximity_threshold

    def find_nearest_caption(
        self, image_box: BoundingBox, text_blocks: List[Tuple[BoundingBox, str]]
    ) -> Optional[str]:
        """Find the text block closest to the given image box.

        Rule #1: Linear control flow with early returns.
        Rule #2: Bounded loop (text_blocks).
        """
        if not text_blocks:
            return None

        best_match: Optional[str] = None
        min_dist = float("inf")
        MAX_SEARCH_BLOCKS = 100

        img_center = image_box.center

        for box, text in text_blocks[:MAX_SEARCH_BLOCKS]:
            dist = self._calculate_distance(img_center, box.center)

            if dist < min_dist and dist <= self.threshold:
                min_dist = dist
                best_match = text

        return best_match

    def _calculate_distance(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_caption_like(self, text: str) -> bool:
        """Heuristic to determine if a text block is a figure/table caption."""
        lower_text = text.lower().strip()
        keywords = ["figure", "fig.", "table", "chart", "diagram"]

        return any(lower_text.startswith(kw) for kw in keywords)
