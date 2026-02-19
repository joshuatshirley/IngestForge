"""Thesis-Evidence Mapper for structured research exports.

Maps source chunks to outline points using LLM-based relevance scoring.
Enables generation of well-structured research documents with citations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.llm.base import LLMClient, GenerationConfig

logger = get_logger(__name__)
MAX_OUTLINE_POINTS = 50
MAX_CHUNKS_PER_POINT = 20
MAX_CHUNKS_TO_EVALUATE = 100


@dataclass
class OutlinePoint:
    """A single point in the thesis outline."""

    id: str
    title: str
    description: str = ""
    level: int = 1  # 1 = main point, 2 = subpoint, 3 = detail
    parent_id: Optional[str] = None

    def to_prompt_text(self) -> str:
        """Format point for LLM prompt."""
        prefix = "#" * self.level
        result = f"{prefix} {self.title}"
        if self.description:
            result += f"\n{self.description}"
        return result


@dataclass
class EvidenceMatch:
    """A chunk matched to an outline point."""

    chunk_id: str
    chunk_content: str
    point_id: str
    relevance_score: float  # 0.0 to 1.0
    source_file: str = ""
    page: Optional[int] = None
    section: str = ""

    def to_citation(self) -> str:
        """Format as citation reference."""
        parts = [self.source_file]
        if self.page:
            parts.append(f"p. {self.page}")
        if self.section:
            parts.append(self.section)
        return ", ".join(parts)


@dataclass
class MappedOutline:
    """Complete outline with evidence mappings."""

    title: str
    points: List[OutlinePoint] = field(default_factory=list)
    evidence: dict[str, List[EvidenceMatch]] = field(default_factory=dict)

    def get_evidence_for_point(self, point_id: str) -> List[EvidenceMatch]:
        """Get all evidence matched to a point."""
        return self.evidence.get(point_id, [])

    def get_total_evidence_count(self) -> int:
        """Get total number of evidence matches."""
        return sum(len(matches) for matches in self.evidence.values())


class OutlineMapper:
    """Maps source chunks to thesis outline points using LLM.

    Uses semantic understanding to match evidence to outline points,
    enabling generation of well-cited research documents.
    """

    def __init__(
        self,
        config: Config,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        """Initialize mapper.

        Args:
            config: IngestForge configuration
            llm_client: Optional LLM client (auto-created if None)
        """
        self.config = config
        self._llm = llm_client

    @property
    def llm(self) -> LLMClient:
        """Lazy-load LLM client."""
        if self._llm is None:
            from ingestforge.llm.factory import get_best_available_client

            self._llm = get_best_available_client(self.config)
            if self._llm is None:
                raise RuntimeError("No LLM provider available for outline mapping")
        return self._llm

    def parse_outline(
        self, outline_text: str, title: str = "Research"
    ) -> MappedOutline:
        """Parse outline text into structured points.

        Args:
            outline_text: Markdown-style outline
            title: Document title

        Returns:
            MappedOutline with parsed points
        """
        points = self._parse_outline_points(outline_text)
        return MappedOutline(title=title, points=points)

    def _parse_outline_points(self, text: str) -> List[OutlinePoint]:
        """Parse outline text into points.

        Args:
            text: Markdown-style outline

        Returns:
            List of OutlinePoint instances
        """
        points: List[OutlinePoint] = []
        lines = text.strip().split("\n")
        parent_stack: List[str] = []

        for idx, line in enumerate(lines[:MAX_OUTLINE_POINTS]):
            line = line.strip()
            if not line:
                continue

            point = self._parse_single_line(line, idx, parent_stack)
            if point:
                points.append(point)
                self._update_parent_stack(parent_stack, point)

        return points

    def _parse_single_line(
        self, line: str, idx: int, parent_stack: List[str]
    ) -> Optional[OutlinePoint]:
        """Parse a single outline line.

        Args:
            line: Line text
            idx: Line index
            parent_stack: Current parent hierarchy

        Returns:
            OutlinePoint or None if not a valid point
        """
        # Determine level and title from leading markers
        level, title = self._extract_level_and_title(line)

        if not title:
            return None

        parent_id = parent_stack[level - 2] if level > 1 and parent_stack else None

        return OutlinePoint(
            id=f"point_{idx}",
            title=title,
            level=level,
            parent_id=parent_id,
        )

    def _extract_level_and_title(self, line: str) -> Tuple[int, str]:
        """Extract outline level and title from line.

        Args:
            line: Line text

        Returns:
            Tuple of (level, title)
        """
        if line.startswith("###"):
            return 3, line[3:].strip()
        if line.startswith("##"):
            return 2, line[2:].strip()
        if line.startswith("#"):
            return 1, line[1:].strip()
        if line.startswith("-") or line.startswith("*"):
            return 2, line[1:].strip()

        return 1, line

    def _update_parent_stack(
        self, parent_stack: List[str], point: OutlinePoint
    ) -> None:
        """Update parent stack after adding a point.

        Args:
            parent_stack: Stack to update (modified in place)
            point: Point that was added
        """
        while len(parent_stack) >= point.level:
            parent_stack.pop()
        parent_stack.append(point.id)

    def map_chunks_to_outline(
        self,
        outline: MappedOutline,
        chunks: List[Any],
        min_relevance: float = 0.5,
    ) -> MappedOutline:
        """Map chunks to outline points using LLM.

        Args:
            outline: Parsed outline structure
            chunks: List of chunk objects
            min_relevance: Minimum relevance score to include (0.0-1.0)

        Returns:
            Outline with evidence mappings populated
        """
        chunks_to_eval = chunks[:MAX_CHUNKS_TO_EVALUATE]

        for point in outline.points:
            matches = self._find_evidence_for_point(
                point, chunks_to_eval, min_relevance
            )
            outline.evidence[point.id] = matches[:MAX_CHUNKS_PER_POINT]

        return outline

    def _find_evidence_for_point(
        self,
        point: OutlinePoint,
        chunks: List[Any],
        min_relevance: float,
    ) -> List[EvidenceMatch]:
        """Find chunks that support an outline point.

        Args:
            point: Outline point to find evidence for
            chunks: Available chunks
            min_relevance: Minimum relevance threshold

        Returns:
            List of EvidenceMatch instances, sorted by relevance
        """
        matches: List[EvidenceMatch] = []

        for chunk in chunks:
            score = self._score_chunk_relevance(point, chunk)
            if score < min_relevance:
                continue

            match = self._create_evidence_match(chunk, point.id, score)
            matches.append(match)

        # Sort by relevance, highest first
        matches.sort(key=lambda m: m.relevance_score, reverse=True)
        return matches

    def _score_chunk_relevance(self, point: OutlinePoint, chunk: Any) -> float:
        """Score how relevant a chunk is to an outline point.

        Args:
            point: Outline point
            chunk: Chunk to evaluate

        Returns:
            Relevance score from 0.0 to 1.0
        """
        chunk_text = self._extract_chunk_text(chunk)

        prompt = self._build_relevance_prompt(point.title, chunk_text)
        gen_config = GenerationConfig(temperature=0.1, max_tokens=50)

        try:
            response = self.llm.generate(prompt, config=gen_config)
            return self._parse_relevance_response(response)
        except Exception as e:
            logger.warning(
                f"Failed to score chunk relevance for point '{point.title}': {e}"
            )
            return 0.0

    def _build_relevance_prompt(self, point_title: str, chunk_text: str) -> str:
        """Build prompt for relevance scoring.

        Args:
            point_title: Outline point title
            chunk_text: Chunk content

        Returns:
            Formatted prompt string
        """
        return f"""Rate how relevant this text passage is to the topic: "{point_title}"

Text passage:
{chunk_text[:500]}

Rate relevance from 0 to 10 where:
- 0-2: Not relevant
- 3-4: Slightly relevant
- 5-6: Moderately relevant
- 7-8: Very relevant
- 9-10: Highly relevant, directly addresses the topic

Respond with just the number (0-10):"""

    def _parse_relevance_response(self, response: str) -> float:
        """Parse LLM relevance score response.

        Args:
            response: LLM response text

        Returns:
            Normalized score from 0.0 to 1.0
        """
        try:
            # Extract first number from response
            import re

            match = re.search(r"\d+", response.strip())
            if match:
                score = int(match.group())
                return min(score / 10.0, 1.0)
        except ValueError:
            logger.exception(
                f"Error parsing relevance score from response: {response[:100]}"
            )
        return 0.0

    def _extract_chunk_text(self, chunk: Any) -> str:
        """Extract text content from chunk.

        Args:
            chunk: Chunk object

        Returns:
            Text content
        """
        if hasattr(chunk, "content"):
            return chunk.content
        if hasattr(chunk, "text"):
            return chunk.text
        if isinstance(chunk, dict):
            return chunk.get("content", chunk.get("text", str(chunk)))
        return str(chunk)

    def _create_evidence_match(
        self, chunk: Any, point_id: str, score: float
    ) -> EvidenceMatch:
        """Create EvidenceMatch from chunk.

        Args:
            chunk: Source chunk
            point_id: ID of matched outline point
            score: Relevance score

        Returns:
            EvidenceMatch instance
        """
        chunk_id = getattr(chunk, "chunk_id", getattr(chunk, "id", "unknown"))
        content = self._extract_chunk_text(chunk)
        source = getattr(chunk, "source_file", "")
        page = getattr(chunk, "page", None)
        section = getattr(chunk, "section_title", "")

        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id", chunk.get("id", "unknown"))
            source = chunk.get("source_file", "")
            page = chunk.get("page")
            section = chunk.get("section_title", "")

        return EvidenceMatch(
            chunk_id=str(chunk_id),
            chunk_content=content,
            point_id=point_id,
            relevance_score=score,
            source_file=source,
            page=page,
            section=section,
        )
