"""Prerequisites detection - Analyze corpus for prerequisite concepts."""

import re
from pathlib import Path
from typing import List, Any, Set
from dataclasses import dataclass

from ingestforge.core.logging import get_logger
from ingestforge.cli.research._utils import (
    load_project,
    split_sentences,
    get_content,
)

logger = get_logger(__name__)

PREREQUISITE_INDICATORS = [
    "requires knowledge of",
    "builds upon",
    "assumes familiarity with",
    "prerequisite",
    "depends on",
    "necessary to understand",
    "background in",
    "foundation in",
    "prior knowledge of",
    "before learning",
    "first understand",
    "requires understanding",
    "based on",
    "extends",
    "builds on",
]


@dataclass
class PrerequisiteLink:
    """A prerequisite relationship between concepts."""

    concept: str
    prerequisite: str
    evidence: str
    source: str


@dataclass
class PrerequisiteResult:
    """Result of prerequisite analysis."""

    topic: str
    prerequisites: List[PrerequisiteLink]
    concept_hierarchy: List[str]  # Ordered from foundational to advanced
    mermaid_diagram: str


def detect_prerequisites(
    topic: str,
    project_path: Path,
    top_k: int = 30,
) -> PrerequisiteResult:
    """
    Analyze corpus for prerequisite concepts.

    Args:
        topic: The topic to analyze
        project_path: Path to the project directory
        top_k: Number of results to retrieve

    Returns:
        PrerequisiteResult with prerequisite links and hierarchy
    """
    config, repository = load_project(project_path)

    results = repository.search(topic, top_k=top_k)

    if not results:
        raise ValueError(
            f"No information found about '{topic}'. "
            "Make sure you have indexed relevant documents."
        )

    links = _extract_prerequisite_links(results, topic)
    hierarchy = _build_hierarchy(links, topic)
    diagram = _generate_mermaid(links, topic)

    return PrerequisiteResult(
        topic=topic,
        prerequisites=links,
        concept_hierarchy=hierarchy,
        mermaid_diagram=diagram,
    )


def _create_link_if_valid(
    topic: str,
    prereq: str,
    sentence: str,
    source_name: str,
    seen: Set[str],
    links: List[PrerequisiteLink],
) -> None:
    """
    Create prerequisite link if valid and not duplicate.

    Rule #1: Early return eliminates nesting
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        topic: Main concept
        prereq: Prerequisite concept
        sentence: Evidence sentence
        source_name: Source document name
        seen: Set of seen prerequisite keys
        links: List to append link to (mutated)
    """
    if not prereq or len(prereq) <= 2:
        return
    key = f"{topic}:{prereq}"
    if key in seen:
        return

    # Valid and unique - add link
    seen.add(key)
    links.append(
        PrerequisiteLink(
            concept=topic,
            prerequisite=prereq,
            evidence=sentence.strip(),
            source=source_name,
        )
    )


def _process_indicator_match(
    sentence: str,
    indicator: str,
    topic: str,
    source_name: str,
    seen: Set[str],
    links: List[PrerequisiteLink],
) -> None:
    """
    Process matched indicator in sentence.

    Rule #1: Helper function eliminates nesting
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        sentence: Sentence containing indicator
        indicator: Matched prerequisite indicator
        topic: Main concept
        source_name: Source document name
        seen: Set of seen prerequisite keys
        links: List to append links to (mutated)
    """
    prereq = _extract_prerequisite_concept(sentence, indicator)
    _create_link_if_valid(topic, prereq, sentence, source_name, seen, links)


def _process_sentence_for_links(
    sentence: str,
    topic: str,
    source_name: str,
    seen: Set[str],
    links: List[PrerequisiteLink],
) -> None:
    """
    Process sentence looking for prerequisite indicators.

    Rule #1: Reduced nesting (max 2 levels)
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        sentence: Sentence to process
        topic: Main concept
        source_name: Source document name
        seen: Set of seen prerequisite keys
        links: List to append links to (mutated)
    """
    sentence_lower = sentence.lower()
    for indicator in PREREQUISITE_INDICATORS:
        if indicator in sentence_lower:
            _process_indicator_match(
                sentence, indicator, topic, source_name, seen, links
            )


def _get_source_name(result: Any) -> str:
    """
    Extract source name from result.

    Rule #1: Early return eliminates nesting
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        result: Search result object

    Returns:
        Source document name
    """
    if not hasattr(result, "source_file"):
        return "Unknown"

    source = result.source_file
    if source == "Unknown":
        return source

    return Path(source).stem


def _process_result_for_links(
    result: Any, topic: str, seen: Set[str], links: List[PrerequisiteLink]
) -> None:
    """
    Process single result for prerequisite links.

    Rule #1: Reduced nesting (max 2 levels)
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        result: Search result to process
        topic: Main concept
        seen: Set of seen prerequisite keys
        links: List to append links to (mutated)
    """
    content = get_content(result)
    source_name = _get_source_name(result)
    sentences = split_sentences(content)
    for sentence in sentences:
        _process_sentence_for_links(sentence, topic, source_name, seen, links)


def _extract_prerequisite_links(results: Any, topic: str) -> List[PrerequisiteLink]:
    """
    Extract prerequisite relationships from results.

    Rule #1: Reduced nesting (max 1 level)
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        results: Search results to process
        topic: Main concept to find prerequisites for

    Returns:
        List of prerequisite links (max 15)
    """
    links: List[PrerequisiteLink] = []
    seen: Set[str] = set()
    for result in results:
        _process_result_for_links(result, topic, seen, links)

    return links[:15]


def _extract_prerequisite_concept(sentence: str, indicator: str) -> str:
    """Extract the prerequisite concept from a sentence."""
    lower = sentence.lower()
    idx = lower.find(indicator)
    if idx == -1:
        return ""

    # Get text after the indicator
    after = sentence[idx + len(indicator) :].strip()

    # Take up to the next punctuation or conjunction
    match = re.match(r"([^.!?,;]+)", after)
    if match:
        concept = match.group(1).strip()
        # Clean up
        concept = re.sub(r"^(a |an |the )", "", concept, flags=re.IGNORECASE)
        return concept[:80]

    return ""


def _build_hierarchy(links: List[PrerequisiteLink], topic: str) -> List[str]:
    """Build concept hierarchy from prerequisite links."""
    # Simple topological ordering
    prerequisites = set()
    for link in links:
        prerequisites.add(link.prerequisite)

    # Order: prerequisites first, then the topic
    hierarchy = sorted(prerequisites)
    if topic not in hierarchy:
        hierarchy.append(topic)

    return hierarchy


def _generate_mermaid(links: List[PrerequisiteLink], topic: str) -> str:
    """Generate Mermaid flowchart of prerequisites."""
    if not links:
        return ""

    lines = ["```mermaid", "graph TD"]

    # Add topic node
    safe_topic = re.sub(r"[^a-zA-Z0-9]", "_", topic)
    lines.append(f"    {safe_topic}[**{topic}**]")

    seen_nodes = {safe_topic}

    for link in links[:15]:
        safe_prereq = re.sub(r"[^a-zA-Z0-9]", "_", link.prerequisite)

        if safe_prereq not in seen_nodes:
            lines.append(f"    {safe_prereq}[{link.prerequisite}]")
            seen_nodes.add(safe_prereq)

        lines.append(f"    {safe_prereq} --> {safe_topic}")

    lines.append("```")
    return "\n".join(lines)


def format_prerequisites_markdown(result: PrerequisiteResult) -> str:
    """Format prerequisites result as markdown."""
    lines = [
        f"# Prerequisites: {result.topic}",
        "",
        "*Concepts you should understand before studying this topic*",
        "",
    ]

    # Hierarchy
    if result.concept_hierarchy:
        lines.extend(["## Learning Path", ""])
        for i, concept in enumerate(result.concept_hierarchy, 1):
            marker = " (target)" if concept == result.topic else ""
            lines.append(f"{i}. {concept}{marker}")
        lines.append("")

    # Diagram
    if result.mermaid_diagram:
        lines.extend(["## Dependency Graph", "", result.mermaid_diagram, ""])

    # Detailed links
    if result.prerequisites:
        lines.extend(["## Evidence", ""])
        for link in result.prerequisites:
            lines.append(f"- **{link.prerequisite}**")
            lines.append(f"  {link.evidence[:200]}")
            lines.append(f"  *{link.source}*")
            lines.append("")

    return "\n".join(lines)
