"""Educational resources - Curated catalog of educational platforms."""

from typing import List, Optional
from dataclasses import dataclass

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EducationalResource:
    """A resource from an educational platform."""

    title: str
    platform: str
    url: str
    difficulty_level: str  # beginner, intermediate, advanced
    topic_match: float  # 0-1 relevance score


# Curated educational platform catalog
EDUCATIONAL_PLATFORMS = {
    "mit_ocw": {
        "name": "MIT OpenCourseWare",
        "base_url": "https://ocw.mit.edu/search/?q=",
        "difficulty": "intermediate",
        "description": "Free MIT course materials",
    },
    "khan_academy": {
        "name": "Khan Academy",
        "base_url": "https://www.khanacademy.org/search?search_query=",
        "difficulty": "beginner",
        "description": "Free educational videos and exercises",
    },
    "openstax": {
        "name": "OpenStax",
        "base_url": "https://openstax.org/search?q=",
        "difficulty": "intermediate",
        "description": "Free peer-reviewed textbooks",
    },
    "coursera": {
        "name": "Coursera",
        "base_url": "https://www.coursera.org/search?query=",
        "difficulty": "intermediate",
        "description": "University courses online",
    },
    "edx": {
        "name": "edX",
        "base_url": "https://www.edx.org/search?q=",
        "difficulty": "intermediate",
        "description": "University-level courses",
    },
    "wikipedia": {
        "name": "Wikipedia",
        "base_url": "https://en.wikipedia.org/wiki/Special:Search?search=",
        "difficulty": "beginner",
        "description": "Encyclopedic reference",
    },
    "stanford_online": {
        "name": "Stanford Online",
        "base_url": "https://online.stanford.edu/search?keywords=",
        "difficulty": "advanced",
        "description": "Stanford University online learning",
    },
}


def find_educational_resources(
    topic: str,
    difficulty: Optional[str] = None,
    max_results: int = 10,
) -> List[EducationalResource]:
    """
    Find educational resources for a topic.

    Currently generates URLs to curated platforms. In the future,
    can integrate with platform APIs for richer results.

    Args:
        topic: The topic to search for
        difficulty: Filter by difficulty (beginner, intermediate, advanced)
        max_results: Maximum results

    Returns:
        List of EducationalResource
    """
    from urllib.parse import quote_plus

    resources = []

    for platform_id, platform in EDUCATIONAL_PLATFORMS.items():
        # Filter by difficulty if specified
        if difficulty and platform["difficulty"] != difficulty:
            continue

        encoded_topic = quote_plus(topic)
        url = platform["base_url"] + encoded_topic

        resources.append(
            EducationalResource(
                title=f"{topic} - {platform['name']}",
                platform=platform["name"],
                url=url,
                difficulty_level=platform["difficulty"],
                topic_match=0.5,  # Default relevance without API verification
            )
        )

    return resources[:max_results]


def format_educational_results(resources: List[EducationalResource]) -> str:
    """Format educational resources as markdown."""
    if not resources:
        return "No educational resources found."

    lines = ["# Educational Resources", ""]

    # Group by difficulty
    by_difficulty = {"beginner": [], "intermediate": [], "advanced": []}
    for r in resources:
        if r.difficulty_level in by_difficulty:
            by_difficulty[r.difficulty_level].append(r)

    for level, items in by_difficulty.items():
        if items:
            lines.append(f"## {level.title()}")
            lines.append("")
            for r in items:
                lines.append(f"- [{r.platform}]({r.url})")
            lines.append("")

    return "\n".join(lines)
