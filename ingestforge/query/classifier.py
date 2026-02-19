"""
Query intent classification.

Classify queries to optimize retrieval strategy.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List


@dataclass
class QueryClassification:
    """Results of query classification."""

    intent: str
    domains: List[str] = field(default_factory=list)
    confidence: float = 1.0


INTENT_PATTERNS: Dict[str, List[str]] = {
    "factual": [
        r"^what (?:is|are|was|were)",
        r"^when (?:is|was|did|does)",
        r"^where (?:is|are|was|were|do|does)",
        r"^who (?:is|are|was|were)",
        r"^which ",
        r"\?$",
    ],
    "procedural": [
        r"^how (?:do|does|can|to|would)",
        r"^how (?:is|are) .+ (?:done|performed|executed)",
        r"steps to",
        r"process for",
        r"procedure",
        r"guide to",
    ],
    "conceptual": [
        r"^what (?:is|are) (?:a |an |the )?(?:definition|meaning)",
        r"^define ",
        r"^explain ",
        r"^describe ",
        r"concept of",
        r"understanding",
    ],
    "comparative": [
        r"difference between",
        r"compare ",
        r"vs\.?",
        r"versus",
        r"better than",
        r"worse than",
        r"similar to",
    ],
    "exploratory": [
        r"^tell me about",
        r"^what do you know about",
        r"^information about",
        r"overview of",
        r"summary of",
    ],
    "literary": [
        r"character",
        r"theme",
        r"motif",
        r"symbol(?:ism)?",
        r"narrative",
        r"protagonist",
        r"antagonist",
        r"literary",
        r"metaphor",
        r"allegory",
        r"foreshadow",
        r"irony",
    ],
}

DOMAIN_PATTERNS: Dict[str, List[str]] = {
    "legal": [
        r"statute",
        r"jurisdiction",
        r"judge",
        r"docket",
        r"court",
        r"legal",
        r"opinion",
        r"ruling",
        r"citation",
        r"bluebook",
        r"case law",
        r"precedent",
        r"plaintiff",
        r"defendant",
        r"lawsuit",
        r"litigation",
    ],
    "tech": [
        r"code",
        r"function",
        r"method",
        r"class",
        r"api",
        r"implementation",
        r"tree-sitter",
        r"language",
        r"import",
        r"export",
        r"dependency",
        r"software",
        r"develop",
        r"debug",
        r"refactor",
        r"algorithm",
    ],
    "pkm": [
        r"vault",
        r"obsidian",
        r"wikilink",
        r"backlink",
        r"frontmatter",
        r"alias",
        r"tag",
        r"daily note",
        r"logseq",
        r"roam",
        r"knowledge base",
    ],
    "genealogy": [
        r"family",
        r"tree",
        r"ancestor",
        r"descendant",
        r"parent",
        r"child",
        r"birth",
        r"death",
        r"marriage",
        r"gedcom",
        r"genealogy",
        r"lineage",
        r"grandfather",
        r"grandmother",
        r"relative",
    ],
    "rpg": [
        r"npc",
        r"character sheet",
        r"stat block",
        r"armor class",
        r"hit points",
        r"dungeon",
        r"campaign",
        r"adventure",
        r"d&d",
        r"tabletop",
        r"dm",
        r"game master",
        r"monster",
        r"item",
        r"rarity",
    ],
    "cyber": [
        r"log",
        r"cve",
        r"vulnerability",
        r"attack",
        r"tactic",
        r"technique",
        r"mitre",
        r"security",
        r"threat",
        r"ip address",
        r"source ip",
        r"firewall",
        r"cloudtrail",
        r"syslog",
        r"incident",
        r"breach",
    ],
}


class QueryClassifier:
    """
    Classify query intent for optimized retrieval.

    Intent types:
    - factual: Fact-seeking questions (what, when, where, who)
    - procedural: How-to questions
    - conceptual: Definition/explanation questions
    - comparative: Comparison questions
    - exploratory: Open-ended exploration
    """

    def __init__(self) -> None:
        """
        Initialize query classifier.

        Rule #4: Reduced from 65 → 12 lines (extracted patterns to module level)
        """
        self.patterns = INTENT_PATTERNS

        # Compile patterns
        self._compiled: Dict[str, List[re.Pattern[str]]] = {}
        for intent, patterns in self.patterns.items():
            self._compiled[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def classify(self, query: str) -> str:
        """
        Classify query intent.

        Args:
            query: User query

        Returns:
            Intent type string
        """
        query = query.strip()

        # Check each intent
        scores: Dict[str, int] = {}
        for intent, patterns in self._compiled.items():
            score = 0
            for pattern in patterns:
                if pattern.search(query):
                    score += 1
            if score > 0:
                scores[intent] = score

        # Return highest scoring intent
        if scores:
            return max(scores, key=scores.get)

        # Default based on query structure
        if query.endswith("?"):
            return "factual"
        return "exploratory"

    def classify_full(self, query: str) -> QueryClassification:
        """Perform full classification including intent and domains."""
        intent = self.classify(query)
        domain_classifier = DomainClassifier()
        domains = domain_classifier.classify(query)

        return QueryClassification(intent=intent, domains=domains)


class DomainClassifier:
    """
    Classify the domain of a query to activate specific tools.
    """

    def __init__(self) -> None:
        self.patterns = DOMAIN_PATTERNS
        self._compiled = {
            domain: [re.compile(p, re.IGNORECASE) for p in patterns]
            for domain, patterns in self.patterns.items()
        }

    def classify(self, query: str) -> List[str]:
        """Identify which domains match the query."""
        matched_domains = []

        for domain, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(query):
                    matched_domains.append(domain)
                    break

        return matched_domains
