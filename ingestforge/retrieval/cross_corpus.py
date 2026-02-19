"""
Cross-Corpus Linker for linking ADO and Code corpora.

Enables bi-directional linking between work items and code,
gap analysis, and coverage matrix generation.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set

from ingestforge.core.logging import get_logger
from ingestforge.storage.base import ChunkRepository, SearchResult

logger = get_logger(__name__)


@dataclass
class CodeReference:
    """Reference from a work item to code."""

    work_item_id: int
    work_item_title: str
    code_entity: str  # Class name, package, etc.
    reference_type: str  # apex_class, package, lwc
    confidence: float
    source_chunk_id: str


@dataclass
class StoryReference:
    """Reference from code to a work item."""

    class_name: str
    package: str
    work_item_id: int
    work_item_type: str
    work_item_title: str
    confidence: float
    source_chunk_id: str


@dataclass
class CoverageEntry:
    """Entry in the coverage matrix."""

    package: str
    class_count: int
    story_count: int
    bug_count: int
    task_count: int
    feature_count: int
    coverage_ratio: float  # stories / classes


@dataclass
class GapReport:
    """Report of orphaned code without story coverage."""

    entity_type: str  # apex_class, lwc, trigger
    entity_name: str
    package: str
    file_path: str
    methods: List[str]
    priority: str  # high, medium, low based on visibility


class CrossCorpusLinker:
    """
    Service for linking ADO work items and Salesforce code.

    Provides methods to:
    - Find code implementing a story
    - Find stories related to code
    - Build coverage matrix
    - Identify orphaned code (gaps)

    Example:
        linker = CrossCorpusLinker(ado_repo, code_repo)

        # Find code for a story
        results = linker.find_code_for_story(29232)

        # Find stories for a class
        results = linker.find_stories_for_code("AccountsSelector")

        # Generate coverage report
        matrix = linker.build_coverage_matrix()
    """

    # Patterns for matching code entities in work items
    APEX_PATTERN = re.compile(
        r"\b([A-Z][a-zA-Z0-9]*(?:Selector|Service|Handler|Controller|Domain|Batch))\b"
    )
    PACKAGE_PATTERN = re.compile(r"\b(aie-[a-z0-9-]+)\b", re.IGNORECASE)
    LWC_PATTERN = re.compile(r"\b([a-z][a-zA-Z0-9]+Component)\b")

    def __init__(
        self,
        ado_repository: ChunkRepository,
        code_repository: ChunkRepository,
    ):
        """
        Initialize cross-corpus linker.

        Args:
            ado_repository: Repository containing ADO work items
            code_repository: Repository containing code chunks
        """
        self.ado_repo = ado_repository
        self.code_repo = code_repository

    def find_code_for_story(
        self,
        story_id: int,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Find code that implements or relates to a story.

        Rule #1: Reduced nesting from 4 → 2 levels via extraction
        Rule #4: Reduced from 79 → 38 lines

        Uses multiple strategies:
        1. Search for story ID mentions in code comments
        2. Extract code references from story and search
        3. Semantic search for story title/description

        Args:
            story_id: ADO work item ID
            top_k: Maximum results to return

        Returns:
            List of SearchResult from code corpus
        """
        results = []
        seen_ids: Set[str] = set()

        # Strategy 1: Direct ID reference in code
        id_results = self.code_repo.search(
            f"#{story_id}", top_k=top_k, library_filter="code"
        )
        self._add_search_results(id_results, seen_ids, results, score_boost=1.2)

        # Strategy 2 & 3: Get story and search for references and title
        story_results = self.ado_repo.search(
            f"#{story_id}", top_k=1, library_filter="ado"
        )
        if story_results:
            story = story_results[0]
            self._search_story_code_references(story.content, seen_ids, results)
            self._search_by_story_title(story, top_k, seen_ids, results)

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _add_search_results(
        self,
        search_results: List[SearchResult],
        seen_ids: Set[str],
        results: List[SearchResult],
        score_boost: float = 1.0,
    ) -> None:
        """
        Add search results while avoiding duplicates.

        Rule #1: Extracted to reduce nesting (max 1 level)
        Rule #4: Function <60 lines
        """
        for result in search_results:
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                result.score *= score_boost
                results.append(result)

    def _search_story_code_references(
        self, story_content: str, seen_ids: Set[str], results: List[SearchResult]
    ) -> None:
        """
        Search for code referenced in story content.

        Rule #1: Extracted to reduce nesting (max 2 levels)
        Rule #4: Function <60 lines
        """
        apex_refs = self.APEX_PATTERN.findall(story_content)

        # Search for each referenced entity (limit to top 5)
        for ref in apex_refs[:5]:
            ref_results = self.code_repo.search(ref, top_k=3, library_filter="code")
            self._add_search_results(ref_results, seen_ids, results, score_boost=1.1)

    def _search_by_story_title(
        self,
        story: SearchResult,
        top_k: int,
        seen_ids: Set[str],
        results: List[SearchResult],
    ) -> None:
        """
        Semantic search using story title.

        Rule #1: Extracted to reduce nesting (max 1 level)
        Rule #4: Function <60 lines
        """
        title = story.metadata.get("title", "") if story.metadata else ""
        if title:
            semantic_results = self.code_repo.search(
                title, top_k=top_k, library_filter="code"
            )
            self._add_search_results(semantic_results, seen_ids, results)

    def find_stories_for_code(
        self,
        class_name: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Find stories related to a code class or component.

        Args:
            class_name: Name of the Apex class, LWC component, etc.
            top_k: Maximum results to return

        Returns:
            List of SearchResult from ADO corpus
        """
        results = []
        seen_ids: Set[str] = set()

        # Strategy 1: Direct class name search
        direct_results = self.ado_repo.search(
            class_name,
            top_k=top_k,
            library_filter="ado",
        )
        for r in direct_results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                results.append(r)

        # Strategy 2: Search for related terms
        # Extract likely functionality from class name
        terms = self._extract_search_terms(class_name)
        for term in terms[:3]:
            term_results = self.ado_repo.search(
                term,
                top_k=5,
                library_filter="ado",
            )
            for r in term_results:
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    r.score *= 0.9  # Slightly lower for indirect matches
                    results.append(r)

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def build_coverage_matrix(self) -> Dict[str, CoverageEntry]:
        """
        Build a matrix of packages vs story coverage.

        Returns:
            Dict mapping package name to CoverageEntry
        """
        # Get all code chunks grouped by package
        package_classes: Dict[str, Set[str]] = {}
        package_files: Dict[str, Set[str]] = {}

        # This would need to iterate through all code chunks
        # For now, return empty - would need repository iteration support
        # This is a placeholder for the actual implementation

        logger.warning("build_coverage_matrix requires repository iteration support")
        return {}

    def find_orphaned_code(
        self,
        package: Optional[str] = None,
        min_importance: str = "low",
    ) -> List[GapReport]:
        """
        Find code without story coverage.

        Args:
            package: Optional package to filter
            min_importance: Minimum importance level (low, medium, high)

        Returns:
            List of GapReport for orphaned code
        """
        gaps = []

        # This would need to:
        # 1. Get all code entities from code corpus
        # 2. For each entity, check if any story references it
        # 3. Report entities with no story links

        logger.warning("find_orphaned_code requires repository iteration support")
        return gaps

    def _populate_story_info(self, story_id: int, report: Dict[str, Any]) -> bool:
        """Populate story information in report.

        Rule #4: No large functions - Extracted from get_traceability_report

        Returns:
            True if story found, False otherwise
        """
        story_results = self.ado_repo.search(
            f"#{story_id}",
            top_k=1,
            library_filter="ado",
        )

        if not story_results:
            report["error"] = f"Story #{story_id} not found"
            return False

        story = story_results[0]
        report["title"] = story.metadata.get("title", "") if story.metadata else ""
        report["state"] = story.metadata.get("state", "") if story.metadata else ""
        return True

    def _add_code_coverage(self, story_id: int, report: Dict[str, Any]) -> list[Any]:
        """Add code coverage to report.

        Rule #4: No large functions - Extracted from get_traceability_report

        Returns:
            List of code results
        """
        code_results = self.find_code_for_story(story_id, top_k=20)
        report["code_coverage"] = [
            {
                "name": r.metadata.get("name", "") if r.metadata else r.source_file,
                "type": r.metadata.get("class_type", "Unknown") if r.metadata else "",
                "package": r.metadata.get("package", "") if r.metadata else "",
                "file": r.source_file,
                "score": r.score,
            }
            for r in code_results
        ]
        return code_results

    def _add_related_stories(
        self, story_id: int, code_results: list, report: Dict[str, Any]
    ) -> None:
        """Add related stories to report.

        Rule #4: No large functions - Extracted from get_traceability_report
        """
        if not code_results:
            return

        # Get the primary code entity
        primary_code = code_results[0]
        code_name = (
            primary_code.metadata.get("name", "") if primary_code.metadata else ""
        )
        if code_name:
            related = self.find_stories_for_code(code_name, top_k=5)
            report["related_stories"] = [
                {
                    "id": r.metadata.get("ado_id", 0) if r.metadata else 0,
                    "title": r.metadata.get("title", "") if r.metadata else "",
                    "type": r.metadata.get("work_item_type", "") if r.metadata else "",
                    "score": r.score,
                }
                for r in related
                if r.metadata and r.metadata.get("ado_id") != story_id
            ]

    def get_traceability_report(
        self,
        story_id: int,
    ) -> Dict[str, Any]:
        """
        Generate a full traceability report for a story.

        Rule #4: Function <60 lines (refactored to 26 lines)

        Args:
            story_id: ADO work item ID

        Returns:
            Dict with code coverage, related items, and gaps
        """
        report = {
            "story_id": story_id,
            "code_coverage": [],
            "related_stories": [],
            "child_items": [],
            "gaps": [],
        }

        # Find and populate story info
        if not self._populate_story_info(story_id, report):
            return report

        # Add code coverage
        code_results = self._add_code_coverage(story_id, report)

        # Add related stories
        self._add_related_stories(story_id, code_results, report)

        return report

    def _extract_search_terms(self, class_name: str) -> List[str]:
        """
        Extract searchable terms from a class name.

        Examples:
            AccountsSelector -> ["Account", "Accounts", "select"]
            OpportunitiesService -> ["Opportunity", "Opportunities", "service"]
        """
        terms = []

        # Split camelCase
        words = re.findall(r"[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z])|[a-z]+", class_name)

        for word in words:
            word_lower = word.lower()
            # Skip common suffixes
            if word_lower in [
                "selector",
                "service",
                "handler",
                "controller",
                "domain",
                "test",
            ]:
                continue
            terms.append(word)

            # Add singular form if plural
            if word.endswith("ies"):
                terms.append(word[:-3] + "y")  # Opportunities -> Opportunity
            elif word.endswith("s") and not word.endswith("ss"):
                terms.append(word[:-1])  # Accounts -> Account

        return terms

    def suggest_stories_for_class(
        self,
        class_name: str,
        class_type: str,
        package: str,
        methods: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Suggest user stories that should exist for a code class.

        Rule #1: Reduced nesting from 4 → 1 level via dictionary dispatch
        Rule #4: Reduced from 67 → 24 lines

        Args:
            class_name: Name of the class
            class_type: Type (Selector, Service, Handler, etc.)
            package: Package name
            methods: List of public method names

        Returns:
            List of suggested story templates
        """
        # Base documentation story
        suggestions = [self._create_documentation_story(class_name)]
        type_generators = {
            "Selector": lambda: self._suggest_selector_stories(class_name, methods),
            "Service": lambda: self._suggest_service_stories(methods),
            "Handler": lambda: self._suggest_handler_stories(class_name),
        }

        generator = type_generators.get(class_type)
        if generator:
            suggestions.extend(generator())

        return suggestions

    def _create_documentation_story(self, class_name: str) -> Dict[str, Any]:
        """Rule #1: Extracted documentation story creation (<60 lines)."""
        return {
            "type": "User Story",
            "title": f"Document {class_name} functionality",
            "description": f"As a developer, I want documentation for {class_name} "
            f"so that I understand its purpose and usage.",
            "priority": "low",
        }

    def _suggest_selector_stories(
        self, class_name: str, methods: List[str]
    ) -> List[Dict[str, Any]]:
        """Rule #1: Extracted selector story generator (max 2 nesting levels)."""
        return [
            {
                "type": "User Story",
                "title": f"Query capability: {method}",
                "description": f"As a user, I want to {self._method_to_capability(method)} "
                f"using {class_name}.{method}().",
                "priority": "medium",
            }
            for method in methods[:5]
            if method.startswith("select")
        ]

    def _suggest_service_stories(self, methods: List[str]) -> List[Dict[str, Any]]:
        """Rule #1: Extracted service story generator (max 2 nesting levels)."""
        return [
            {
                "type": "User Story",
                "title": f"Business operation: {method}",
                "description": f"As a user, I want to {self._method_to_capability(method)} "
                f"so that I can complete my workflow.",
                "priority": "medium",
            }
            for method in methods[:5]
            if not method.startswith("_")
        ]

    def _suggest_handler_stories(self, class_name: str) -> List[Dict[str, Any]]:
        """Rule #1: Extracted handler story generator (<60 lines)."""
        entity = class_name.replace("Handler", "").replace("Trigger", "")
        return [
            {
                "type": "User Story",
                "title": f"Trigger automation for {entity}",
                "description": "As a user, I want automatic processing when records change "
                "so that related data stays consistent.",
                "priority": "high",
            }
        ]

    def _method_to_capability(self, method_name: str) -> str:
        """Convert method name to human-readable capability."""
        # Split camelCase
        words = re.findall(r"[A-Z][a-z]+|[a-z]+", method_name)
        return " ".join(words).lower()


# Alias for backwards compatibility
CrossCorpusRetriever = CrossCorpusLinker
