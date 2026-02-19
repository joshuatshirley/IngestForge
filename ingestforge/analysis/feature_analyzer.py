"""
Feature Analyzer for AIE + Army Doctrine Integration.

Analyzes feature descriptions by:
1. Searching the AIE code library for related components
2. Querying Army Doctrine RAG for applicable regulations
3. Finding related existing ADO work items
4. Identifying integration dependencies

This is a query-time analysis tool, not a batch processor.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.storage.base import ChunkRepository, SearchResult
from ingestforge.analysis.doctrine_client import DoctrineAPIClient

logger = get_logger(__name__)


@dataclass
class CodeMatch:
    """A code component related to a feature."""

    name: str  # Class/component name
    file_path: str  # Path to source file
    component_type: str  # Selector, Service, Handler, LWC, Trigger, etc.
    package: str  # SFDX package name
    relevance_score: float  # Semantic similarity score
    summary: str = ""  # Brief description of what this code does
    methods: List[str] = field(default_factory=list)  # Key methods
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "file_path": self.file_path,
            "component_type": self.component_type,
            "package": self.package,
            "relevance_score": self.relevance_score,
            "summary": self.summary,
            "methods": self.methods,
            "metadata": self.metadata,
        }


@dataclass
class RegulationMatch:
    """A regulation or policy applicable to a feature."""

    document: str  # e.g., "AR 601-210"
    section: str  # e.g., "Chapter 4-7"
    title: str  # Section title if available
    authority_level: int  # 1=Core, 2=Policy, 3=Guide
    relevance_score: float
    content: str  # Relevant excerpt
    compliance_note: str = ""  # How this applies to the feature

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document": self.document,
            "section": self.section,
            "title": self.title,
            "authority_level": self.authority_level,
            "relevance_score": self.relevance_score,
            "content": self.content,
            "compliance_note": self.compliance_note,
        }


@dataclass
class StoryMatch:
    """An existing ADO work item related to a feature."""

    ado_id: int  # Work item ID
    title: str
    work_item_type: str  # Feature, User Story, Bug, Task
    state: str  # Active, Closed, etc.
    relevance_score: float
    description: str = ""
    acceptance_criteria: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ado_id": self.ado_id,
            "title": self.title,
            "work_item_type": self.work_item_type,
            "state": self.state,
            "relevance_score": self.relevance_score,
            "description": self.description,
            "acceptance_criteria": self.acceptance_criteria,
        }


@dataclass
class FeatureAnalysis:
    """Complete analysis of a feature request."""

    feature_description: str
    related_code: List[CodeMatch]
    applicable_regulations: List[RegulationMatch]
    existing_stories: List[StoryMatch]
    integration_dependencies: List[str]
    suggested_keywords: List[str] = field(default_factory=list)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_description": self.feature_description,
            "related_code": [c.to_dict() for c in self.related_code],
            "applicable_regulations": [
                r.to_dict() for r in self.applicable_regulations
            ],
            "existing_stories": [s.to_dict() for s in self.existing_stories],
            "integration_dependencies": self.integration_dependencies,
            "suggested_keywords": self.suggested_keywords,
            "analysis_metadata": self.analysis_metadata,
        }

    @property
    def code_count(self) -> int:
        return len(self.related_code)

    @property
    def regulation_count(self) -> int:
        return len(self.applicable_regulations)

    @property
    def story_count(self) -> int:
        return len(self.existing_stories)


# Integration patterns detected from code
INTEGRATION_PATTERNS = {
    "MIRS": {
        "keywords": ["MIRSService", "MedicalVisit", "Physical", "MIRS", "medical"],
        "description": "Medical Information Retrieval System - medical visit scheduling and tracking",
    },
    "DISS": {
        "keywords": [
            "DISSService",
            "BackgroundCheck",
            "PSIExtract",
            "DISS",
            "security clearance",
        ],
        "description": "Defense Information Security System - background investigations",
    },
    "DocuSign": {
        "keywords": ["DocusignEnvelope", "eSignature", "ContentDocument", "DocuSign"],
        "description": "Electronic signature and document management",
    },
    "Mulesoft": {
        "keywords": ["BulkLeadUpload", "IntegrationService", "Mulesoft", "ESB"],
        "description": "Enterprise Service Bus for system integration",
    },
    "Territory": {
        "keywords": ["Territory2", "Assignment", "ZipCode", "Territory"],
        "description": "Geographic territory assignment and management",
    },
    "OmniStudio": {
        "keywords": ["FlexCard", "OmniScript", "DataRaptor", "OmniStudio"],
        "description": "Salesforce OmniStudio guided workflows",
    },
}


class FeatureAnalyzer:
    """
    Analyze features to find dependencies and generate context.

    Searches across:
    - AIE code library (Apex classes, LWC components)
    - Army Doctrine RAG API (regulations, policies)
    - ADO work items library (existing stories)

    Example:
        analyzer = FeatureAnalyzer(config, code_repo, ado_repo)
        analysis = await analyzer.analyze(
            "Implement medical waiver tracking for MIRS integration"
        )
        print(f"Found {analysis.code_count} related code components")
        print(f"Found {analysis.regulation_count} applicable regulations")
    """

    def __init__(
        self,
        config: Config,
        code_repository: Optional[ChunkRepository] = None,
        ado_repository: Optional[ChunkRepository] = None,
        doctrine_client: Optional[DoctrineAPIClient] = None,
    ):
        """
        Initialize the feature analyzer.

        Args:
            config: IngestForge configuration
            code_repository: Repository containing AIE code chunks
            ado_repository: Repository containing ADO work items
            doctrine_client: Client for Army Doctrine RAG API
        """
        self.config = config
        self.code_repository = code_repository
        self.ado_repository = ado_repository

        # Initialize doctrine client from config if not provided
        if doctrine_client:
            self.doctrine_client = doctrine_client
        elif config.feature_analysis.doctrine_api.enabled:
            self.doctrine_client = DoctrineAPIClient(
                base_url=config.feature_analysis.doctrine_api.url,
                timeout_seconds=config.feature_analysis.doctrine_api.timeout_seconds,
            )
        else:
            self.doctrine_client = None

    async def analyze(
        self,
        feature_description: str,
        max_code_results: Optional[int] = None,
        max_regulation_results: Optional[int] = None,
        max_story_results: Optional[int] = None,
    ) -> FeatureAnalysis:
        """
        Analyze a feature and return comprehensive context.

        Args:
            feature_description: Natural language description of the feature
            max_code_results: Override for max code results
            max_regulation_results: Override for max regulation results
            max_story_results: Override for max story results

        Returns:
            FeatureAnalysis with all dependencies and context
        """
        max_code = max_code_results or self.config.feature_analysis.max_code_results
        max_regs = (
            max_regulation_results or self.config.feature_analysis.doctrine_api.top_k
        )
        max_stories = (
            max_story_results or self.config.feature_analysis.max_story_results
        )

        # Extract keywords from feature description
        keywords = self._extract_keywords(feature_description)

        # Run all searches
        related_code = await self._search_code(feature_description, keywords, max_code)
        regulations = await self._search_regulations(feature_description, max_regs)
        existing_stories = await self._search_stories(
            feature_description, keywords, max_stories
        )

        # Detect integration dependencies
        integrations = self._detect_integrations(feature_description, related_code)

        return FeatureAnalysis(
            feature_description=feature_description,
            related_code=related_code,
            applicable_regulations=regulations,
            existing_stories=existing_stories,
            integration_dependencies=integrations,
            suggested_keywords=keywords,
            analysis_metadata={
                "code_library": "aie-code" if self.code_repository else None,
                "ado_library": "ado-stories" if self.ado_repository else None,
                "doctrine_enabled": self.doctrine_client is not None,
            },
        )

    def analyze_sync(
        self,
        feature_description: str,
        max_code_results: Optional[int] = None,
        max_regulation_results: Optional[int] = None,
        max_story_results: Optional[int] = None,
    ) -> FeatureAnalysis:
        """Synchronous wrapper for analyze()."""
        import asyncio

        return asyncio.run(
            self.analyze(
                feature_description,
                max_code_results,
                max_regulation_results,
                max_story_results,
            )
        )

    async def _search_code(
        self,
        feature_description: str,
        keywords: List[str],
        max_results: int,
    ) -> List[CodeMatch]:
        """Search for related code components."""
        if not self.code_repository:
            logger.warning("No code repository configured, skipping code search")
            return []

        results = []
        try:
            # Semantic search with feature description
            search_results = self.code_repository.search(
                query=feature_description,
                top_k=max_results,
                library_filter="aie-code",
            )

            for r in search_results:
                match = self._convert_to_code_match(r)
                if match:
                    results.append(match)

            logger.info(f"Found {len(results)} related code components")

        except Exception as e:
            logger.error(f"Code search failed: {e}")

        return results

    async def _search_regulations(
        self,
        feature_description: str,
        max_results: int,
    ) -> List[RegulationMatch]:
        """Search for applicable regulations via Doctrine API."""
        if not self.doctrine_client:
            logger.warning("Doctrine API not configured, skipping regulation search")
            return []

        results = []
        try:
            doctrine_results = await self.doctrine_client.retrieve(
                query=feature_description,
                top_k=max_results,
            )

            for r in doctrine_results:
                match = RegulationMatch(
                    document=r.document,
                    section=r.section,
                    title="",  # API may not return title
                    authority_level=r.authority_level,
                    relevance_score=r.relevance_score,
                    content=r.content[:500]
                    if r.content
                    else "",  # Truncate for display
                )
                results.append(match)

            logger.info(f"Found {len(results)} applicable regulations")

        except Exception as e:
            logger.error(f"Regulation search failed: {e}")

        return results

    async def _search_stories(
        self,
        feature_description: str,
        keywords: List[str],
        max_results: int,
    ) -> List[StoryMatch]:
        """Search for related existing ADO work items."""
        if not self.ado_repository:
            logger.warning("No ADO repository configured, skipping story search")
            return []

        results = []
        try:
            search_results = self.ado_repository.search(
                query=feature_description,
                top_k=max_results,
                library_filter="ado-stories",
            )

            for r in search_results:
                match = self._convert_to_story_match(r)
                if match:
                    results.append(match)

            logger.info(f"Found {len(results)} related ADO stories")

        except Exception as e:
            logger.error(f"Story search failed: {e}")

        return results

    def _convert_to_code_match(self, result: SearchResult) -> Optional[CodeMatch]:
        """
        Convert a SearchResult to a CodeMatch.

        Rule #1: Early return eliminates nesting
        """
        try:
            metadata = result.metadata or {}
            component_type = self._determine_component_type(
                metadata, result.source_file
            )

            # Extract method names
            methods = []
            method_data = metadata.get("methods", [])
            if isinstance(method_data, list):
                methods = [
                    m.get("name", "") for m in method_data if isinstance(m, dict)
                ]

            # Build summary from content
            summary = ""
            apexdoc = metadata.get("apexdoc", {})
            if isinstance(apexdoc, dict) and "description" in apexdoc:
                summary = apexdoc["description"][:200]

            return CodeMatch(
                name=metadata.get("name", result.document_id),
                file_path=result.source_file,
                component_type=component_type,
                package=metadata.get("package", ""),
                relevance_score=result.score,
                summary=summary,
                methods=methods[:5],  # Top 5 methods
                metadata={
                    "soql_queries": metadata.get("soql_queries", [])[:3],
                    "apex_imports": metadata.get("apex_imports", []),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to convert code result: {e}")
            return None

    def _determine_component_type(
        self, metadata: Dict[str, Any], source_file: str
    ) -> str:
        """
        Determine component type from metadata and file path.

        Rule #1: Early return pattern eliminates if/elif chain
        Rule #4: Extracted helper function
        """
        if metadata.get("is_trigger"):
            return "Trigger"
        if "lwc" in source_file.lower():
            return "LWC"

        # Default to class_type from metadata
        return metadata.get("class_type", "Unknown")

    def _convert_to_story_match(self, result: SearchResult) -> Optional[StoryMatch]:
        """Convert a SearchResult to a StoryMatch."""
        try:
            metadata = result.metadata or {}

            return StoryMatch(
                ado_id=metadata.get("ado_id", 0),
                title=metadata.get("title", result.section_title),
                work_item_type=metadata.get("work_item_type", "Unknown"),
                state=metadata.get("state", "Unknown"),
                relevance_score=result.score,
                description=result.content[:300] if result.content else "",
                acceptance_criteria=metadata.get("acceptance_criteria", [])[:3],
            )
        except Exception as e:
            logger.warning(f"Failed to convert story result: {e}")
            return None

    def _detect_integrations(
        self,
        feature_description: str,
        code_matches: List[CodeMatch],
    ) -> List[str]:
        """Detect integration dependencies from feature and code.

        Rule #1: Reduced nesting via helper extraction
        """
        detected = set()
        feature_lower = feature_description.lower()

        # Check feature description for integration keywords
        self._check_text_for_integrations(feature_lower, detected)

        # Check code matches for integration patterns
        for code in code_matches:
            code_text = f"{code.name} {code.file_path} {' '.join(code.methods)}"
            self._check_text_for_integrations(code_text.lower(), detected)

        return sorted(detected)

    def _check_text_for_integrations(self, text_lower: str, detected: set) -> None:
        """Check text for integration keywords and add to detected set.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        for integration, info in INTEGRATION_PATTERNS.items():
            if self._has_integration_keyword(text_lower, info["keywords"]):
                detected.add(integration)

    def _has_integration_keyword(self, text_lower: str, keywords: List[str]) -> bool:
        """Check if text contains any integration keyword.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return True
        return False

    def _extract_keywords(self, feature_description: str) -> List[str]:
        """Extract significant keywords from feature description."""
        # Common stop words to filter out
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "and",
            "but",
            "or",
            "so",
            "if",
            "when",
            "then",
            "that",
            "this",
            "it",
            "implement",
            "add",
            "create",
            "update",
            "feature",
            "system",
        }

        # Extract words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", feature_description)

        # Filter and deduplicate
        keywords = []
        seen = set()
        for word in words:
            word_lower = word.lower()
            if word_lower not in stop_words and word_lower not in seen:
                seen.add(word_lower)
                keywords.append(word)

        # Prioritize capitalized words (likely proper nouns/system names)
        def sort_key(w: Any) -> Any:
            if w[0].isupper():
                return (0, w)
            return (1, w)

        keywords.sort(key=sort_key)
        return keywords[:10]  # Top 10 keywords

    def get_integration_description(self, integration_name: str) -> str:
        """Get description for an integration dependency."""
        info = INTEGRATION_PATTERNS.get(integration_name, {})
        return info.get("description", f"Integration with {integration_name}")
