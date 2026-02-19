"""
Feature Analysis Module for AIE + Army Doctrine Integration.

This module provides tools for analyzing feature requests by:
1. Searching the AIE codebase for related components
2. Querying Army Doctrine RAG for applicable regulations
3. Finding related existing ADO work items
4. Generating user stories with compliance requirements

Components:
- DoctrineAPIClient: HTTP client for Army Doctrine RAG API
- FeatureAnalyzer: Core analysis engine
- StoryGenerator: LLM-based story generation with llama.cpp
"""

from ingestforge.analysis.doctrine_client import (
    DoctrineAPIClient,
    DoctrineResult,
)
from ingestforge.analysis.feature_analyzer import (
    FeatureAnalyzer,
    FeatureAnalysis,
    CodeMatch,
    RegulationMatch,
    StoryMatch,
)
from ingestforge.analysis.story_generator import (
    StoryGenerator,
    GeneratedStory,
)

from ingestforge.analysis.metrics import (
    MetricCollector,
    StyleMetrics,
    ReadabilityScores,
    VocabularyMetrics,
    SentenceMetrics,
    ReadabilityLevel,
    analyze_text,
    get_readability_summary,
)
from ingestforge.analysis.style_critique import (
    StyleCritic,
    CritiqueResult,
    Suggestion,
    SuggestionType,
    compare_styles,
    critique_text,
)
from ingestforge.analysis.similarity_window import (
    SemanticWindowComparator,
    ComparisonResult,
    SimilarityMatch,
    TextWindow,
    WindowConfig,
    create_comparator,
    find_similar_passages,
)

# CYBER-004: Incident Timeline Builder
from ingestforge.analysis.timeline_builder import (
    TimelineBuilder,
    TimelineEntry,
    CorrelationGroup,
    build_timeline_from_logs,
    build_timeline_from_chunks,
)

__all__ = [
    # Doctrine Client
    "DoctrineAPIClient",
    "DoctrineResult",
    # Feature Analyzer
    "FeatureAnalyzer",
    "FeatureAnalysis",
    "CodeMatch",
    "RegulationMatch",
    "StoryMatch",
    # Story Generator
    "StoryGenerator",
    "GeneratedStory",
    # Style Metrics (P3-AI-003.1)
    "MetricCollector",
    "StyleMetrics",
    "ReadabilityScores",
    "VocabularyMetrics",
    "SentenceMetrics",
    "ReadabilityLevel",
    "analyze_text",
    "get_readability_summary",
    # Style Critique (P3-AI-003.2)
    "StyleCritic",
    "CritiqueResult",
    "Suggestion",
    "SuggestionType",
    "compare_styles",
    "critique_text",
    # Similarity Window (P3-AI-004.1)
    "SemanticWindowComparator",
    "ComparisonResult",
    "SimilarityMatch",
    "TextWindow",
    "WindowConfig",
    "create_comparator",
    "find_similar_passages",
    # CYBER-004: Timeline Builder
    "TimelineBuilder",
    "TimelineEntry",
    "CorrelationGroup",
    "build_timeline_from_logs",
    "build_timeline_from_chunks",
]
