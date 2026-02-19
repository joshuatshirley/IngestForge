"""
Enrichment Module for Chunk Enhancement.

This module handles Stage 4 of the pipeline: adding derived metadata to chunks
including embeddings, entities, hypothetical questions, and quality scores.

Architecture Position
---------------------
    CLI (outermost)
      └── **Feature Modules** (you are here)
            └── Shared (patterns, interfaces, utilities)
                  └── Core (innermost)

Pipeline Stage: 4 (Enrich)

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  ChunkRecords   │────→│    Enrich       │────→│ EnrichedChunks  │
    │  (raw chunks)   │     │  (embeddings+)  │     │ + embeddings    │
    └─────────────────┘     └─────────────────┘     │ + entities      │
                                                    │ + questions     │
                                                    └─────────────────┘

Enrichment Types
----------------
**Embeddings** (EmbeddingGenerator)
    Vector representations for semantic search.
    Default model: all-MiniLM-L6-v2 (384 dimensions)
    GPU acceleration supported with automatic batch sizing.

**Entities** (EntityExtractor - in entities.py)
    Named entity recognition: people, places, organizations.
    Uses spaCy with regex fallback when spaCy unavailable.

**Questions** (QuestionGenerator - in questions.py)
    Hypothetical questions the chunk could answer.
    Enables question-to-question matching for better retrieval.
    Uses LLM with template fallback.

**Summaries** (SummaryGenerator - in summary.py)
    One-liner summaries for each chunk.
    Uses LLM (Ollama/qwen2.5:14b) with extractive fallback.
    Improves retrieval and provides chunk previews.

**Quality Scores** (in quality_scorer module)
    Numeric score indicating chunk usefulness.
    Filters out low-value chunks (boilerplate, headers).

Key Components
--------------
**EmbeddingGenerator**
    Primary enricher for vector embeddings:
    - Automatic VRAM detection for batch sizing
    - GPU/CPU fallback
    - Batch processing for efficiency

**VRAMInfo**
    Dataclass with GPU memory information for optimization.

**JournalDateExtractor**
    Extracts publication dates from academic content.
    Handles various date formats in journal articles.

**Benchmark utilities**
    Performance testing for embedding models:
    - run_comparison_benchmark(): Compare embedding models
    - print_benchmark_report(): Format benchmark results

Configuration
-------------
Enrichment behavior is controlled by EnrichmentConfig:

    enrichment:
      generate_embeddings: true
      embedding_model: all-MiniLM-L6-v2
      extract_entities: true
      generate_questions: false      # Expensive, off by default
      generate_summaries: true       # Chunk summaries via LLM
      compute_quality: true

Usage Example
-------------
    from ingestforge.enrichment import EmbeddingGenerator

    # Create embedding generator
    embedder = EmbeddingGenerator(config)

    # Enrich a batch of chunks
    enriched_chunks = embedder.enrich_batch(chunks)

    # Access embeddings
    for chunk in enriched_chunks:
        print(f"Chunk {chunk.chunk_id}: {len(chunk.embedding)} dims")

    # Check VRAM for batch sizing
    from ingestforge.enrichment import get_batch_size_recommendation
    batch_size = get_batch_size_recommendation()
    print(f"Recommended batch size: {batch_size}")

Composition with EnrichmentPipeline
-----------------------------------
    from ingestforge.shared.patterns import EnrichmentPipeline
    from ingestforge.enrichment.entities import EntityExtractor
    from ingestforge.enrichment.questions import QuestionGenerator

    pipeline = EnrichmentPipeline([
        EntityExtractor(),
        QuestionGenerator(config),
        EmbeddingGenerator(config),
    ])

    enriched = pipeline.enrich(chunks)
"""

from ingestforge.enrichment.embeddings import (
    EmbeddingGenerator,
    VRAMInfo,
    calculate_optimal_batch_size,
    get_batch_size_recommendation,
)
from ingestforge.enrichment.benchmark import (
    BenchmarkResult,
    BenchmarkComparison,
    run_comparison_benchmark,
    print_benchmark_report,
)
from ingestforge.enrichment.journal_date_extractor import (
    JournalDateExtractor,
    ExtractedDate,
)
from ingestforge.enrichment.summary import SummaryGenerator

# Week 6: NER + Relationship Extraction + Knowledge Graphs
from ingestforge.enrichment.ner import (
    Entity,
    NERExtractor,
)
from ingestforge.enrichment.entity_linker import (
    EntityLinker,
    EntityProfile,
    LinkedEntity,
    EntityIndex,
    find_similar_entities,
    link_entity,
    build_entity_index,
)
from ingestforge.enrichment.relationships import (
    Relationship,
    SVOTriple,
    RELATIONSHIP_TYPES,
    extract_svo,
    extract_with_entities,
    extract_by_type,
    get_relationship_types,
)
from ingestforge.enrichment.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphBuilder,
    build_graph_from_text,
    export_to_mermaid_file,
)

# P3-AI-002: Fact-Checking Components
from ingestforge.enrichment.contradiction import (
    ContradictionDetector,
    ContradictionResult,
    ContradictionPair,
)
from ingestforge.enrichment.evidence_linker import (
    EvidenceLinker,
    LinkedEvidence,
    EvidenceLinkResult,
    SupportType,
)

# RES-001: Research Vertical - LaTeX Refiner
from ingestforge.enrichment.latex_refiner import (
    LaTeXRefiner,
    refine_latex,
    extract_equations,
    to_unicode,
)

# LEGAL-002: Legal Vertical - Bluebook Citation Parser
from ingestforge.enrichment.bluebook_parser import (
    BluebookParser,
    LegalCitation,
    extract_citations,
    parse_citation,
    enrich_with_citations,
    FEDERAL_REPORTERS,
    STATE_REPORTERS,
    COURT_ABBREVIATIONS,
)

# CYBER-001: Cyber Vertical - Log Flattener
from ingestforge.enrichment.log_flattener import (
    LogFlattener,
    LogFlattenerConfig,
    LogFormat,
    EventCategory,
    Severity,
    FlattenedLog,
    flatten_log,
    detect_log_format,
    extract_events_from_file,
)

# RES-003: Research Vertical - Citation Provenance Engine
from ingestforge.enrichment.citation_provenance import (
    CitationProvenanceEngine,
    CitationProvenance,
    CitationStyle as AcademicCitationStyle,
    extract_academic_citations,
    detect_citation_style,
    link_citations_to_chunks,
)

# New Verticals (Week 7+)
from ingestforge.enrichment.tech_metadata import TechMetadataRefiner
from ingestforge.enrichment.obsidian import ObsidianMetadataRefiner
from ingestforge.enrichment.family_tree import FamilyTreeEnricher
from ingestforge.enrichment.rpg import RPGMetadataRefiner
from ingestforge.enrichment.hr import HRMetadataRefiner
from ingestforge.enrichment.museum import MuseumMetadataRefiner
from ingestforge.enrichment.bio import BioMetadataRefiner
from ingestforge.enrichment.grant import GrantMetadataRefiner
from ingestforge.enrichment.cyber import CyberMetadataRefiner
from ingestforge.enrichment.edu import EduMetadataRefiner
from ingestforge.enrichment.mfg import MfgMetadataRefiner
from ingestforge.enrichment.disaster import DisasterMetadataRefiner
from ingestforge.enrichment.political import PoliticalMetadataRefiner
from ingestforge.enrichment.wellness import WellnessMetadataRefiner
from ingestforge.enrichment.spiritual import SpiritualMetadataRefiner
from ingestforge.enrichment.auto import AutoMetadataRefiner
from ingestforge.enrichment.gaming import GamingMetadataRefiner
from ingestforge.enrichment.ai_safety import AISafetyMetadataRefiner
from ingestforge.enrichment.urban import UrbanMetadataRefiner

__all__ = [
    # Embeddings
    "EmbeddingGenerator",
    "VRAMInfo",
    "calculate_optimal_batch_size",
    "get_batch_size_recommendation",
    # Benchmarking
    "BenchmarkResult",
    "BenchmarkComparison",
    "run_comparison_benchmark",
    "print_benchmark_report",
    # Journal Date Extraction
    "JournalDateExtractor",
    "ExtractedDate",
    # Summary Generation
    "SummaryGenerator",
    # Week 6: Named Entity Recognition
    "Entity",
    "NERExtractor",
    # Week 6: Entity Linking
    "EntityLinker",
    "EntityProfile",
    "LinkedEntity",
    "EntityIndex",
    "find_similar_entities",
    "link_entity",
    "build_entity_index",
    # Week 6: Relationship Extraction
    "Relationship",
    "SVOTriple",
    "RELATIONSHIP_TYPES",
    "extract_svo",
    "extract_with_entities",
    "extract_by_type",
    "get_relationship_types",
    # Week 6: Knowledge Graphs
    "KnowledgeGraph",
    "KnowledgeGraphBuilder",
    "build_graph_from_text",
    "export_to_mermaid_file",
    # P3-AI-002: Fact-Checking
    "ContradictionDetector",
    "ContradictionResult",
    "ContradictionPair",
    "EvidenceLinker",
    "LinkedEvidence",
    "EvidenceLinkResult",
    "SupportType",
    # RES-001: LaTeX Refiner
    "LaTeXRefiner",
    "refine_latex",
    "extract_equations",
    "to_unicode",
    # LEGAL-002: Bluebook Citation Parser
    "BluebookParser",
    "LegalCitation",
    "extract_citations",
    "parse_citation",
    "enrich_with_citations",
    "FEDERAL_REPORTERS",
    "STATE_REPORTERS",
    "COURT_ABBREVIATIONS",
    # CYBER-001: Log Flattener
    "LogFlattener",
    "LogFlattenerConfig",
    "LogFormat",
    "EventCategory",
    "Severity",
    "FlattenedLog",
    "flatten_log",
    "detect_log_format",
    "extract_events_from_file",
    # RES-003: Citation Provenance Engine
    "CitationProvenanceEngine",
    "CitationProvenance",
    "AcademicCitationStyle",
    "extract_academic_citations",
    "detect_citation_style",
    "link_citations_to_chunks",
    # New Verticals (Week 7+)
    "TechMetadataRefiner",
    "ObsidianMetadataRefiner",
    "FamilyTreeEnricher",
    "RPGMetadataRefiner",
    "HRMetadataRefiner",
    "MuseumMetadataRefiner",
    "BioMetadataRefiner",
    "GrantMetadataRefiner",
    "CyberMetadataRefiner",
    "EduMetadataRefiner",
    "MfgMetadataRefiner",
    "DisasterMetadataRefiner",
    "PoliticalMetadataRefiner",
    "WellnessMetadataRefiner",
    "SpiritualMetadataRefiner",
    "MuseumMetadataRefiner",
    "BioMetadataRefiner",
    "AutoMetadataRefiner",
    "GamingMetadataRefiner",
    "AISafetyMetadataRefiner",
    "UrbanMetadataRefiner",
]
