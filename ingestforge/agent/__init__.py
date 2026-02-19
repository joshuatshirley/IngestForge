"""Autonomous agent capabilities for IngestForge.

This package provides agentic functionality:
- react_engine: ReAct reasoning loop (AGENT-001.1)
- tool_registry: Tool management and discovery (AGENT-001.2)
- llm_adapter: LLM adapter for ReActEngine (TASK-2.1.1-A)
- knowledge_tools: Knowledge base tools for agents (AGENT-002.1)
- synthesis: Report generation from agent results (AGENT-001.3)
- fact_checker: Adversarial debate verification (AGENT-002)
- verification_ui: Rich-based result display (AGENT-002.3)
- debate_adapter: LLM adapter for fact-checker (AGENT-002)
- paper_summarizer: Multi-agent paper summarization (RES-004)
- legal_classifier: Facts vs. opinions classifier (LEGAL-003)
- attack_mapper: MITRE ATT&CK technique mapping (CYBER-003)
- case_conflict: Case conflict detector for legal vertical (LEGAL-004)
"""

from ingestforge.agent.react_engine import (
    AgentState,
    AgentResult,
    ReActEngine,
    ReActStep,
    Tool,
    ToolOutput,
    ToolResult,
    SimpleTool,
    create_engine,
    MAX_ITERATIONS,
    MAX_TOOLS,
)
from ingestforge.agent.tool_registry import (
    ToolCategory,
    ToolParameter,
    ToolMetadata,
    RegisteredTool,
    ToolRegistry,
    create_registry,
    register_builtin_tools,
)
from ingestforge.agent.llm_adapter import (
    LLMThinkAdapter,
    create_llm_think_adapter,
)
from ingestforge.agent.knowledge_tools import (
    search_knowledge_base,
    ingest_document,
    get_chunk_details,
    register_knowledge_tools,
)
from ingestforge.agent.synthesis import (
    ReportFormat,
    SectionType,
    Finding,
    ReportSection,
    Report,
    ReportSynthesizer,
    ReportExporter,
    create_synthesizer,
    synthesize_report,
)
from ingestforge.agent.fact_checker import (
    DebateOrchestrator,
    VerificationResult,
    VerificationStatus,
    DebateRole,
    Evidence,
    Claim,
    DebateArgument,
    create_orchestrator,
)
from ingestforge.agent.verification_ui import (
    VerificationDisplay,
    DisplayConfig,
    display_result,
    display_batch,
)
from ingestforge.agent.debate_adapter import (
    create_proponent_function,
    create_critic_function,
)
from ingestforge.agent.paper_summarizer import (
    AgentRole,
    AgentOutput,
    PaperSummary,
    SummarizationPrompts,
    PaperSummarizer,
    create_paper_summarizer,
    summarize_paper,
)
from ingestforge.agent.legal_classifier import (
    LegalRole,
    LegalClassification,
    ClassificationPrompts,
    LegalClassifier,
    create_legal_classifier,
    classify_legal_text,
)
from ingestforge.agent.attack_mapper import (
    ATTACKMapper,
    ATTACKMapping,
    TechniqueInfo,
    Tactic,
    create_mapper,
    create_llm_mapping_function,
    map_chunk_to_attack,
    get_technique_database,
    get_all_tactics,
)
from ingestforge.agent.case_conflict import (
    CaseConflict,
    CaseConflictDetector,
    ConflictPrompts,
    ConflictType,
    create_case_conflict_detector,
)

__all__ = [
    # ReAct Engine (AGENT-001.1)
    "ReActEngine",
    "AgentState",
    "AgentResult",
    "ReActStep",
    # Tool protocol
    "Tool",
    "ToolOutput",
    "ToolResult",
    "SimpleTool",
    # Tool Registry (AGENT-001.2)
    "ToolRegistry",
    "ToolCategory",
    "ToolParameter",
    "ToolMetadata",
    "RegisteredTool",
    # LLM Adapter (TASK-2.1.1-A)
    "LLMThinkAdapter",
    "create_llm_think_adapter",
    # Knowledge Tools (AGENT-002.1)
    "search_knowledge_base",
    "ingest_document",
    "get_chunk_details",
    "register_knowledge_tools",
    # Synthesis (AGENT-001.3)
    "ReportFormat",
    "SectionType",
    "Finding",
    "ReportSection",
    "Report",
    "ReportSynthesizer",
    "ReportExporter",
    # Fact Checker (AGENT-002)
    "DebateOrchestrator",
    "VerificationResult",
    "VerificationStatus",
    "DebateRole",
    "Evidence",
    "Claim",
    "DebateArgument",
    "create_orchestrator",
    # Verification UI (AGENT-002.3)
    "VerificationDisplay",
    "DisplayConfig",
    "display_result",
    "display_batch",
    # Debate Adapter (AGENT-002)
    "create_proponent_function",
    "create_critic_function",
    # Paper Summarizer (RES-004)
    "AgentRole",
    "AgentOutput",
    "PaperSummary",
    "SummarizationPrompts",
    "PaperSummarizer",
    "create_paper_summarizer",
    "summarize_paper",
    # Legal Classifier (LEGAL-003)
    "LegalRole",
    "LegalClassification",
    "ClassificationPrompts",
    "LegalClassifier",
    "create_legal_classifier",
    "classify_legal_text",
    # ATT&CK Mapper (CYBER-003)
    "ATTACKMapper",
    "ATTACKMapping",
    "TechniqueInfo",
    "Tactic",
    "create_mapper",
    "create_llm_mapping_function",
    "map_chunk_to_attack",
    "get_technique_database",
    "get_all_tactics",
    # Case Conflict Detector (LEGAL-004)
    "CaseConflict",
    "CaseConflictDetector",
    "ConflictPrompts",
    "ConflictType",
    "create_case_conflict_detector",
    # Factory functions
    "create_engine",
    "create_registry",
    "register_builtin_tools",
    "create_synthesizer",
    "synthesize_report",
    # Constants
    "MAX_ITERATIONS",
    "MAX_TOOLS",
]
