"""Multi-Agent Paper Summarizer for Research Documents.

Implements a 3-agent approach for academic paper summarization:
- Abstract Agent: Extracts and refines the paper's main thesis
- Methodology Agent: Identifies and explains research methods
- Critique Agent: Provides critical analysis and limitations
Usage Example
-------------
    from ingestforge.agent.paper_summarizer import PaperSummarizer

    summarizer = PaperSummarizer(llm_client)
    summary = summarizer.summarize(document_text)
    print(summary.abstract_summary)
    print(summary.methodology_summary)
    print(summary.critique_summary)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.llm.base import GenerationConfig, LLMClient

logger = get_logger(__name__)
MAX_DOCUMENT_LENGTH = 100000
MAX_SUMMARY_LENGTH = 2000
MAX_FINDINGS = 10
MAX_LIMITATIONS = 10
MAX_SOURCE_CHUNKS = 20


class AgentRole(Enum):
    """Role of summarization agent."""

    ABSTRACT = "abstract"
    METHODOLOGY = "methodology"
    CRITIQUE = "critique"


@dataclass
class PaperSummary:
    """Summary output from multi-agent paper analysis.

    Attributes:
        title: Paper title
        abstract_summary: Refined thesis and main contribution
        methodology_summary: Research methods explained
        critique_summary: Critical analysis
        key_findings: Bullet points of key findings
        limitations: Identified limitations
        source_chunks: References to source material
    """

    title: str
    abstract_summary: str
    methodology_summary: str
    critique_summary: str
    key_findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    source_chunks: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and truncate fields."""
        self.abstract_summary = self.abstract_summary[:MAX_SUMMARY_LENGTH]
        self.methodology_summary = self.methodology_summary[:MAX_SUMMARY_LENGTH]
        self.critique_summary = self.critique_summary[:MAX_SUMMARY_LENGTH]
        self.key_findings = self.key_findings[:MAX_FINDINGS]
        self.limitations = self.limitations[:MAX_LIMITATIONS]
        self.source_chunks = self.source_chunks[:MAX_SOURCE_CHUNKS]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "abstract_summary": self.abstract_summary,
            "methodology_summary": self.methodology_summary,
            "critique_summary": self.critique_summary,
            "key_findings": self.key_findings,
            "limitations": self.limitations,
            "source_chunks": self.source_chunks,
        }

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        sections = [
            f"# {self.title}",
            "",
            "## Abstract & Thesis",
            self.abstract_summary,
            "",
            "## Methodology",
            self.methodology_summary,
            "",
            "## Critical Analysis",
            self.critique_summary,
            "",
        ]

        if self.key_findings:
            sections.append("## Key Findings")
            for finding in self.key_findings:
                sections.append(f"- {finding}")
            sections.append("")

        if self.limitations:
            sections.append("## Limitations")
            for limitation in self.limitations:
                sections.append(f"- {limitation}")
            sections.append("")

        return "\n".join(sections)


@dataclass
class AgentOutput:
    """Output from a single agent analysis."""

    role: AgentRole
    summary: str
    key_points: List[str] = field(default_factory=list)
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "summary": self.summary[:MAX_SUMMARY_LENGTH],
            "key_points": self.key_points[:MAX_FINDINGS],
            "confidence": self.confidence,
        }


class SummarizationPrompts:
    """Prompt templates for summarization agents."""

    @staticmethod
    def abstract_prompt(document: str, title: str) -> str:
        """Generate prompt for abstract agent.

        Args:
            document: Document text
            title: Paper title

        Returns:
            Prompt string
        """
        return f"""Analyze this academic paper and extract the main thesis and contribution.

Paper Title: {title}

Document:
{document[:MAX_DOCUMENT_LENGTH]}

Provide your analysis in this format:

Summary: [2-3 paragraph summary of the main thesis and key contribution]
Key Points:
- [First key point]
- [Second key point]
- [Third key point]
Confidence: [0.0-1.0]

Focus on:
1. The central research question or problem
2. The main argument or hypothesis
3. The key contribution to the field
4. The scope and significance of the work"""

    @staticmethod
    def methodology_prompt(document: str, title: str) -> str:
        """Generate prompt for methodology agent.

        Args:
            document: Document text
            title: Paper title

        Returns:
            Prompt string
        """
        return f"""Analyze the research methodology in this academic paper.

Paper Title: {title}

Document:
{document[:MAX_DOCUMENT_LENGTH]}

Provide your analysis in this format:

Summary: [2-3 paragraph explanation of the research methodology]
Key Points:
- [First methodological point]
- [Second methodological point]
- [Third methodological point]
Confidence: [0.0-1.0]

Focus on:
1. Research design (experimental, observational, theoretical, etc.)
2. Data collection methods
3. Analysis techniques
4. Sample/population (if applicable)
5. Tools and frameworks used"""

    @staticmethod
    def critique_prompt(document: str, title: str) -> str:
        """Generate prompt for critique agent.

        Args:
            document: Document text
            title: Paper title

        Returns:
            Prompt string
        """
        return f"""Provide a critical analysis of this academic paper.

Paper Title: {title}

Document:
{document[:MAX_DOCUMENT_LENGTH]}

Provide your analysis in this format:

Summary: [2-3 paragraph critical analysis]
Key Points:
- [First limitation or critique]
- [Second limitation or critique]
- [Third limitation or critique]
Confidence: [0.0-1.0]

Focus on:
1. Methodological limitations
2. Gaps in the analysis
3. Potential biases
4. Alternative interpretations
5. Generalizability concerns
6. Future research directions

Be constructive but rigorous in your analysis."""


class PaperSummarizer:
    """Multi-agent paper summarizer.

    Uses three specialized agents to analyze academic papers:
    - Abstract Agent: Main thesis and contribution
    - Methodology Agent: Research methods
    - Critique Agent: Limitations and analysis
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        """Initialize the summarizer.

        Args:
            llm_client: LLM client for text generation
            config: Optional generation configuration

        Raises:
            ValueError: If llm_client is None
        """
        if llm_client is None:
            raise ValueError("llm_client cannot be None")

        self._llm = llm_client
        self._config = config or GenerationConfig(
            max_tokens=1500,
            temperature=0.3,
        )
        self._prompts = SummarizationPrompts()

    def summarize(
        self,
        document: str,
        title: str = "Untitled Paper",
        source_chunks: Optional[List[str]] = None,
    ) -> PaperSummary:
        """Summarize a paper using multi-agent approach.

        Args:
            document: Full document text
            title: Paper title
            source_chunks: Optional references to source chunks

        Returns:
            PaperSummary with all analysis components
        """
        if not document.strip():
            logger.warning("Empty document provided")
            return self._make_empty_summary(title)

        # Truncate document
        document = document[:MAX_DOCUMENT_LENGTH]

        # Run each agent
        abstract_output = self._run_abstract_agent(document, title)
        methodology_output = self._run_methodology_agent(document, title)
        critique_output = self._run_critique_agent(document, title)

        # Combine results
        return self._build_summary(
            title=title,
            abstract_output=abstract_output,
            methodology_output=methodology_output,
            critique_output=critique_output,
            source_chunks=source_chunks or [],
        )

    def _run_abstract_agent(self, document: str, title: str) -> AgentOutput:
        """Run abstract extraction agent.

        Args:
            document: Document text
            title: Paper title

        Returns:
            Agent output with thesis summary
        """
        prompt = self._prompts.abstract_prompt(document, title)
        return self._execute_agent(prompt, AgentRole.ABSTRACT)

    def _run_methodology_agent(self, document: str, title: str) -> AgentOutput:
        """Run methodology analysis agent.

        Args:
            document: Document text
            title: Paper title

        Returns:
            Agent output with methodology summary
        """
        prompt = self._prompts.methodology_prompt(document, title)
        return self._execute_agent(prompt, AgentRole.METHODOLOGY)

    def _run_critique_agent(self, document: str, title: str) -> AgentOutput:
        """Run critique analysis agent.

        Args:
            document: Document text
            title: Paper title

        Returns:
            Agent output with critical analysis
        """
        prompt = self._prompts.critique_prompt(document, title)
        return self._execute_agent(prompt, AgentRole.CRITIQUE)

    def _execute_agent(self, prompt: str, role: AgentRole) -> AgentOutput:
        """Execute an agent with the given prompt.

        Args:
            prompt: Agent prompt
            role: Agent role

        Returns:
            Parsed agent output
        """
        try:
            response = self._llm.generate(prompt, self._config)
        except Exception as e:
            logger.error(f"LLM generation failed for {role.value}: {e}")
            return AgentOutput(
                role=role,
                summary=f"Analysis failed: {str(e)}",
                key_points=[],
                confidence=0.0,
            )

        return self._parse_agent_response(response, role)

    def _parse_agent_response(self, response: str, role: AgentRole) -> AgentOutput:
        """Parse agent response into structured output.

        Args:
            response: Raw LLM response
            role: Agent role

        Returns:
            Parsed agent output
        """
        if not response.strip():
            logger.warning(f"Empty response from {role.value} agent")
            return AgentOutput(role=role, summary="No response", confidence=0.0)

        summary = self._extract_summary(response)
        key_points = self._extract_key_points(response)
        confidence = self._extract_confidence(response)

        return AgentOutput(
            role=role,
            summary=summary,
            key_points=key_points,
            confidence=confidence,
        )

    def _extract_summary(self, text: str) -> str:
        """Extract summary from response.

        Args:
            text: Response text

        Returns:
            Summary string
        """
        match = re.search(
            r"Summary:\s*(.+?)(?=\n(?:Key Points|Confidence|$))",
            text,
            re.DOTALL | re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

        # Fallback: use first paragraph
        paragraphs = text.strip().split("\n\n")
        return paragraphs[0] if paragraphs else "Unable to extract summary"

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from response.

        Args:
            text: Response text

        Returns:
            List of key points
        """
        key_points: List[str] = []

        # Find Key Points section
        match = re.search(
            r"Key Points:\s*(.+?)(?=\n(?:Confidence|$))",
            text,
            re.DOTALL | re.IGNORECASE,
        )

        if not match:
            return key_points

        points_text = match.group(1)
        lines = points_text.strip().split("\n")

        for line in lines:
            cleaned = re.sub(r"^[-*]\s*", "", line.strip())
            if cleaned:
                key_points.append(cleaned[:200])

        return key_points[:MAX_FINDINGS]

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from response.

        Args:
            text: Response text

        Returns:
            Confidence value (0.0-1.0)
        """
        match = re.search(
            r"Confidence:\s*(0?\.\d+|1\.0|[01])",
            text,
            re.IGNORECASE,
        )

        if match:
            try:
                confidence = float(match.group(1))
                return max(0.0, min(1.0, confidence))
            except ValueError:
                pass

        return 0.7  # Default confidence

    def _build_summary(
        self,
        title: str,
        abstract_output: AgentOutput,
        methodology_output: AgentOutput,
        critique_output: AgentOutput,
        source_chunks: List[str],
    ) -> PaperSummary:
        """Build final summary from agent outputs.

        Args:
            title: Paper title
            abstract_output: Abstract agent output
            methodology_output: Methodology agent output
            critique_output: Critique agent output
            source_chunks: Source references

        Returns:
            Complete paper summary
        """
        # Combine key findings from abstract and methodology
        key_findings = abstract_output.key_points + methodology_output.key_points

        # Limitations come from critique
        limitations = critique_output.key_points

        return PaperSummary(
            title=title,
            abstract_summary=abstract_output.summary,
            methodology_summary=methodology_output.summary,
            critique_summary=critique_output.summary,
            key_findings=key_findings[:MAX_FINDINGS],
            limitations=limitations[:MAX_LIMITATIONS],
            source_chunks=source_chunks[:MAX_SOURCE_CHUNKS],
        )

    def _make_empty_summary(self, title: str) -> PaperSummary:
        """Create empty summary for invalid input.

        Args:
            title: Paper title

        Returns:
            Empty paper summary
        """
        return PaperSummary(
            title=title,
            abstract_summary="No content available for analysis",
            methodology_summary="No content available for analysis",
            critique_summary="No content available for analysis",
            key_findings=[],
            limitations=[],
            source_chunks=[],
        )


def create_paper_summarizer(
    llm_client: LLMClient,
    config: Optional[GenerationConfig] = None,
) -> PaperSummarizer:
    """Factory function to create paper summarizer.

    Args:
        llm_client: LLM client instance
        config: Optional generation configuration

    Returns:
        Configured paper summarizer
    """
    return PaperSummarizer(llm_client=llm_client, config=config)


def summarize_paper(
    document: str,
    llm_client: LLMClient,
    title: str = "Untitled Paper",
    source_chunks: Optional[List[str]] = None,
) -> PaperSummary:
    """Convenience function to summarize a paper.

    Args:
        document: Full document text
        llm_client: LLM client instance
        title: Paper title
        source_chunks: Optional source references

    Returns:
        Paper summary
    """
    summarizer = create_paper_summarizer(llm_client)
    return summarizer.summarize(document, title, source_chunks)
