"""
User Story Generator for Feature Analysis.

Generates user stories with:
- Given-When-Then acceptance criteria
- Regulation citations for compliance
- Technical enablers for dependencies
- Links to related existing work items

Uses llama.cpp (local LLM) via IngestForge's LLM factory for generation.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.llm.base import LLMClient, GenerationConfig
from ingestforge.llm.factory import get_llm_client
from ingestforge.analysis.feature_analyzer import (
    FeatureAnalysis,
)

logger = get_logger(__name__)


@dataclass
class AcceptanceCriterion:
    """A single acceptance criterion in Given-When-Then format."""

    given: str
    when: str
    then: str

    def to_dict(self) -> Dict[str, Any]:
        return {"given": self.given, "when": self.when, "then": self.then}

    def to_string(self) -> str:
        return f"Given {self.given}, When {self.when}, Then {self.then}"


@dataclass
class GeneratedStory:
    """A generated user story or technical enabler."""

    id: str  # Generated ID like , TE-001, SP-001
    story_type: str  # "User Story", "Technical Enabler", "Spike"
    title: str
    description: str
    acceptance_criteria: List[AcceptanceCriterion]
    regulation_citations: List[str]  # e.g., ["AR 601-210 4-7(a)"]
    related_code: List[str]  # Class/component names
    related_stories: List[int]  # ADO IDs
    dependency_of: Optional[str] = None  # Parent story ID if this is a dependency
    priority: str = "Medium"  # High, Medium, Low
    story_points: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "story_type": self.story_type,
            "title": self.title,
            "description": self.description,
            "acceptance_criteria": [ac.to_dict() for ac in self.acceptance_criteria],
            "regulation_citations": self.regulation_citations,
            "related_code": self.related_code,
            "related_stories": self.related_stories,
            "dependency_of": self.dependency_of,
            "priority": self.priority,
            "story_points": self.story_points,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Format as markdown."""
        lines = [
            f"### {self.id}: {self.title}",
            f"**Type**: {self.story_type}",
            "",
            f"**Description**: {self.description}",
            "",
            "**Acceptance Criteria**:",
        ]

        for i, ac in enumerate(self.acceptance_criteria, 1):
            lines.append(f"- AC{i}: {ac.to_string()}")

        if self.regulation_citations:
            lines.append("")
            lines.append("**Regulation Compliance**:")
            for cite in self.regulation_citations:
                lines.append(f"- {cite}")

        if self.related_code:
            lines.append("")
            lines.append(f"**Related Code**: {', '.join(self.related_code)}")

        if self.related_stories:
            lines.append("")
            lines.append(
                f"**Related Stories**: {', '.join(f'#{s}' for s in self.related_stories)}"
            )

        if self.dependency_of:
            lines.append("")
            lines.append(f"**Dependency Of**: {self.dependency_of}")

        return "\n".join(lines)


# Prompt template for story generation
STORY_GENERATION_PROMPT = """You are a Salesforce business analyst creating user stories for an Army recruiting application (AIE - Automated Integration Environment).

## Feature Request
{feature_description}

## Related Code Found
{code_summaries}

## Applicable Army Regulations
{regulation_excerpts}

## Existing Related Stories
{existing_stories}

## Integration Dependencies
{integrations}

---

Generate {num_stories} user stories for this feature. For each story, provide:

1. A clear title (5-10 words)
2. Story type: "User Story" (user-facing), "Technical Enabler" (infrastructure), or "Spike" (research needed)
3. A 2-3 sentence description using "As a [persona], I want [goal], so that [benefit]" format for user stories
4. 2-3 acceptance criteria in Given-When-Then format
5. Any regulation compliance requirements from the applicable regulations above
6. Related code components that will be affected

Format your response as follows for each story:

### STORY: [Title]
TYPE: [User Story|Technical Enabler|Spike]
DESCRIPTION: [Description text]
ACCEPTANCE_CRITERIA:
- Given [precondition], When [action], Then [result]
- Given [precondition], When [action], Then [result]
COMPLIANCE: [Regulation reference and requirement, or "None"]
RELATED_CODE: [Class1, Class2, or "None"]
PRIORITY: [High|Medium|Low]
---

Focus on practical, implementable stories that address the core feature request while ensuring regulatory compliance."""


class StoryGenerator:
    """
    Generate user stories from feature analysis.

    Uses local LLM (llama.cpp) to generate stories with:
    - Proper Given-When-Then acceptance criteria
    - Regulation citations for compliance
    - Technical enabler identification
    - Dependency relationships

    Example:
        generator = StoryGenerator(config)
        stories = generator.generate_stories(analysis, max_stories=5)
        for story in stories:
            print(story.to_markdown())
    """

    def __init__(
        self,
        config: Config,
        llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize the story generator.

        Args:
            config: IngestForge configuration
            llm_client: Optional LLM client (uses config default if not provided)
        """
        self.config = config
        self._llm_client = llm_client
        self._story_counter = {"US": 0, "TE": 0, "SP": 0}

    @property
    def llm_client(self) -> LLMClient:
        """Get or create LLM client (lazy initialization)."""
        if self._llm_client is None:
            self._llm_client = get_llm_client(self.config)
        return self._llm_client

    def generate_stories(
        self,
        analysis: FeatureAnalysis,
        max_stories: Optional[int] = None,
    ) -> List[GeneratedStory]:
        """
        Generate user stories from feature analysis.

        Args:
            analysis: FeatureAnalysis from the feature analyzer
            max_stories: Maximum number of stories to generate

        Returns:
            List of GeneratedStory objects
        """
        max_stories = max_stories or self.config.feature_analysis.max_generated_stories

        # Build the prompt
        prompt = self._build_prompt(analysis, max_stories)

        # Generate with LLM
        try:
            gen_config = GenerationConfig(
                temperature=0.4,  # Moderate creativity
                max_tokens=2000,
            )

            response = self.llm_client.generate_with_context(
                system_prompt="You are a skilled business analyst creating user stories for enterprise software.",
                user_prompt=prompt,
                config=gen_config,
            )

            # Parse the response
            stories = self._parse_stories(response, analysis)
            logger.info(f"Generated {len(stories)} stories from analysis")
            return stories

        except Exception as e:
            logger.error(f"Story generation failed: {e}")
            return []

    def generate_technical_enablers(
        self,
        analysis: FeatureAnalysis,
    ) -> List[GeneratedStory]:
        """
        Generate technical enablers for integration dependencies.

        Creates enabler stories for each detected integration.

        Args:
            analysis: FeatureAnalysis with detected integrations

        Returns:
            List of GeneratedStory for technical enablers
        """
        enablers = []

        for integration in analysis.integration_dependencies:
            self._story_counter["TE"] += 1
            story_id = f"TE-{self._story_counter['TE']:03d}"

            # Create enabler for this integration
            enabler = GeneratedStory(
                id=story_id,
                story_type="Technical Enabler",
                title=f"{integration} Integration Support",
                description=f"Implement necessary infrastructure to support {integration} integration for this feature.",
                acceptance_criteria=[
                    AcceptanceCriterion(
                        given=f"the {integration} service is available",
                        when="the system sends a request",
                        then="a valid response is received within acceptable latency",
                    ),
                    AcceptanceCriterion(
                        given="an error occurs in the integration",
                        when="the system handles the error",
                        then="appropriate error logging and user feedback is provided",
                    ),
                ],
                regulation_citations=[],
                related_code=[],
                related_stories=[],
                priority="High",
                metadata={"integration": integration},
            )
            enablers.append(enabler)

        return enablers

    def _build_prompt(self, analysis: FeatureAnalysis, max_stories: int) -> str:
        """Build the LLM prompt from analysis."""
        # Format code summaries
        code_summaries = "None found"
        if analysis.related_code:
            code_lines = []
            for code in analysis.related_code[:5]:  # Top 5
                methods = ", ".join(code.methods[:3]) if code.methods else "N/A"
                code_lines.append(
                    f"- {code.name} ({code.component_type}, {code.package}): {code.summary or 'No description'}\n"
                    f"  Methods: {methods}"
                )
            code_summaries = "\n".join(code_lines)

        # Format regulations
        regulation_excerpts = "None found"
        if analysis.applicable_regulations:
            reg_lines = []
            for reg in analysis.applicable_regulations[:5]:  # Top 5
                content = (
                    reg.content[:200] + "..." if len(reg.content) > 200 else reg.content
                )
                reg_lines.append(f"- {reg.document} {reg.section}: {content}")
            regulation_excerpts = "\n".join(reg_lines)

        # Format existing stories
        existing_stories = "None found"
        if analysis.existing_stories:
            story_lines = []
            for story in analysis.existing_stories[:5]:  # Top 5
                story_lines.append(
                    f"- #{story.ado_id} ({story.work_item_type}, {story.state}): {story.title}"
                )
            existing_stories = "\n".join(story_lines)

        # Format integrations
        integrations = "None detected"
        if analysis.integration_dependencies:
            integrations = ", ".join(analysis.integration_dependencies)

        return STORY_GENERATION_PROMPT.format(
            feature_description=analysis.feature_description,
            code_summaries=code_summaries,
            regulation_excerpts=regulation_excerpts,
            existing_stories=existing_stories,
            integrations=integrations,
            num_stories=max_stories,
        )

    def _parse_stories(
        self,
        response: str,
        analysis: FeatureAnalysis,
    ) -> List[GeneratedStory]:
        """Parse LLM response into GeneratedStory objects."""
        stories = []

        # Split by story delimiter
        story_blocks = re.split(r"\n---\n|\n### STORY:", response)

        for block in story_blocks:
            block = block.strip()
            if not block or len(block) < 50:
                continue

            story = self._parse_single_story(block, analysis)
            if story:
                stories.append(story)

        return stories

    def _extract_title(self, block: str) -> str:
        """
        Extract title from story block.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            block: Story block text

        Returns:
            Extracted title
        """
        title_match = re.search(r"^(?:### STORY:\s*)?(.+?)$", block, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Untitled Story"
        # Remove any leading #
        return re.sub(r"^#+\s*", "", title)

    def _determine_story_type(self, block: str) -> str:
        """
        Determine story type from block.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            block: Story block text

        Returns:
            Story type (User Story, Technical Enabler, or Spike)
        """
        type_match = re.search(r"TYPE:\s*(.+?)$", block, re.MULTILINE | re.IGNORECASE)

        if not type_match:
            return "User Story"

        type_str = type_match.group(1).strip().lower()

        if "enabler" in type_str:
            return "Technical Enabler"
        if "spike" in type_str:
            return "Spike"

        return "User Story"

    def _extract_acceptance_criteria(self, block: str) -> List[AcceptanceCriterion]:
        """
        Extract acceptance criteria from block.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            block: Story block text

        Returns:
            List of AcceptanceCriterion objects
        """
        ac_match = re.search(
            r"ACCEPTANCE[_\s]?CRITERIA:\s*(.+?)(?=COMPLIANCE|RELATED|PRIORITY|$)",
            block,
            re.DOTALL | re.IGNORECASE,
        )

        if not ac_match:
            return []

        ac_text = ac_match.group(1)
        ac_items = re.findall(
            r"-\s*Given\s+(.+?),?\s*When\s+(.+?),?\s*Then\s+(.+?)(?=\n-|\n\n|$)",
            ac_text,
            re.IGNORECASE | re.DOTALL,
        )

        return [
            AcceptanceCriterion(
                given=given.strip(),
                when=when.strip(),
                then=then.strip(),
            )
            for given, when, then in ac_items
        ]

    def _extract_compliance(self, block: str) -> List[str]:
        """
        Extract compliance/regulation citations.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            block: Story block text

        Returns:
            List of regulation citations
        """
        compliance_match = re.search(
            r"COMPLIANCE:\s*(.+?)(?=RELATED|PRIORITY|$)",
            block,
            re.DOTALL | re.IGNORECASE,
        )
        if not compliance_match:
            return []

        compliance_text = compliance_match.group(1).strip()
        if compliance_text.lower() == "none":
            return []

        # Extract regulation references like "AR 601-210 4-7(a)"
        reg_refs = re.findall(
            r"(?:AR|DoDI|USAREC)\s*[\d-]+(?:\s*[\d\w.-]+)*", compliance_text
        )
        return reg_refs if reg_refs else [compliance_text]

    def _extract_related_code(self, block: str) -> List[str]:
        """
        Extract related code references.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            block: Story block text

        Returns:
            List of code references
        """
        code_match = re.search(
            r"RELATED[_\s]?CODE:\s*(.+?)(?=PRIORITY|$)",
            block,
            re.DOTALL | re.IGNORECASE,
        )
        if not code_match:
            return []

        code_text = code_match.group(1).strip()
        if code_text.lower() == "none":
            return []

        return [c.strip() for c in code_text.split(",")]

    def _extract_priority(self, block: str) -> str:
        """
        Extract priority from block.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            block: Story block text

        Returns:
            Priority (High, Medium, or Low)
        """
        priority_match = re.search(
            r"PRIORITY:\s*(.+?)$", block, re.MULTILINE | re.IGNORECASE
        )

        if not priority_match:
            return "Medium"

        p = priority_match.group(1).strip().lower()

        if "high" in p:
            return "High"
        if "low" in p:
            return "Low"

        return "Medium"

    def _parse_single_story(
        self,
        block: str,
        analysis: FeatureAnalysis,
    ) -> Optional[GeneratedStory]:
        """
        Parse a single story block.

        Rule #4: Reduced from 61 → 38 lines

        Args:
            block: Story block text
            analysis: Feature analysis context

        Returns:
            GeneratedStory or None if parsing fails
        """
        try:
            # Extract basic fields
            title = self._extract_title(block)
            story_type = self._determine_story_type(block)
            story_id = self._generate_story_id(story_type)
            description = self._extract_description_field(block, title)

            # Extract detailed fields
            acceptance_criteria = self._extract_acceptance_criteria(block)
            regulation_citations = self._extract_compliance(block)
            related_code = self._extract_related_code(block)
            priority = self._extract_priority(block)
            related_stories = [
                s.ado_id for s in analysis.existing_stories[:3] if s.ado_id
            ]

            return GeneratedStory(
                id=story_id,
                story_type=story_type,
                title=title[:100],
                description=description[:500],
                acceptance_criteria=acceptance_criteria[:5],
                regulation_citations=regulation_citations[:5],
                related_code=related_code[:5],
                related_stories=related_stories,
                priority=priority,
            )

        except Exception as e:
            logger.warning(f"Failed to parse story block: {e}")
            return None

    def _generate_story_id(self, story_type: str) -> str:
        """Rule #4: Extracted ID generation (<60 lines)."""
        type_prefix = {"User Story": "US", "Technical Enabler": "TE", "Spike": "SP"}
        prefix = type_prefix[story_type]
        self._story_counter[prefix] += 1
        return f"{prefix}-{self._story_counter[prefix]:03d}"

    def _extract_description_field(self, block: str, fallback_title: str) -> str:
        """Rule #4: Extracted description extraction (<60 lines)."""
        desc_match = re.search(
            r"DESCRIPTION:\s*(.+?)(?=ACCEPTANCE|COMPLIANCE|RELATED|PRIORITY|$)",
            block,
            re.DOTALL | re.IGNORECASE,
        )
        return desc_match.group(1).strip() if desc_match else fallback_title

    def reset_counters(self) -> None:
        """Reset story ID counters."""
        self._story_counter = {"US": 0, "TE": 0, "SP": 0}
