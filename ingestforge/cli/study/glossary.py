"""Glossary command - Auto-extract key terms and definitions.

Generates glossaries of important terms from your knowledge base.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import typer
from rich.table import Table
from rich.panel import Panel

from ingestforge.cli.study.base import StudyCommand
from ingestforge.cli.study.glossary_parser import (
    GlossaryTextParser,
    TermDeduplicator,
    GlossaryValidator,
    CategoryInferrer,
    ParsedTerm,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class GlossaryCommand(StudyCommand):
    """Generate glossary of key terms."""

    # Maximum retry attempts for LLM generation
    MAX_RETRIES = 2

    def execute(
        self,
        topic: Optional[str] = None,
        count: int = 30,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """
        Generate glossary of key terms.

        Rule #4: Function under 60 lines
        """
        try:
            # Validate all inputs (Commandments #5, #7)
            validation_error = self._validate_execute_params(count, project, output)
            if validation_error is not None:
                return validation_error

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Get LLM client
            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            # Search for context
            if topic:
                self.validate_topic(topic)
                chunks = self.search_topic_context(ctx["storage"], topic, k=50)
            else:
                # Get general sample for overall glossary
                chunks = self._get_sample_chunks(ctx["storage"], k=50)

            if not chunks:
                self._handle_no_context(topic)
                return 0

            # Generate glossary with retry logic
            glossary_data = self._try_generate_with_retry(
                llm_client, chunks, count, topic
            )

            if not glossary_data:
                self.print_error("Failed to generate glossary")
                return 1

            # Display glossary
            self._display_glossary(glossary_data, topic)

            # Save to file if requested
            if output:
                self._save_glossary(output, glossary_data)

            return 0

        except Exception as e:
            return self.handle_error(e, "Glossary generation failed")

    def _validate_execute_params(
        self,
        count: int,
        project: Optional[Path],
        output: Optional[Path],
    ) -> Optional[int]:
        """Validate execute() parameters.

        Args:
            count: Number of terms
            project: Optional project path
            output: Optional output path

        Returns:
            Error code if validation fails, None if valid
        """
        # Commandment #5: Assertion density - precondition assertions
        assert count > 0, f"count must be positive, got {count}"

        # Commandment #7: Check parameters - validate inputs
        self.validate_count(count, min_val=5, max_val=100)

        # Validate project path if provided
        if project is not None:
            if not project.exists():
                self.print_error(f"Project path does not exist: {project}")
                return 1
            if not project.is_dir():
                self.print_error(f"Project path is not a directory: {project}")
                return 1

        # Validate output path suffix if provided
        if output is not None:
            valid_suffixes = {".json", ".md"}
            if output.suffix.lower() not in valid_suffixes:
                self.print_error(f"Output must be .json or .md, got: {output.suffix}")
                return 1

        return None

    def _get_sample_chunks(self, storage: Any, k: int) -> list[Any]:
        """Get sample chunks for general glossary.

        Args:
            storage: Storage instance
            k: Number of chunks

        Returns:
            Sample chunks
        """
        return storage.search("", k=k)

    def _handle_no_context(self, topic: Optional[str]) -> None:
        """Handle case where no context found.

        Args:
            topic: Optional topic
        """
        if topic:
            self.print_warning(f"No context found for topic: '{topic}'")
        else:
            self.print_warning("No content found in knowledge base")

        self.print_info(
            "Try:\n"
            "  1. Ingesting documents first\n"
            "  2. Using a different topic\n"
            "  3. Checking project path"
        )

    def _try_generate_with_retry(
        self,
        llm_client: Any,
        chunks: list,
        count: int,
        topic: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Generate glossary with retry on failure.

        Uses different prompts on retry for better chances of success.

        Args:
            llm_client: LLM provider instance
            chunks: Context chunks
            count: Number of terms
            topic: Optional topic filter

        Returns:
            Glossary data dict or None
        """
        context = self.format_context_for_prompt(chunks, max_length=5000)
        prompts = self._build_prompt_variants(count, topic, context)

        for attempt, prompt in enumerate(prompts):
            logger.debug(
                f"Glossary generation attempt {attempt + 1}/{len(prompts)}",
                attempt=attempt + 1,
                total_attempts=len(prompts),
            )

            glossary_data = self._generate_glossary(
                llm_client, prompt, count, topic, attempt
            )

            if glossary_data and glossary_data.get("terms"):
                return glossary_data

            if attempt < len(prompts) - 1:
                logger.warning(
                    f"Attempt {attempt + 1} failed, trying simplified prompt",
                    attempt=attempt + 1,
                )

        return None

    def _build_prompt_variants(
        self, count: int, topic: Optional[str], context: str
    ) -> List[str]:
        """Build list of prompts to try (detailed first, then simplified).

        Args:
            count: Number of terms
            topic: Optional topic
            context: Context from chunks

        Returns:
            List of prompts to try in order
        """
        return [
            self._build_glossary_prompt(count, topic, context),
            self._build_simplified_prompt(count, topic, context),
        ]

    def _generate_glossary(
        self,
        llm_client: Any,
        prompt: str,
        count: int,
        topic: Optional[str],
        attempt: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Generate glossary using LLM.

        Args:
            llm_client: LLM provider instance
            prompt: Prompt to use
            count: Number of terms
            topic: Optional topic filter
            attempt: Current attempt number

        Returns:
            Glossary data dict or None
        """
        attempt_str = f" (attempt {attempt + 1})" if attempt > 0 else ""

        # Generate glossary
        response = self.generate_with_llm(
            llm_client, prompt, f"glossary ({count} terms){attempt_str}"
        )

        # Try JSON parsing first
        glossary_data = self.parse_json_response(response)

        if glossary_data:
            # Validate and clean JSON response terms
            glossary_data = self._validate_json_glossary(glossary_data, count)
            if glossary_data and glossary_data.get("terms"):
                logger.info(
                    "Successfully parsed JSON glossary",
                    terms=len(glossary_data.get("terms", [])),
                )
                return glossary_data

        # Fallback: parse as text
        logger.debug("JSON parsing failed, falling back to text parsing")
        glossary_data = self._parse_text_glossary(response, count, topic)

        return glossary_data

    def _validate_json_glossary(
        self, data: Dict[str, Any], count: int
    ) -> Dict[str, Any]:
        """Validate and clean JSON glossary response.

        Args:
            data: Parsed JSON data
            count: Expected count

        Returns:
            Validated glossary data
        """
        terms = data.get("terms", [])
        if not terms:
            return data

        # Convert to ParsedTerm objects
        parsed_terms: List[ParsedTerm] = []
        for term_dict in terms:
            if isinstance(term_dict, dict) and term_dict.get("term"):
                parsed_terms.append(
                    ParsedTerm(
                        term=term_dict.get("term", ""),
                        definition=term_dict.get("definition", ""),
                        category=term_dict.get("category", "general"),
                        related_terms=term_dict.get("related_terms", [])[:5],
                    )
                )

        # Validate
        validator = GlossaryValidator()
        valid_terms, invalid_count = validator.validate_and_clean(parsed_terms)

        # Deduplicate
        deduplicator = TermDeduplicator()
        deduped_terms = deduplicator.deduplicate(valid_terms)

        # Convert back to dict format
        data["terms"] = [t.to_dict() for t in deduped_terms]
        data["term_count"] = len(deduped_terms)

        return data

    def _build_glossary_prompt(
        self, count: int, topic: Optional[str], context: str
    ) -> str:
        """Build enhanced prompt for glossary generation.

        Args:
            count: Number of terms
            topic: Optional topic
            context: Context from chunks

        Returns:
            Formatted prompt
        """
        topic_str = f" about {topic}" if topic else ""
        topic_value = topic if topic else "General"

        prompt_parts = [
            "=== TASK ===\n",
            f"Extract and define {count} key terms{topic_str} from the provided context.\n\n",
            "=== CONTEXT ===\n",
            context,
            "\n\n=== OUTPUT FORMAT ===\n",
            "Return a JSON object with this exact structure:\n\n",
            "```json\n",
            "{\n",
            f'  "topic": "{topic_value}",\n',
            f'  "term_count": {count},\n',
            '  "terms": [\n',
            "    {\n",
            '      "term": "Algorithm",\n',
            '      "definition": "A step-by-step procedure or formula for solving a problem. Algorithms are fundamental to computer science and are used to process data, perform calculations, and automate reasoning tasks.",\n',
            '      "category": "computer science",\n',
            '      "related_terms": ["function", "data structure"]\n',
            "    }\n",
            "  ]\n",
            "}\n",
            "```\n\n",
            "=== REQUIREMENTS ===\n",
            f"1. Extract exactly {count} of the most important terms from the context\n",
            "2. Each definition should be 10-100 words (concise but complete)\n",
            "3. Sort terms alphabetically\n",
            "4. Assign appropriate categories (e.g., computer science, mathematics, science, business)\n",
            "5. Include 0-3 related terms per entry when applicable\n",
            "6. Focus on key concepts, not trivial or common words\n",
            "7. Definitions should be clear and educational\n\n",
            "Return ONLY the JSON object. No additional text, explanation, or markdown outside the JSON.\n",
        ]

        return "".join(prompt_parts)

    def _build_simplified_prompt(
        self, count: int, topic: Optional[str], context: str
    ) -> str:
        """Build simplified fallback prompt.

        Args:
            count: Number of terms
            topic: Optional topic
            context: Context from chunks

        Returns:
            Simplified prompt
        """
        topic_str = f" about {topic}" if topic else ""

        prompt_parts = [
            f"Create a glossary of {count} key terms{topic_str}.\n\n",
            "Context:\n",
            context[:3000],  # Shorter context for simplified prompt
            "\n\n",
            "For each term, provide:\n",
            "- Term name\n",
            "- Clear definition (20-50 words)\n",
            "- Category\n\n",
            "Format each entry as:\n",
            "**Term Name**: Definition goes here.\n\n",
            f"List exactly {count} terms, sorted alphabetically.\n",
        ]

        return "".join(prompt_parts)

    def _parse_text_glossary(
        self, response: str, count: int, topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """Parse glossary from plain text response using robust parser.

        Args:
            response: LLM response text
            count: Expected term count
            topic: Optional topic

        Returns:
            Glossary data dict
        """
        logger.debug(
            "Parsing text response",
            response_length=len(response),
            expected_count=count,
        )

        # Parse using robust parser
        parser = GlossaryTextParser()
        terms = parser.parse(response)

        if not terms:
            logger.warning(
                "Text parser found no terms",
                response_preview=response[:200] if response else "empty",
            )

        # Deduplicate
        deduplicator = TermDeduplicator()
        terms = deduplicator.deduplicate(terms)

        # Infer categories for terms without them
        inferrer = CategoryInferrer()
        terms = inferrer.infer_categories(terms)

        # Validate
        validator = GlossaryValidator()
        terms, invalid_count = validator.validate_and_clean(terms)

        # Limit to requested count
        terms = terms[:count]

        logger.info(
            "Text parsing complete",
            terms_found=len(terms),
            requested=count,
        )

        return {
            "topic": topic if topic else "General",
            "term_count": len(terms),
            "terms": [t.to_dict() for t in terms],
        }

    def _display_glossary(
        self, glossary_data: Dict[str, Any], topic: Optional[str]
    ) -> None:
        """Display glossary.

        Args:
            glossary_data: Glossary data dict
            topic: Optional topic
        """
        self.console.print()

        # Title
        topic_str = topic if topic else glossary_data.get("topic", "General")
        title = f"[bold cyan]Glossary: {topic_str}[/bold cyan]"
        self.console.print(Panel(title, border_style="cyan"))

        self.console.print()

        # Terms
        terms = glossary_data.get("terms", [])

        if not terms:
            self.print_warning("No terms found in glossary")
            return

        # Display as table
        table = Table(title=f"{len(terms)} Key Terms", show_lines=True)
        table.add_column("Term", style="yellow", width=25)
        table.add_column("Definition", style="white", width=60)
        table.add_column("Category", style="cyan", width=15)

        for term_data in terms:
            term = term_data.get("term", "")
            definition = term_data.get("definition", "")
            category = term_data.get("category", "-")

            # Truncate long definitions
            if len(definition) > 200:
                definition = definition[:197] + "..."

            table.add_row(term, definition, category)

        self.console.print(table)

        # Related terms if available
        self._display_related_terms(terms)

    def _display_related_terms(self, terms: List[Dict]) -> None:
        """Display related terms info.

        Args:
            terms: List of term dicts
        """
        # Count terms with related links
        with_related = sum(1 for t in terms if t.get("related_terms", []))

        if with_related > 0:
            self.console.print()
            self.print_info(f"{with_related} terms have related concepts listed")

    def _save_glossary(self, output: Path, glossary_data: Dict[str, Any]) -> None:
        """Save glossary to file.

        Args:
            output: Output path
            glossary_data: Glossary data
        """
        # Determine format from extension
        if output.suffix.lower() == ".md":
            self._save_markdown_glossary(output, glossary_data)
        else:
            # Default to JSON
            self.save_json_output(output, glossary_data, f"Glossary saved to: {output}")

    def _save_markdown_glossary(
        self, output: Path, glossary_data: Dict[str, Any]
    ) -> None:
        """Save glossary as markdown.

        Args:
            output: Output path
            glossary_data: Glossary data
        """
        topic = glossary_data.get("topic", "General")
        terms = glossary_data.get("terms", [])

        md_parts = [
            f"# Glossary: {topic}\n",
            f"\n**Total Terms**: {len(terms)}\n",
            "\n---\n\n",
        ]

        # Group by category (Commandment #1: extract to reduce nesting)
        categorized = self._group_terms_by_category(terms)

        # Format each category
        for category in sorted(categorized.keys()):
            md_parts.extend(
                self._format_category_markdown(category, categorized[category])
            )

        content = "".join(md_parts)
        output.write_text(content, encoding="utf-8")
        self.print_success(f"Glossary saved to: {output}")

    def _group_terms_by_category(self, terms: List[Dict]) -> Dict[str, List[Dict]]:
        """Group terms by their category.

        Args:
            terms: List of term dicts

        Returns:
            Dict mapping category to list of terms
        """
        categorized: Dict[str, List[Dict]] = {}
        for term_data in terms:
            category = term_data.get("category", "General")
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(term_data)
        return categorized

    def _format_category_markdown(self, category: str, terms: List[Dict]) -> List[str]:
        """Format a category section as markdown.

        Args:
            category: Category name
            terms: Terms in this category

        Returns:
            List of markdown strings
        """
        parts: List[str] = []

        if category != "General":
            parts.append(f"## {category}\n\n")

        for term_data in terms:
            parts.extend(self._format_term_markdown(term_data))

        return parts

    def _format_term_markdown(self, term_data: Dict) -> List[str]:
        """Format a single term as markdown.

        Args:
            term_data: Term dict

        Returns:
            List of markdown strings
        """
        term = term_data.get("term", "")
        definition = term_data.get("definition", "")
        related = term_data.get("related_terms", [])

        parts = [f"**{term}**  \n", f"{definition}\n"]

        if related:
            parts.append(f"*Related: {', '.join(related)}*\n")

        parts.append("\n")
        return parts


# Typer command wrapper
def command(
    topic: Optional[str] = typer.Argument(
        None, help="Optional topic to focus glossary on"
    ),
    count: int = typer.Option(
        30, "--count", "-n", help="Number of terms to include (5-100)"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (.json or .md)"
    ),
) -> None:
    """Generate glossary of key terms and definitions.

    Auto-extracts important terms from your knowledge base and
    provides clear definitions.

    Examples:
        # Generate general glossary
        ingestforge study glossary

        # Topic-specific glossary
        ingestforge study glossary "Machine Learning"

        # More terms
        ingestforge study glossary --count 50

        # Save to file
        ingestforge study glossary -o glossary.md

        # JSON format
        ingestforge study glossary -o terms.json

        # Specific project
        ingestforge study glossary "Biology" -p /path/to/project
    """
    cmd = GlossaryCommand()
    exit_code = cmd.execute(topic, count, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
