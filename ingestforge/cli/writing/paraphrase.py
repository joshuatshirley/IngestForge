"""Paraphrase command - Paraphrase and rewrite text.

This module provides text transformation functionality with:
- Multiple paraphrase styles
- Text rewriting with tone adjustment
- Simplification for different reading levels
- Batch file processing"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.writing.base import WritingCommand

logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes (Rule #9: Full type hints)
# ============================================================================


@dataclass
class ParaphraseResult:
    """Result of a paraphrase operation.

    Attributes:
        original: Original text
        paraphrased: Paraphrased text
        style: Style used
        maintains_meaning: Whether meaning is preserved
        word_count_change: Change in word count
    """

    original: str
    paraphrased: str
    style: str = "formal"
    maintains_meaning: bool = True
    word_count_change: int = 0

    def __post_init__(self) -> None:
        """Calculate word count change."""
        original_words = len(self.original.split())
        new_words = len(self.paraphrased.split())
        self.word_count_change = new_words - original_words

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Result as dictionary
        """
        return {
            "original": self.original,
            "paraphrased": self.paraphrased,
            "style": self.style,
            "maintains_meaning": self.maintains_meaning,
            "word_count_change": self.word_count_change,
        }


@dataclass
class RewriteResult:
    """Result of a rewrite operation.

    Attributes:
        original: Original text
        rewritten: Rewritten text
        tone: Tone used
        reading_level: Target reading level
        improvements: List of improvements made
    """

    original: str
    rewritten: str
    tone: str = "professional"
    reading_level: str = "college"
    improvements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Result as dictionary
        """
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "tone": self.tone,
            "reading_level": self.reading_level,
            "improvements": self.improvements,
        }


@dataclass
class BatchResult:
    """Result of batch processing.

    Attributes:
        results: Individual results
        total_processed: Number processed
        total_failed: Number failed
        errors: Error messages
    """

    results: List[ParaphraseResult] = field(default_factory=list)
    total_processed: int = 0
    total_failed: int = 0
    errors: List[str] = field(default_factory=list)

    def add_result(self, result: ParaphraseResult) -> None:
        """Add a successful result."""
        self.results.append(result)
        self.total_processed += 1

    def add_error(self, error: str) -> None:
        """Add an error."""
        self.errors.append(error)
        self.total_failed += 1


# ============================================================================
# Paraphraser Class (Rule #4: Small focused classes)
# ============================================================================


class Paraphraser:
    """Paraphrase and rewrite text with various styles.

    Supports:
    - Style-based paraphrasing (formal, casual, academic)
    - Tone-based rewriting
    - Simplification by reading level
    """

    # Style prompts (Rule #6: Class-level constants)
    STYLE_PROMPTS: Dict[str, str] = {
        "formal": (
            "Rewrite in formal language with proper grammar, "
            "avoiding contractions and colloquialisms."
        ),
        "casual": (
            "Rewrite in casual, conversational language. "
            "Use contractions and friendly tone."
        ),
        "academic": (
            "Rewrite in academic style with precise terminology, "
            "objective tone, and scholarly language."
        ),
        "professional": (
            "Rewrite in professional business language. "
            "Be clear, concise, and direct."
        ),
    }

    TONE_PROMPTS: Dict[str, str] = {
        "formal": "Use formal, respectful language appropriate for official documents.",
        "casual": "Use relaxed, friendly language like talking to a friend.",
        "professional": "Use clear business language that is confident and direct.",
        "academic": "Use scholarly language with precise terminology.",
        "enthusiastic": "Use energetic, positive language to engage readers.",
        "neutral": "Use balanced, objective language without emotional bias.",
    }

    READING_LEVEL_PROMPTS: Dict[str, str] = {
        "elementary": (
            "Simplify for elementary school level. "
            "Use simple words, short sentences, and basic concepts."
        ),
        "high_school": (
            "Simplify for high school level. "
            "Use clear language and explain technical terms."
        ),
        "college": (
            "Write for college level. "
            "Use appropriate academic language and complex concepts."
        ),
        "professional": (
            "Write for professional audience. "
            "Use industry terminology and sophisticated language."
        ),
    }

    def __init__(self, llm_client: Any) -> None:
        """Initialize paraphraser.

        Args:
            llm_client: LLM client for generation
        """
        self.llm_client = llm_client

    def paraphrase(self, text: str, style: str = "formal") -> ParaphraseResult:
        """Paraphrase text in specified style.

        Rule #1: Early return for empty
        Rule #4: <60 lines

        Args:
            text: Text to paraphrase
            style: Target style (formal, casual, academic, professional)

        Returns:
            ParaphraseResult with paraphrased text
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for paraphrasing")
            return ParaphraseResult(original=text, paraphrased=text, style=style)

        style = self._validate_style(style)
        style_prompt = self.STYLE_PROMPTS.get(style, self.STYLE_PROMPTS["formal"])

        prompt = f"""{style_prompt}

Original text:
{text}

Paraphrase the text while maintaining its meaning. Return only the paraphrased text."""

        paraphrased = self.llm_client.generate(prompt)

        return ParaphraseResult(
            original=text,
            paraphrased=paraphrased.strip(),
            style=style,
            maintains_meaning=True,
        )

    def paraphrase_multiple(
        self,
        text: str,
        count: int = 3,
    ) -> List[ParaphraseResult]:
        """Generate multiple paraphrase versions.

        Args:
            text: Text to paraphrase
            count: Number of versions to generate

        Returns:
            List of ParaphraseResult objects
        """
        if not text or not text.strip():
            return []

        prompt = f"""Generate {count} different paraphrases of this text:

"{text}"

Return JSON:
{{
    "paraphrases": [
        {{"version": "...", "style": "formal|casual|academic", "maintains_meaning": true}}
    ]
}}

Each version should be distinct while maintaining the original meaning."""

        response = self.llm_client.generate(prompt)
        data = self._parse_json_response(response)

        results: List[ParaphraseResult] = []
        for item in data.get("paraphrases", []):
            results.append(
                ParaphraseResult(
                    original=text,
                    paraphrased=item.get("version", ""),
                    style=item.get("style", "formal"),
                    maintains_meaning=item.get("maintains_meaning", True),
                )
            )

        return results

    def rewrite(
        self,
        text: str,
        tone: str = "professional",
        preserve_meaning: bool = True,
    ) -> RewriteResult:
        """Rewrite text with specified tone.

        Args:
            text: Text to rewrite
            tone: Target tone
            preserve_meaning: Whether to preserve original meaning

        Returns:
            RewriteResult with rewritten text
        """
        if not text or not text.strip():
            return RewriteResult(original=text, rewritten=text, tone=tone)

        tone = self._validate_tone(tone)
        tone_prompt = self.TONE_PROMPTS.get(tone, self.TONE_PROMPTS["professional"])

        meaning_instruction = ""
        if preserve_meaning:
            meaning_instruction = " Preserve the original meaning and key points."

        prompt = f"""{tone_prompt}{meaning_instruction}

Original text:
{text}

Rewrite the text with the specified tone. Return JSON:
{{
    "rewritten": "...",
    "improvements": ["improvement 1", "improvement 2"]
}}"""

        response = self.llm_client.generate(prompt)
        data = self._parse_json_response(response)

        # Handle both JSON and plain text responses
        rewritten = (
            data.get("rewritten", response) if isinstance(data, dict) else response
        )

        return RewriteResult(
            original=text,
            rewritten=rewritten.strip()
            if isinstance(rewritten, str)
            else str(rewritten),
            tone=tone,
            improvements=data.get("improvements", []) if isinstance(data, dict) else [],
        )

    def simplify(
        self,
        text: str,
        reading_level: str = "high_school",
    ) -> RewriteResult:
        """Simplify text for target reading level.

        Args:
            text: Text to simplify
            reading_level: Target level (elementary, high_school, college, professional)

        Returns:
            RewriteResult with simplified text
        """
        if not text or not text.strip():
            return RewriteResult(
                original=text, rewritten=text, reading_level=reading_level
            )

        reading_level = self._validate_reading_level(reading_level)
        level_prompt = self.READING_LEVEL_PROMPTS.get(
            reading_level,
            self.READING_LEVEL_PROMPTS["high_school"],
        )

        prompt = f"""{level_prompt}

Original text:
{text}

Simplify the text for the target reading level. Return only the simplified text."""

        simplified = self.llm_client.generate(prompt)

        return RewriteResult(
            original=text,
            rewritten=simplified.strip(),
            reading_level=reading_level,
        )

    def batch_process(
        self,
        texts: List[str],
        style: str = "formal",
    ) -> BatchResult:
        """Process multiple texts.

        Args:
            texts: List of texts to process
            style: Style to apply

        Returns:
            BatchResult with all results
        """
        batch_result = BatchResult()

        for text in texts:
            try:
                result = self.paraphrase(text, style)
                batch_result.add_result(result)
            except Exception as e:
                logger.error(f"Failed to paraphrase text: {e}")
                batch_result.add_error(str(e))

        return batch_result

    def process_file(
        self,
        file_path: Path,
        style: str = "formal",
    ) -> BatchResult:
        """Process a file (each line as separate text).

        Args:
            file_path: Path to input file
            style: Style to apply

        Returns:
            BatchResult with all results
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            result = BatchResult()
            result.add_error(f"File not found: {file_path}")
            return result

        content = file_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        return self.batch_process(lines, style)

    def _validate_style(self, style: str) -> str:
        """Validate and normalize style.

        Args:
            style: Input style

        Returns:
            Validated style
        """
        style = style.lower()
        if style not in self.STYLE_PROMPTS:
            logger.warning(f"Unknown style '{style}', using 'formal'")
            return "formal"
        return style

    def _validate_tone(self, tone: str) -> str:
        """Validate and normalize tone.

        Args:
            tone: Input tone

        Returns:
            Validated tone
        """
        tone = tone.lower()
        if tone not in self.TONE_PROMPTS:
            logger.warning(f"Unknown tone '{tone}', using 'professional'")
            return "professional"
        return tone

    def _validate_reading_level(self, level: str) -> str:
        """Validate and normalize reading level.

        Args:
            level: Input level

        Returns:
            Validated level
        """
        level = level.lower().replace(" ", "_")
        if level not in self.READING_LEVEL_PROMPTS:
            logger.warning(f"Unknown reading level '{level}', using 'high_school'")
            return "high_school"
        return level

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response.

        Args:
            response: LLM response

        Returns:
            Parsed data or empty dict
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try extracting JSON from response
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return {}  # Could not parse extracted JSON
            return {}  # No JSON found in response


# ============================================================================
# Command Implementation
# ============================================================================


class ParaphraseCommand(WritingCommand):
    """Generate paraphrase suggestions CLI command."""

    def execute(
        self,
        text: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        style: str = "formal",
        count: int = 3,
    ) -> int:
        """Execute paraphrase command.

        Args:
            text: Text to paraphrase
            project: Project directory
            output: Output file path
            style: Paraphrase style
            count: Number of versions

        Returns:
            Exit code
        """
        try:
            return self._execute_paraphrase(text, project, output, style, count)
        except Exception as e:
            logger.error(f"Paraphrase failed: {e}")
            return self.handle_error(e, "Paraphrase generation failed")

    def _execute_paraphrase(
        self,
        text: str,
        project: Optional[Path],
        output: Optional[Path],
        style: str,
        count: int,
    ) -> int:
        """Internal paraphrase logic."""
        ctx = self.initialize_context(project, require_storage=True)
        llm_client = self.get_llm_client(ctx)

        if not llm_client:
            self.print_error("No LLM client available")
            return 1

        paraphraser = Paraphraser(llm_client)
        results = self._generate_paraphrases(paraphraser, text, count)

        self._display_results(text, results)

        if output:
            self._save_results(output, text, results)

        return 0

    def _generate_paraphrases(
        self,
        paraphraser: Paraphraser,
        text: str,
        count: int,
    ) -> List[ParaphraseResult]:
        """Generate paraphrases with progress."""
        from ingestforge.cli.core import ProgressManager

        return ProgressManager.run_with_spinner(
            lambda: paraphraser.paraphrase_multiple(text, count),
            "Generating paraphrases...",
            "Complete",
        )

    def _display_results(
        self,
        original: str,
        results: List[ParaphraseResult],
    ) -> None:
        """Display paraphrase results."""
        self.console.print()
        self.console.print(
            Panel(f"[bold]Original:[/bold]\n{original}", border_style="blue")
        )
        self.console.print()

        table = Table(title="Paraphrases")
        table.add_column("#", width=5)
        table.add_column("Paraphrase", width=60)
        table.add_column("Style", width=12)

        for idx, result in enumerate(results, 1):
            table.add_row(str(idx), result.paraphrased, result.style)

        self.console.print(table)

    def _save_results(
        self,
        output: Path,
        original: str,
        results: List[ParaphraseResult],
    ) -> None:
        """Save results to file."""
        data = {
            "original": original,
            "paraphrases": [r.to_dict() for r in results],
        }
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.print_success(f"Paraphrases saved to: {output}")


class RewriteCommand(WritingCommand):
    """Rewrite text with tone adjustment CLI command."""

    def execute(
        self,
        input_path: Path,
        output: Optional[Path] = None,
        project: Optional[Path] = None,
        tone: str = "professional",
        preserve_meaning: bool = True,
    ) -> int:
        """Execute rewrite command."""
        try:
            return self._execute_rewrite(
                input_path, output, project, tone, preserve_meaning
            )
        except Exception as e:
            logger.error(f"Rewrite failed: {e}")
            return self.handle_error(e, "Rewrite failed")

    def _execute_rewrite(
        self,
        input_path: Path,
        output: Optional[Path],
        project: Optional[Path],
        tone: str,
        preserve_meaning: bool,
    ) -> int:
        """Internal rewrite logic."""
        if not input_path.exists():
            self.print_error(f"File not found: {input_path}")
            return 1

        ctx = self.initialize_context(project, require_storage=True)
        llm_client = self.get_llm_client(ctx)

        if not llm_client:
            self.print_error("No LLM client available")
            return 1

        text = input_path.read_text(encoding="utf-8")
        paraphraser = Paraphraser(llm_client)

        result = self._do_rewrite(paraphraser, text, tone, preserve_meaning)
        self._display_rewrite_result(result)

        if output:
            output.write_text(result.rewritten, encoding="utf-8")
            self.print_success(f"Rewritten text saved to: {output}")

        return 0

    def _do_rewrite(
        self,
        paraphraser: Paraphraser,
        text: str,
        tone: str,
        preserve_meaning: bool,
    ) -> RewriteResult:
        """Perform rewrite with progress."""
        from ingestforge.cli.core import ProgressManager

        return ProgressManager.run_with_spinner(
            lambda: paraphraser.rewrite(text, tone, preserve_meaning),
            f"Rewriting with {tone} tone...",
            "Complete",
        )

    def _display_rewrite_result(self, result: RewriteResult) -> None:
        """Display rewrite result."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]Tone:[/bold] {result.tone}",
                title="Rewrite Summary",
                border_style="green",
            )
        )
        self.console.print()
        self.console.print(result.rewritten)

        if result.improvements:
            self.console.print("\n[bold]Improvements Made:[/bold]")
            for imp in result.improvements:
                self.console.print(f"  - {imp}")


# ============================================================================
# CLI Command Functions
# ============================================================================


def command(
    text: str = typer.Argument(..., help="Text to paraphrase"),
    project: Optional[Path] = typer.Option(
        None, "-p", "--project", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file"),
    style: str = typer.Option(
        "formal",
        "-s",
        "--style",
        help="Paraphrase style (formal, casual, academic, professional)",
    ),
    count: int = typer.Option(3, "-n", "--count", help="Number of paraphrases"),
) -> None:
    """Generate paraphrase suggestions.

    Examples:
        # Generate 3 paraphrases
        ingestforge writing paraphrase "The quick brown fox jumps."

        # Casual style
        ingestforge writing paraphrase "The text to paraphrase" -s casual

        # Save to file
        ingestforge writing paraphrase "Text here" -o paraphrases.json
    """
    exit_code = ParaphraseCommand().execute(text, project, output, style, count)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def rewrite_command(
    input_file: Path = typer.Argument(..., help="Input file to rewrite"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file"),
    project: Optional[Path] = typer.Option(
        None, "-p", "--project", help="Project directory"
    ),
    tone: str = typer.Option(
        "professional",
        "-t",
        "--tone",
        help="Target tone (formal, casual, professional, academic, enthusiastic, neutral)",
    ),
    preserve: bool = typer.Option(
        True,
        "--preserve/--no-preserve",
        help="Preserve original meaning",
    ),
) -> None:
    """Rewrite text with specified tone.

    Examples:
        # Rewrite with professional tone
        ingestforge writing rewrite input.txt --tone professional

        # Save rewritten text
        ingestforge writing rewrite input.txt -o output.txt -t casual
    """
    exit_code = RewriteCommand().execute(input_file, output, project, tone, preserve)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def simplify_command(
    input_file: Path = typer.Argument(..., help="Input file to simplify"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file"),
    project: Optional[Path] = typer.Option(
        None, "-p", "--project", help="Project directory"
    ),
    level: str = typer.Option(
        "high_school",
        "-l",
        "--level",
        help="Reading level (elementary, high_school, college, professional)",
    ),
) -> None:
    """Simplify text for target reading level.

    Examples:
        # Simplify for high school level
        ingestforge writing simplify document.txt -l high_school

        # Simplify for elementary
        ingestforge writing simplify complex.txt -l elementary -o simple.txt
    """
    if not input_file.exists():
        raise typer.BadParameter(f"File not found: {input_file}")

    # Reuse rewrite infrastructure
    exit_code = RewriteCommand().execute(
        input_file, output, project, level, preserve_meaning=True
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
