"""Clean command - Clean and normalize text.

Cleans and normalizes text content for processing.
Integrates TextCleanerRefiner for advanced OCR cleanup (Phase 4).

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
import typer

from ingestforge.cli.transform.base import TransformCommand


class CleanCommand(TransformCommand):
    """Clean and normalize text with advanced OCR cleanup."""

    def _build_clean_results(
        self, input_file: Path, stats: dict[str, Any], changes: List[str]
    ) -> dict[str, Any]:
        """Build results dictionary for clean operation.

        Rule #4: No large functions - Extracted from execute
        """
        details = [
            f"File: {input_file.name}",
            f"Original: {stats['original_chars']} chars, {stats['original_words']} words",
            f"Cleaned: {stats['cleaned_chars']} chars, {stats['cleaned_words']} words",
            f"Reduction: {stats['reduction_pct']:.1f}%",
        ]
        # Add TextCleanerRefiner changes
        if changes:
            details.extend([f"  - {c}" for c in changes])

        return {
            "input_count": 1,
            "output_count": 1,
            "successful": 1,
            "failed": 0,
            "errors": [],
            "details": details,
        }

    def _output_cleaned_content(
        self, output: Optional[Path], cleaned_content: str
    ) -> None:
        """Output cleaned content to file or preview.

        Rule #4: No large functions - Extracted from execute
        """
        if output:
            self._save_cleaned(output, cleaned_content)
        else:
            self.print_info("\n[dim]Preview (first 500 chars):[/dim]")
            self.console.print(cleaned_content[:500])

    def execute(
        self,
        input_file: Path,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        remove_urls: bool = False,
        remove_emails: bool = False,
        lowercase: bool = False,
        group_paragraphs: bool = False,
        clean_bullets: bool = False,
        clean_prefix_postfix: bool = False,
    ) -> int:
        """Clean and normalize text file.

        Rule #4: Function <60 lines

        Args:
            input_file: Input file to clean
            project: Project directory
            output: Output file for cleaned text
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            lowercase: Convert to lowercase
            group_paragraphs: Join broken paragraphs (OCR fix)
            clean_bullets: Normalize bullet characters
            clean_prefix_postfix: Remove page numbers/headers

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs (Commandment #7: Check parameters)
            self.validate_file_path(input_file, must_exist=True)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)

            # Read and clean file
            original_content = self._read_file(input_file)

            # Apply TextCleanerRefiner if any advanced options enabled
            refiner_changes: List[str] = []
            cleaned_content = self._apply_text_cleaner(
                original_content,
                group_paragraphs,
                clean_bullets,
                clean_prefix_postfix,
                refiner_changes,
            )

            # Apply basic cleaning
            cleaned_content = self._clean_content(
                cleaned_content, remove_urls, remove_emails, lowercase
            )

            # Calculate stats and display
            stats = self._calculate_stats(original_content, cleaned_content)
            results = self._build_clean_results(input_file, stats, refiner_changes)
            summary = self.create_transform_summary(results, "Clean")
            self.console.print(summary)

            # Output cleaned content
            self._output_cleaned_content(output, cleaned_content)

            return 0

        except Exception as e:
            return self.handle_error(e, "Clean operation failed")

    def _apply_text_cleaner(
        self,
        content: str,
        group_paragraphs: bool,
        clean_bullets: bool,
        clean_prefix_postfix: bool,
        changes: List[str],
    ) -> str:
        """Apply TextCleanerRefiner for advanced OCR cleanup.

        Rule #4: Function <60 lines

        Args:
            content: Content to clean
            group_paragraphs: Join broken paragraphs
            clean_bullets: Normalize bullet characters
            clean_prefix_postfix: Remove page numbers/headers
            changes: List to append change descriptions

        Returns:
            Cleaned content
        """
        # Only apply if any option is enabled
        if not (group_paragraphs or clean_bullets or clean_prefix_postfix):
            return content

        from ingestforge.ingest.refiners.text_cleaners import TextCleanerRefiner

        cleaner = TextCleanerRefiner(
            group_paragraphs=group_paragraphs,
            clean_bullets=clean_bullets,
            clean_prefix_postfix=clean_prefix_postfix,
        )

        result = cleaner.refine(content)
        changes.extend(result.changes or [])

        return result.refined

    def _read_file(self, file_path: Path) -> str:
        """Read file content.

        Args:
            file_path: File to read

        Returns:
            File content

        Raises:
            ValueError: If file cannot be read
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Cannot read file: {e}")

    def _clean_content(
        self,
        content: str,
        remove_urls: bool,
        remove_emails: bool,
        lowercase: bool,
    ) -> str:
        """Clean content with specified options.

        Args:
            content: Content to clean
            remove_urls: Remove URLs
            remove_emails: Remove emails
            lowercase: Convert to lowercase

        Returns:
            Cleaned content
        """
        import re

        # Basic cleaning
        content = self.clean_text_simple(content)
        content = self.normalize_whitespace(content)

        # Remove URLs if requested
        if remove_urls:
            content = re.sub(r"https?://\S+|www\.\S+", "", content)

        # Remove emails if requested
        if remove_emails:
            content = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "",
                content,
            )

        # Convert to lowercase if requested
        if lowercase:
            content = content.lower()

        # Final cleanup
        content = self.clean_text_simple(content)
        content = self.normalize_whitespace(content)

        return content

    def _calculate_stats(self, original: str, cleaned: str) -> dict[str, Any]:
        """Calculate cleaning statistics.

        Args:
            original: Original content
            cleaned: Cleaned content

        Returns:
            Statistics dictionary
        """
        original_chars = len(original)
        cleaned_chars = len(cleaned)
        original_words = len(original.split())
        cleaned_words = len(cleaned.split())

        reduction = original_chars - cleaned_chars
        reduction_pct = (reduction / original_chars * 100) if original_chars > 0 else 0

        return {
            "original_chars": original_chars,
            "cleaned_chars": cleaned_chars,
            "original_words": original_words,
            "cleaned_words": cleaned_words,
            "reduction_pct": reduction_pct,
        }

    def _save_cleaned(self, output: Path, content: str) -> None:
        """Save cleaned content.

        Args:
            output: Output file path
            content: Content to save
        """
        try:
            output.write_text(content, encoding="utf-8")
            self.print_success(f"Cleaned content saved: {output}")

        except Exception as e:
            self.print_error(f"Failed to save cleaned content: {e}")


# Typer command wrapper
def command(
    input_file: Path = typer.Argument(..., help="Input file to clean"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for cleaned text"
    ),
    remove_urls: bool = typer.Option(
        False, "--remove-urls", help="Remove URLs from text"
    ),
    remove_emails: bool = typer.Option(
        False, "--remove-emails", help="Remove email addresses"
    ),
    lowercase: bool = typer.Option(
        False, "--lowercase", help="Convert text to lowercase"
    ),
    group_paragraphs: bool = typer.Option(
        False, "--group-paragraphs", help="Join broken paragraphs (OCR fix)"
    ),
    clean_bullets: bool = typer.Option(
        False, "--clean-bullets", help="Normalize bullet characters"
    ),
    clean_prefix_postfix: bool = typer.Option(
        False, "--clean-prefix-postfix", help="Remove page numbers/headers"
    ),
) -> None:
    """Clean and normalize text file.

    Cleans and normalizes text content by removing extra whitespace,
    standardizing formatting, and optionally removing URLs and emails.

    Features:
    - Whitespace normalization
    - Line ending standardization
    - Optional URL removal
    - Optional email removal
    - Optional lowercase conversion
    - OCR cleanup: join broken paragraphs (--group-paragraphs)
    - OCR cleanup: normalize bullets (--clean-bullets)
    - OCR cleanup: remove page numbers (--clean-prefix-postfix)
    - Statistics reporting

    Examples:
        # Basic cleaning
        ingestforge transform clean document.txt -o clean.txt

        # Remove URLs and emails
        ingestforge transform clean scraped.txt --remove-urls --remove-emails -o clean.txt

        # OCR cleanup - fix broken paragraphs
        ingestforge transform clean ocr_output.txt --group-paragraphs -o clean.txt

        # Full OCR cleanup
        ingestforge transform clean scan.txt --group-paragraphs --clean-bullets \\
            --clean-prefix-postfix -o clean.txt

        # Preview without saving
        ingestforge transform clean document.txt

        # Specific project
        ingestforge transform clean doc.txt -p /path/to/project -o clean.txt
    """
    cmd = CleanCommand()
    exit_code = cmd.execute(
        input_file,
        project,
        output,
        remove_urls,
        remove_emails,
        lowercase,
        group_paragraphs,
        clean_bullets,
        clean_prefix_postfix,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
