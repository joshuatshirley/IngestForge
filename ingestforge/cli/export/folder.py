"""Folder export command - Generate complete study package.

Creates a comprehensive study folder with overview, glossary, notes,
flashcards, quiz, concept map, reading list, and bibliography.

This is IngestForge's PRIMARY differentiator - 95% of users rely on this.

Implements UX-003: Progress Indicators with ETA, quiet mode, and CI fallback.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, List, Callable
import typer
from datetime import datetime

from ingestforge.cli.export.base import ExportCommand
from ingestforge.cli.core.progress import ProgressReporter


class FolderExportCommand(ExportCommand):
    """Generate complete study package in a folder."""

    def execute(
        self,
        output_dir: Path,
        topic: Optional[str] = None,
        project: Optional[Path] = None,
        include_all: bool = True,
        llm_provider: Optional[str] = None,
        include_flashcards: bool = True,
        include_quiz: bool = True,
        quiet: bool = False,
    ) -> int:
        """
        Export complete study package to folder.

        Implements UX-003: Progress Indicators with ETA and quiet mode.

        Rule #4: Function under 60 lines

        Args:
            output_dir: Output directory for package
            topic: Optional topic name for package
            project: Optional project directory
            include_all: Include all components
            llm_provider: Optional LLM provider override
            include_flashcards: Include flashcard CSV
            include_quiz: Include quiz markdown
            quiet: Suppress progress bar output
        """
        try:
            self._validate_output_directory(output_dir)
            ctx = self.initialize_context(project, require_storage=True)
            llm_client = self._get_llm_client(ctx, llm_provider)
            chunks = self.get_all_chunks_from_storage(ctx["storage"])

            if not chunks:
                self.print_warning("No content found in knowledge base")
                return 0

            output_dir.mkdir(parents=True, exist_ok=True)

            self._generate_study_package(
                output_dir,
                chunks,
                topic,
                llm_client,
                include_all,
                include_flashcards,
                include_quiz,
                quiet,
            )
            if not quiet:
                self._display_package_summary(
                    output_dir, include_flashcards, include_quiz
                )
            return 0

        except Exception as e:
            return self.handle_error(e, "Folder export failed")

    def _validate_output_directory(self, output_dir: Path) -> None:
        """Validate output directory path.

        Args:
            output_dir: Output directory path

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if output_dir.exists() and not output_dir.is_dir():
            raise typer.BadParameter(
                f"Output path exists but is not a directory: {output_dir}"
            )

        if output_dir.exists() and any(output_dir.iterdir()):
            self.print_warning(
                f"Directory already exists and contains files: {output_dir}"
            )

    def _get_llm_client(
        self, ctx: dict, provider: Optional[str] = None
    ) -> Optional[Any]:
        """Get LLM client for generation.

        Args:
            ctx: Context dictionary
            provider: Optional LLM provider override

        Returns:
            LLM client or None
        """
        try:
            if provider:
                from ingestforge.llm.factory import get_llm_client

                return get_llm_client(ctx["config"], provider)

            from ingestforge.llm.factory import get_best_available_client

            return get_best_available_client(ctx["config"])
        except Exception as e:
            self.print_warning(f"LLM not available: {e}")
            return None

    def _generate_study_package(
        self,
        output_dir: Path,
        chunks: list,
        topic: Optional[str],
        llm_client: Optional[Any],
        include_all: bool,
        include_flashcards: bool = True,
        include_quiz: bool = True,
        quiet: bool = False,
    ) -> None:
        """Generate all components of study package with progress.

        Implements UX-003: Progress Indicators with ETA and quiet mode.

        Rule #4: Function <60 lines

        Args:
            output_dir: Output directory
            chunks: All chunks from knowledge base
            topic: Optional topic filter
            llm_client: Optional LLM client
            include_all: Whether to include all components
            include_flashcards: Whether to include flashcards
            include_quiz: Whether to include quiz
            quiet: Suppress progress output
        """
        if not quiet:
            self.print_info("Generating study package...")

        # Build list of steps to execute
        steps: List[tuple[str, Callable[[], None]]] = []

        # 1. Start here guide (always)
        steps.append(
            ("Start guide", lambda: self._generate_start_here(output_dir, topic))
        )

        # 2. Overview/synthesis
        if include_all:
            steps.append(
                (
                    "Overview",
                    lambda: self._generate_overview(output_dir, chunks, llm_client),
                )
            )

        # 3. Glossary
        if include_all:
            steps.append(
                (
                    "Glossary",
                    lambda: self._generate_glossary(output_dir, chunks, llm_client),
                )
            )

        # 4. Concept map
        if include_all:
            steps.append(
                (
                    "Concept map",
                    lambda: self._generate_concept_map(output_dir, chunks, llm_client),
                )
            )

        # 5. Study notes directory (always)
        steps.append(
            ("Study notes", lambda: self._generate_study_notes(output_dir, chunks))
        )

        # 6. Flashcards (Anki-compatible CSV) - controlled by flag
        if include_all and include_flashcards:
            steps.append(
                (
                    "Flashcards",
                    lambda: self._generate_flashcards_csv(
                        output_dir, chunks, llm_client
                    ),
                )
            )

        # 7. Quiz questions - controlled by flag
        if include_all and include_quiz:
            steps.append(
                ("Quiz", lambda: self._generate_quiz(output_dir, chunks, llm_client))
            )

        # 8. Reading list (always)
        steps.append(
            ("Reading list", lambda: self._generate_reading_list(output_dir, chunks))
        )

        # 9. Bibliography (BibTeX)
        if include_all:
            steps.append(
                (
                    "Bibliography",
                    lambda: self._generate_bibliography(output_dir, chunks),
                )
            )

        # Execute with progress reporter (UX-003)
        with ProgressReporter(
            total=len(steps),
            description="Generating package",
            quiet=quiet,
        ).start() as reporter:
            for idx, (step_name, step_func) in enumerate(steps, 1):
                step_func()
                reporter.update(current=idx, item_name=step_name)

    def _build_start_here_content(self, topic_str: str, timestamp: str) -> str:
        """Build start here guide template.

        Rule #4: Function <60 lines (refactored from 64)
        """
        header = self._build_header_section(topic_str, timestamp)
        materials = self._build_materials_section()
        study_path = self._build_study_path_section()
        tips = self._build_tips_section()
        footer = self._build_footer_section()

        return f"{header}\n{materials}\n{study_path}\n{tips}\n{footer}"

    def _build_header_section(self, topic_str: str, timestamp: str) -> str:
        """Build header section of start guide."""
        return f"""# {topic_str} - Study Package

Generated by IngestForge on {timestamp}

## Welcome! 👋

This folder contains everything you need to master {topic_str}. Here's what's included:"""

    def _build_materials_section(self) -> str:
        """Build materials and tools section."""
        return """
### 📚 Core Materials

1. **01_overview.md** - High-level synthesis of all content
2. **02_glossary.md** - Key terms and definitions
3. **03_concept_map.md** - How concepts relate to each other
4. **04_study_notes/** - Detailed notes organized by topic

### 🎯 Study Tools

5. **05_flashcards.csv** - Anki-compatible flashcards for spaced repetition
6. **06_quiz.md** - Practice questions to test your knowledge
7. **07_reading_list.md** - Source documents and recommended reading order

### 📖 Reference

8. **bibliography.bib** - Complete citations in BibTeX format"""

    def _build_study_path_section(self) -> str:
        """Build recommended study path section."""
        return """
## Recommended Study Path

1. **Start** with `01_overview.md` to get the big picture
2. **Read** source materials in order suggested in `07_reading_list.md`
3. **Review** `02_glossary.md` as you encounter new terms
4. **Create** connections using `03_concept_map.md`
5. **Study** detailed notes in `04_study_notes/`
6. **Test** yourself with `06_quiz.md`
7. **Reinforce** with `05_flashcards.csv` using spaced repetition"""

    def _build_tips_section(self) -> str:
        """Build tips and Anki import instructions."""
        return """
## Tips for Success

- **Spaced Repetition**: Review flashcards regularly (daily → weekly → monthly)
- **Active Recall**: Try to answer questions before checking answers
- **Concept Mapping**: Draw connections between ideas
- **Practice**: Complete quizzes multiple times
- **Deep Work**: Focus on one topic at a time

## Import Flashcards to Anki

1. Open Anki
2. File → Import
3. Select `05_flashcards.csv`
4. Choose "Front" and "Back" fields
5. Create new deck or select existing one"""

    def _build_footer_section(self) -> str:
        """Build footer section."""
        return """
## Generated by IngestForge

Learn more: https://github.com/IngestForge/ingestforge

---

*Happy studying! 📖*
"""

    def _generate_start_here(self, output_dir: Path, topic: Optional[str]) -> None:
        """Generate welcome guide.

        Rule #4: No large functions - Refactored to <60 lines

        Args:
            output_dir: Output directory
            topic: Optional topic name
        """
        self.console.print("  [cyan]→[/cyan] Generating start guide...")

        topic_str = topic if topic else "Your Knowledge Base"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = self._build_start_here_content(topic_str, timestamp)

        (output_dir / "00_START_HERE.md").write_text(content, encoding="utf-8")

    def _generate_overview(
        self, output_dir: Path, chunks: list, llm_client: Optional[Any]
    ) -> None:
        """Generate overview/synthesis.

        Args:
            output_dir: Output directory
            chunks: All chunks
            llm_client: Optional LLM client
        """
        self.console.print("  [cyan]→[/cyan] Generating overview...")

        if llm_client:
            # Generate with LLM
            overview_text = self._generate_llm_overview(chunks, llm_client)
        else:
            # Fallback: basic overview
            overview_text = self._generate_basic_overview(chunks)

        content = f"""# Overview

{overview_text}

---

*Generated by IngestForge*
"""

        (output_dir / "01_overview.md").write_text(content, encoding="utf-8")

    def _generate_llm_overview(self, chunks: list, llm_client: Any) -> str:
        """Generate overview using LLM.

        Args:
            chunks: All chunks
            llm_client: LLM client

        Returns:
            Overview text
        """
        # Build context from chunks
        context = self._build_context_sample(chunks, max_chunks=50)

        prompt = f"""Analyze this knowledge base and provide a comprehensive overview.

Context:
{context}

Generate a synthesis that covers:
1. Main topics and themes
2. Key concepts and how they relate
3. Important findings or conclusions
4. Scope and coverage

Write in clear, engaging prose. Use markdown formatting.
"""

        try:
            from ingestforge.cli.core import ProgressManager

            return ProgressManager.run_with_spinner(
                lambda: llm_client.generate(prompt),
                "Generating overview...",
                "Overview generated",
            )
        except Exception:
            return self._generate_basic_overview(chunks)

    def _generate_basic_overview(self, chunks: list) -> str:
        """Generate basic overview without LLM.

        Args:
            chunks: All chunks

        Returns:
            Overview text
        """
        # Group by source
        by_source = self.group_chunks_by_source(chunks)

        overview = [
            "## Knowledge Base Summary\n",
            f"**Total Documents**: {len(by_source)}",
            f"**Total Chunks**: {len(chunks)}\n",
            "## Included Sources\n",
        ]

        for idx, source in enumerate(sorted(by_source.keys()), 1):
            chunk_count = len(by_source[source])
            overview.append(f"{idx}. {source} ({chunk_count} sections)")

        return "\n".join(overview)

    def _build_context_sample(self, chunks: list, max_chunks: int = 50) -> str:
        """Build context sample from chunks.

        Args:
            chunks: All chunks
            max_chunks: Maximum chunks to include

        Returns:
            Context string
        """
        sample = chunks[:max_chunks]
        context_parts = []

        for chunk in sample:
            text = self.extract_chunk_text(chunk)
            context_parts.append(text[:500])  # Limit per chunk

        return "\n\n---\n\n".join(context_parts)

    def _generate_glossary(
        self, output_dir: Path, chunks: list, llm_client: Optional[Any]
    ) -> None:
        """Generate glossary of key terms.

        Args:
            output_dir: Output directory
            chunks: All chunks
            llm_client: Optional LLM client
        """
        self.console.print("  [cyan]→[/cyan] Generating glossary...")

        if llm_client:
            glossary_text = self._generate_llm_glossary(chunks, llm_client)
        else:
            glossary_text = "Glossary generation requires an LLM client.\n\nTerms will be listed here."

        content = f"""# Glossary

{glossary_text}

---

*Generated by IngestForge*
"""

        (output_dir / "02_glossary.md").write_text(content, encoding="utf-8")

    def _generate_llm_glossary(self, chunks: list, llm_client: Any) -> str:
        """Generate glossary using LLM.

        Args:
            chunks: All chunks
            llm_client: LLM client

        Returns:
            Glossary text
        """
        context = self._build_context_sample(chunks, max_chunks=30)

        prompt = f"""Extract and define key terms from this knowledge base.

Context:
{context}

Generate a glossary with:
- 20-30 most important terms
- Clear, concise definitions
- Alphabetical order
- Markdown formatting

Format as:
**Term**: Definition
"""

        try:
            from ingestforge.cli.core import ProgressManager

            return ProgressManager.run_with_spinner(
                lambda: llm_client.generate(prompt),
                "Generating glossary...",
                "Glossary generated",
            )
        except Exception:
            return "Error generating glossary."

    def _generate_concept_map(
        self, output_dir: Path, chunks: list, llm_client: Optional[Any]
    ) -> None:
        """Generate concept map/relationship diagram.

        Args:
            output_dir: Output directory
            chunks: All chunks
            llm_client: Optional LLM client
        """
        self.console.print("  [cyan]→[/cyan] Generating concept map...")

        if llm_client:
            concept_map_text = self._generate_llm_concept_map(chunks, llm_client)
        else:
            concept_map_text = "Concept map generation requires an LLM client."

        content = f"""# Concept Map

{concept_map_text}

---

*Generated by IngestForge*
"""

        (output_dir / "03_concept_map.md").write_text(content, encoding="utf-8")

    def _generate_llm_concept_map(self, chunks: list, llm_client: Any) -> str:
        """Generate concept map using LLM.

        Args:
            chunks: All chunks
            llm_client: LLM client

        Returns:
            Concept map text
        """
        context = self._build_context_sample(chunks, max_chunks=30)

        prompt = f"""Map the relationships between key concepts in this knowledge base.

Context:
{context}

Generate a concept map showing:
- Main concepts (5-10)
- How they relate to each other
- Dependencies and prerequisites
- Visual representation using markdown/mermaid

Use markdown formatting and consider creating a mermaid diagram.
"""

        try:
            from ingestforge.cli.core import ProgressManager

            return ProgressManager.run_with_spinner(
                lambda: llm_client.generate(prompt),
                "Generating concept map...",
                "Concept map generated",
            )
        except Exception:
            return "Error generating concept map."

    def _generate_study_notes(self, output_dir: Path, chunks: list) -> None:
        """Generate study notes directory.

        Args:
            output_dir: Output directory
            chunks: All chunks
        """
        self.console.print("  [cyan]→[/cyan] Generating study notes...")

        notes_dir = output_dir / "04_study_notes"
        notes_dir.mkdir(exist_ok=True)

        # Group by source
        by_source = self.group_chunks_by_source(chunks)

        for source, source_chunks in by_source.items():
            # Create note file for each source
            filename = self._sanitize_filename(source) + ".md"
            note_path = notes_dir / filename

            note_content = self._build_note_content(source, source_chunks)
            note_path.write_text(note_content, encoding="utf-8")

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename.

        Args:
            name: Original name

        Returns:
            Safe filename
        """
        # Remove extension
        name = Path(name).stem

        # Replace invalid characters
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)

        # Limit length
        return safe[:100]

    def _build_note_content(self, source: str, chunks: list) -> str:
        """Build note content for source.

        Args:
            source: Source name
            chunks: Chunks from source

        Returns:
            Note content
        """
        parts = [f"# {source}\n", f"**Sections**: {len(chunks)}\n", "---\n"]

        for idx, chunk in enumerate(chunks, 1):
            text = self.extract_chunk_text(chunk)
            metadata = self.extract_chunk_metadata(chunk)

            parts.append(f"## Section {idx}\n")

            # Add metadata if available
            if "page" in metadata:
                parts.append(f"*Page {metadata['page']}*\n")

            parts.append(f"{text}\n\n")

        return "\n".join(parts)

    def _generate_flashcards_csv(
        self, output_dir: Path, chunks: list, llm_client: Optional[Any]
    ) -> None:
        """Generate Anki-compatible flashcards CSV.

        Args:
            output_dir: Output directory
            chunks: All chunks
            llm_client: Optional LLM client
        """
        self.console.print("  [cyan]→[/cyan] Generating flashcards...")

        csv_path = output_dir / "05_flashcards.csv"

        if llm_client:
            # Generate flashcards with LLM
            self._generate_llm_flashcards(csv_path, chunks, llm_client)
        else:
            # Create template
            csv_path.write_text("Front,Back\n", encoding="utf-8")

    def _generate_llm_flashcards(
        self, csv_path: Path, chunks: list, llm_client: Any
    ) -> None:
        """Generate flashcards using LLM.

        Args:
            csv_path: CSV file path
            chunks: All chunks
            llm_client: LLM client
        """
        context = self._build_context_sample(chunks, max_chunks=20)

        prompt = f"""Generate 20-30 flashcard questions from this content.

Context:
{context}

Return as CSV format:
Front,Back
"Question 1","Answer 1"
"Question 2","Answer 2"

Focus on:
- Key concepts and definitions
- Important facts
- Critical relationships
- Common misconceptions

Return ONLY the CSV data, no additional text.
"""

        try:
            from ingestforge.cli.core import ProgressManager

            csv_content = ProgressManager.run_with_spinner(
                lambda: llm_client.generate(prompt),
                "Generating flashcards...",
                "Flashcards generated",
            )

            # Ensure CSV header
            if not csv_content.startswith("Front,Back"):
                csv_content = "Front,Back\n" + csv_content

            csv_path.write_text(csv_content, encoding="utf-8")

        except Exception:
            csv_path.write_text("Front,Back\n", encoding="utf-8")

    def _generate_quiz(
        self, output_dir: Path, chunks: list, llm_client: Optional[Any]
    ) -> None:
        """Generate quiz questions.

        Args:
            output_dir: Output directory
            chunks: All chunks
            llm_client: Optional LLM client
        """
        self.console.print("  [cyan]→[/cyan] Generating quiz...")

        if llm_client:
            quiz_text = self._generate_llm_quiz(chunks, llm_client)
        else:
            quiz_text = "Quiz generation requires an LLM client."

        content = f"""# Practice Quiz

{quiz_text}

---

*Generated by IngestForge*
"""

        (output_dir / "06_quiz.md").write_text(content, encoding="utf-8")

    def _generate_llm_quiz(self, chunks: list, llm_client: Any) -> str:
        """Generate quiz using LLM.

        Args:
            chunks: All chunks
            llm_client: LLM client

        Returns:
            Quiz text
        """
        context = self._build_context_sample(chunks, max_chunks=20)

        prompt = f"""Generate a 10-question quiz from this content.

Context:
{context}

Include:
- 5 multiple choice questions
- 3 short answer questions
- 2 essay questions

Format in markdown with clear sections.
"""

        try:
            from ingestforge.cli.core import ProgressManager

            return ProgressManager.run_with_spinner(
                lambda: llm_client.generate(prompt),
                "Generating quiz...",
                "Quiz generated",
            )
        except Exception:
            return "Error generating quiz."

    def _generate_reading_list(self, output_dir: Path, chunks: list) -> None:
        """Generate reading list.

        Args:
            output_dir: Output directory
            chunks: All chunks
        """
        self.console.print("  [cyan]→[/cyan] Generating reading list...")

        by_source = self.group_chunks_by_source(chunks)

        content = ["# Reading List\n"]
        content.append(f"**Total Sources**: {len(by_source)}\n")
        content.append("## Recommended Reading Order\n")

        for idx, source in enumerate(sorted(by_source.keys()), 1):
            chunk_count = len(by_source[source])
            content.append(f"{idx}. **{source}** ({chunk_count} sections)")

        content.append("\n---\n\n*Generated by IngestForge*")

        (output_dir / "07_reading_list.md").write_text(
            "\n".join(content), encoding="utf-8"
        )

    def _generate_bibliography(self, output_dir: Path, chunks: list) -> None:
        """Generate BibTeX bibliography.

        Args:
            output_dir: Output directory
            chunks: All chunks
        """
        self.console.print("  [cyan]→[/cyan] Generating bibliography...")

        by_source = self.group_chunks_by_source(chunks)

        bib_entries = []

        for source in sorted(by_source.keys()):
            # Generate BibTeX entry
            cite_key = self._generate_cite_key(source)
            entry = self._generate_bibtex_entry(cite_key, source)
            bib_entries.append(entry)

        bib_content = "\n\n".join(bib_entries)

        (output_dir / "bibliography.bib").write_text(bib_content, encoding="utf-8")

    def _generate_cite_key(self, source: str) -> str:
        """Generate BibTeX citation key.

        Args:
            source: Source name

        Returns:
            Citation key
        """
        # Simple key from filename
        key = Path(source).stem
        key = "".join(c if c.isalnum() else "" for c in key)
        return key[:30].lower()

    def _generate_bibtex_entry(self, cite_key: str, source: str) -> str:
        """Generate BibTeX entry.

        Args:
            cite_key: Citation key
            source: Source name

        Returns:
            BibTeX entry
        """
        year = datetime.now().year

        return f"""@misc{{{cite_key},
  title = {{{source}}},
  year = {{{year}}},
  note = {{Processed by IngestForge}}
}}"""

    def _display_package_summary(
        self,
        output_dir: Path,
        include_flashcards: bool = True,
        include_quiz: bool = True,
    ) -> None:
        """Display package generation summary.

        Args:
            output_dir: Output directory
            include_flashcards: Whether flashcards were included
            include_quiz: Whether quiz was included
        """
        self.console.print()
        self.print_success(f"Study package created: {output_dir}")
        self.console.print()

        self.console.print("[bold cyan]Package Contents:[/bold cyan]")
        self.console.print("  [green]OK[/green] 00_START_HERE.md - Welcome guide")
        self.console.print("  [green]OK[/green] 01_overview.md - Content synthesis")
        self.console.print("  [green]OK[/green] 02_glossary.md - Key terms")
        self.console.print("  [green]OK[/green] 03_concept_map.md - Relationships")
        self.console.print("  [green]OK[/green] 04_study_notes/ - Detailed notes")

        if include_flashcards:
            self.console.print(
                "  [green]OK[/green] 05_flashcards.csv - Anki flashcards"
            )
        else:
            self.console.print("  [dim]--[/dim] 05_flashcards.csv - Skipped")

        if include_quiz:
            self.console.print("  [green]OK[/green] 06_quiz.md - Practice questions")
        else:
            self.console.print("  [dim]--[/dim] 06_quiz.md - Skipped")

        self.console.print("  [green]OK[/green] 07_reading_list.md - Source list")
        self.console.print("  [green]OK[/green] bibliography.bib - Citations")

        self.console.print()
        self.print_info(f"Start with: {output_dir / '00_START_HERE.md'}")


# Typer command wrapper
def command(
    output_dir: Path = typer.Argument(..., help="Output directory for study package"),
    topic: Optional[str] = typer.Option(
        None, "--topic", "-t", help="Topic name for package"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    llm: Optional[str] = typer.Option(
        None,
        "--llm",
        "-l",
        help="LLM provider (claude, openai, gemini, ollama, llamacpp)",
    ),
    include_flashcards: bool = typer.Option(
        True,
        "--include-flashcards/--no-flashcards",
        help="Include Anki-compatible flashcards",
    ),
    include_quiz: bool = typer.Option(
        True, "--include-quiz/--no-quiz", help="Include practice quiz questions"
    ),
    include_all: bool = typer.Option(
        True, "--all/--basic", help="Include all components or basic only"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress progress bar output"
    ),
) -> None:
    """Export complete study package to a folder.

    Creates a comprehensive learning package with:
    - Overview and synthesis
    - Glossary of key terms
    - Concept relationship map
    - Detailed study notes
    - Anki-compatible flashcards
    - Practice quiz
    - Reading list
    - BibTeX bibliography

    This is IngestForge's signature feature - a complete study
    system generated from your knowledge base.

    Examples:
        # Generate full package
        ingestforge export folder-export ./study-package

        # With specific LLM provider
        ingestforge export folder-export ./output --llm claude

        # With topic name
        ingestforge export folder-export ./biology-study --topic "Biology 101"

        # Without flashcards
        ingestforge export folder-export ./study --no-flashcards

        # Basic package only
        ingestforge export folder-export ./study --basic

        # Specific project
        ingestforge export folder-export ./package -p /path/to/project

        # Suppress progress bar (for CI/scripts)
        ingestforge export folder-export ./study --quiet
    """
    cmd = FolderExportCommand()
    exit_code = cmd.execute(
        output_dir,
        topic,
        project,
        include_all,
        llm_provider=llm,
        include_flashcards=include_flashcards,
        include_quiz=include_quiz,
        quiet=quiet,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
