"""Analyze command - Analyze code structure and patterns.

Analyzes code structure, patterns, and provides architectural insights.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from ingestforge.cli.code.base import CodeAnalysisCommand


class AnalyzeCommand(CodeAnalysisCommand):
    """Analyze code structure and patterns."""

    def execute(
        self,
        target: Path,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        pattern: Optional[str] = None,
    ) -> int:
        """Analyze code structure and patterns.

        Args:
            target: Code file or directory to analyze
            project: Project directory
            output: Output file for analysis
            pattern: Specific pattern to analyze

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs (Commandment #7: Check parameters)
            self.validate_code_path(target)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Get LLM client
            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            # Analyze code
            analysis_data = self._analyze_code(
                target, ctx["storage"], llm_client, pattern
            )

            # Display results
            self._display_analysis(analysis_data, target)

            # Save to file if requested
            if output:
                self._save_analysis(output, analysis_data)

            return 0

        except Exception as e:
            return self.handle_error(e, "Code analysis failed")

    def _analyze_code(
        self,
        target: Path,
        storage: Any,
        llm_client: Optional[Any],
        pattern: Optional[str],
    ) -> Dict[str, Any]:
        """Analyze code structure.

        Args:
            target: Target path
            storage: Storage instance
            llm_client: LLM client
            pattern: Optional specific pattern

        Returns:
            Analysis data dictionary
        """
        # Get file statistics
        stats = self._get_file_statistics(target)

        # Search for relevant context
        context_chunks = self._search_for_context(storage, target.name, pattern)

        # Generate analysis if context available
        if context_chunks and llm_client:
            analysis = self._generate_analysis(
                llm_client, target, context_chunks, pattern
            )
        else:
            analysis = "No context available for detailed analysis"

        return {
            "target": str(target),
            "statistics": stats,
            "analysis": analysis,
            "pattern": pattern,
        }

    def _get_file_statistics(self, target: Path) -> Dict[str, Any]:
        """Get statistics for target.

        Args:
            target: Target path

        Returns:
            Statistics dictionary
        """
        if target.is_file():
            return self._analyze_single_file(target)
        else:
            return self._analyze_directory(target)

    def _analyze_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze single file.

        Args:
            file_path: File to analyze

        Returns:
            File statistics
        """
        file_info = self.extract_file_info(file_path)

        return {
            "type": "file",
            "files": 1,
            "total_lines": file_info["lines"],
            "total_size": file_info["size"],
            "languages": {file_info["extension"]: 1},
        }

    def _analyze_directory(self, directory: Path) -> Dict[str, Any]:
        """Analyze directory.

        Args:
            directory: Directory to analyze

        Returns:
            Directory statistics
        """
        code_files = self.find_code_files(directory)

        if not code_files:
            return {
                "type": "directory",
                "files": 0,
                "total_lines": 0,
                "total_size": 0,
                "languages": {},
            }

        total_lines = 0
        total_size = 0
        languages: Dict[str, int] = {}

        for file_path in code_files:
            file_info = self.extract_file_info(file_path)
            total_lines += file_info["lines"]
            total_size += file_info["size"]

            ext = file_info["extension"]
            languages[ext] = languages.get(ext, 0) + 1

        return {
            "type": "directory",
            "files": len(code_files),
            "total_lines": total_lines,
            "total_size": total_size,
            "languages": languages,
        }

    def _search_for_context(
        self, storage: Any, target_name: str, pattern: Optional[str]
    ) -> list[Any]:
        """Search for relevant code context.

        Args:
            storage: Storage instance
            target_name: Name of target
            pattern: Optional pattern

        Returns:
            List of chunks
        """
        if pattern:
            query = f"{target_name} {pattern}"
        else:
            query = f"code {target_name}"

        try:
            return self.search_code_context(storage, query, k=15)
        except Exception:
            return []

    def _generate_analysis(
        self,
        llm_client: Any,
        target: Path,
        chunks: list,
        pattern: Optional[str],
    ) -> str:
        """Generate code analysis using LLM.

        Args:
            llm_client: LLM client
            target: Target path
            chunks: Context chunks
            pattern: Optional pattern

        Returns:
            Generated analysis
        """
        # Build context and prompt
        context = self.format_context_for_prompt(chunks)
        prompt = self._build_analysis_prompt(target, context, pattern)

        # Generate analysis
        return self.generate_with_llm(llm_client, prompt, "code analysis")

    def _build_analysis_prompt(
        self, target: Path, context: str, pattern: Optional[str]
    ) -> str:
        """Build prompt for code analysis.

        Args:
            target: Target path
            context: Context from knowledge base
            pattern: Optional pattern

        Returns:
            Formatted prompt
        """
        prompt_parts = [
            f"Analyze the code structure and patterns for: {target.name}\n",
            "\nContext from knowledge base:\n",
            context,
            "\n\n",
        ]

        if pattern:
            prompt_parts.append(f"Focus specifically on: {pattern}\n\n")

        prompt_parts.extend(
            [
                "Provide analysis covering:\n",
                "1. Overall structure and organization\n",
                "2. Key architectural patterns identified\n",
                "3. Code quality observations\n",
                "4. Potential improvements or concerns\n",
                "5. Design patterns in use\n\n",
                "Format as clear, structured markdown.",
            ]
        )

        return "".join(prompt_parts)

    def _display_analysis(self, analysis_data: Dict[str, Any], target: Path) -> None:
        """Display analysis results.

        Args:
            analysis_data: Analysis data
            target: Target path
        """
        self.console.print()

        # Display statistics
        self._display_statistics(analysis_data["statistics"])

        # Display analysis
        self.console.print()

        panel = Panel(
            Markdown(analysis_data["analysis"]),
            title=f"[bold cyan]Code Analysis: {target.name}[/bold cyan]",
            border_style="cyan",
        )

        self.console.print(panel)

    def _display_statistics(self, stats: Dict[str, Any]) -> None:
        """Display statistics table.

        Args:
            stats: Statistics dictionary
        """
        table = Table(title="Code Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Type", stats["type"].capitalize())
        table.add_row("Files", str(stats["files"]))
        table.add_row("Total Lines", f"{stats['total_lines']:,}")
        table.add_row("Total Size", f"{stats['total_size'] / 1024:.1f} KB")

        if stats["languages"]:
            lang_str = ", ".join(
                f"{ext}({count})" for ext, count in sorted(stats["languages"].items())
            )
            table.add_row("Languages", lang_str)

        self.console.print(table)

    def _save_analysis(self, output: Path, analysis_data: Dict[str, Any]) -> None:
        """Save analysis to file.

        Args:
            output: Output file path
            analysis_data: Analysis data
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            content = [
                f"# Code Analysis: {Path(analysis_data['target']).name}\n\n",
                f"**Generated:** {timestamp}\n",
                f"**Target:** {analysis_data['target']}\n",
            ]

            if analysis_data["pattern"]:
                content.append(f"**Pattern:** {analysis_data['pattern']}\n")

            content.append("\n## Statistics\n\n")

            stats = analysis_data["statistics"]
            content.append(f"- Type: {stats['type']}\n")
            content.append(f"- Files: {stats['files']}\n")
            content.append(f"- Lines: {stats['total_lines']:,}\n")
            content.append(f"- Size: {stats['total_size'] / 1024:.1f} KB\n")

            content.append("\n## Analysis\n\n")
            content.append(analysis_data["analysis"])
            content.append("\n")

            output.write_text("".join(content), encoding="utf-8")
            self.print_success(f"Analysis saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save analysis: {e}")


# Typer command wrapper
def command(
    target: Path = typer.Argument(..., help="Code file or directory to analyze"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for analysis"
    ),
    pattern: Optional[str] = typer.Option(
        None, "--pattern", help="Specific pattern to analyze"
    ),
) -> None:
    """Analyze code structure and patterns.

    Analyzes code to identify structure, patterns, and provides
    insights about architecture and quality.

    Requires code or documentation to be ingested first.

    Examples:
        # Analyze a file
        ingestforge code analyze src/main.py

        # Analyze a directory
        ingestforge code analyze src/

        # Focus on specific pattern
        ingestforge code analyze src/ --pattern "error handling"

        # Save analysis
        ingestforge code analyze app.py -o analysis.md

        # Specific project
        ingestforge code analyze code/ -p /path/to/project
    """
    cmd = AnalyzeCommand()
    exit_code = cmd.execute(target, project, output, pattern)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
