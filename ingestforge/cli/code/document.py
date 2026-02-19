"""Document command - Auto-generate code documentation."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.table import Table
from ingestforge.cli.code.base import CodeAnalysisCommand


class DocumentCommand(CodeAnalysisCommand):
    """Auto-generate code documentation."""

    def execute(
        self,
        path: Path,
        format_type: str = "markdown",
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """Generate documentation for code."""
        try:
            # Validate path exists
            if not path.exists():
                self.print_error(f"Path not found: {path}")
                return 1

            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            llm_client = self.get_llm_client(ctx)

            if not llm_client:
                return 1

            # Find code files
            code_files = self._find_code_files(path)

            if not code_files:
                self.print_warning(f"No code files found in: {path}")
                return 0

            # Generate documentation
            docs = self._generate_documentation(llm_client, code_files, format_type)

            # Display summary
            self._display_documentation_summary(docs, path)

            # Save if requested
            if output:
                self._save_documentation(output, docs, format_type)

            return 0

        except Exception as e:
            return self.handle_error(e, "Documentation generation failed")

    def _find_code_files(self, path: Path) -> List[Path]:
        """Find code files in path."""
        code_extensions = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"}

        if path.is_file():
            return [path] if path.suffix in code_extensions else []

        # Search directory
        files = []
        for ext in code_extensions:
            files.extend(path.rglob(f"*{ext}"))

        return sorted(files)[:20]  # Limit to 20 files

    def _generate_documentation(
        self, llm_client: any, files: List[Path], format_type: str
    ) -> List[Dict[str, Any]]:
        """Generate documentation for files."""
        docs = []

        for file_path in files:
            code = file_path.read_text(encoding="utf-8")
            doc = self._generate_file_doc(llm_client, code, str(file_path), format_type)
            docs.append({"file": str(file_path), **doc})

        return docs

    def _generate_file_doc(
        self, llm_client: any, code: str, file_path: str, format_type: str
    ) -> Dict[str, Any]:
        """Generate documentation for single file."""
        prompt = f"""Generate documentation for: {file_path}

Code (first 2000 chars):
```
{code[:2000]}
```

Return JSON:
{{
  "summary": "One-line summary",
  "functions": [
    {{"name": "func1", "description": "...", "params": ["p1", "p2"], "returns": "..."}},
    ...
  ],
  "classes": [
    {{"name": "Class1", "description": "...", "methods": ["m1", "m2"]}},
    ...
  ],
  "usage_example": "Example code snippet"
}}"""

        response = self.generate_with_llm(llm_client, prompt, "documentation")
        return self.parse_json(response) or {"summary": "Documentation unavailable"}

    def _display_documentation_summary(self, docs: List[Dict], path: Path) -> None:
        """Display documentation summary."""
        self.console.print()
        self.console.print(f"[bold cyan]Documentation Generated: {path}[/bold cyan]\n")

        table = Table(title="Files Documented")
        table.add_column("File", width=40)
        table.add_column("Summary", width=50)

        for doc in docs:
            file_name = Path(doc["file"]).name
            summary = doc.get("summary", "")[:80]
            table.add_row(file_name, summary)

        self.console.print(table)
        self.print_info(f"Total files: {len(docs)}")

    def _save_documentation(
        self, output: Path, docs: List[Dict], format_type: str
    ) -> None:
        """Save documentation to file."""
        output.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "markdown":
            self._save_as_markdown(output, docs)
        else:
            self.save_json_output(output, docs, f"Documentation saved to: {output}")

    def _save_as_markdown(self, output: Path, docs: List[Dict]) -> None:
        """Save documentation as Markdown."""
        lines = ["# Code Documentation\n\n"]

        for doc in docs:
            file_name = Path(doc["file"]).name
            lines.append(f"## {file_name}\n\n")

            if "summary" in doc:
                lines.append(f"{doc['summary']}\n\n")

            if "functions" in doc:
                lines.append("### Functions\n\n")
                for func in doc["functions"]:
                    name = func.get("name", "")
                    desc = func.get("description", "")
                    lines.append(f"- `{name}`: {desc}\n")
                lines.append("\n")

            if "classes" in doc:
                lines.append("### Classes\n\n")
                for cls in doc["classes"]:
                    name = cls.get("name", "")
                    desc = cls.get("description", "")
                    lines.append(f"- `{name}`: {desc}\n")
                lines.append("\n")

        output.write_text("".join(lines), encoding="utf-8")
        self.print_success(f"Documentation saved to: {output}")


def command(
    path: Path = typer.Argument(..., help="File or directory to document"),
    format_type: str = typer.Option(
        "markdown", "--format", "-f", help="Output format: markdown, json"
    ),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
) -> None:
    """Auto-generate code documentation.

    Examples:
        ingestforge code document src/
        ingestforge code document app.py --format markdown -o docs.md
        ingestforge code document project/ --format json -o api.json
    """
    cmd = DocumentCommand()
    exit_code = cmd.execute(path, format_type, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
