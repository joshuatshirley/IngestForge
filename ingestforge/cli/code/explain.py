"""Explain command - Explain code from files."""

from pathlib import Path
from typing import Any, Optional
import typer
from rich.markdown import Markdown
from rich.panel import Panel
from ingestforge.cli.code.base import CodeAnalysisCommand


class ExplainCommand(CodeAnalysisCommand):
    """Explain code from files."""

    def execute(
        self,
        file_path: Path,
        mode: str = "detailed",
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """Explain code in a file."""
        try:
            # Validate file exists
            if not file_path.exists():
                self.print_error(f"File not found: {file_path}")
                return 1

            # Read code
            code_content = file_path.read_text(encoding="utf-8")

            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            llm_client = self.get_llm_client(ctx)

            if not llm_client:
                return 1

            # Generate explanation
            explanation = self._generate_explanation(
                llm_client, code_content, str(file_path), mode
            )

            # Display explanation
            self._display_explanation(explanation, str(file_path), mode)

            # Save if requested
            if output:
                self.save_json_output(
                    output, explanation, f"Explanation saved to: {output}"
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Code explanation failed")

    def _generate_explanation(
        self, llm_client: any, code: str, file_path: str, mode: str
    ) -> dict[str, Any]:
        """Generate code explanation."""
        mode_instructions = self._get_mode_instructions(mode)

        prompt = f"""Explain this code from {file_path}:

```
{code[:3000]}
```

{mode_instructions}

Return JSON:
{{
  "summary": "One-sentence summary",
  "purpose": "What this code does",
  "key_components": ["component1", "component2", ...],
  "complexity": "simple|moderate|complex",
  "dependencies": ["dep1", "dep2", ...],
  "explanation": "Detailed explanation"
}}"""

        response = self.generate_with_llm(llm_client, prompt, "code explanation")
        return self.parse_json(response) or {"explanation": response}

    def _get_mode_instructions(self, mode: str) -> str:
        """Get mode-specific instructions."""
        modes = {
            "simple": "Explain in simple terms, as if to a beginner.",
            "detailed": "Provide detailed technical explanation.",
            "architecture": "Focus on architectural patterns and design decisions.",
        }
        return modes.get(mode, modes["detailed"])

    def _display_explanation(
        self, explanation: dict, file_path: str, mode: str
    ) -> None:
        """Display code explanation."""
        self.console.print()
        self.console.print(f"[bold cyan]Code Explanation: {file_path}[/bold cyan]\n")

        # Summary
        if "summary" in explanation:
            summary = Panel(
                explanation["summary"],
                title="Summary",
                border_style="cyan",
            )
            self.console.print(summary)

        # Purpose
        if "purpose" in explanation:
            self.console.print()
            self.console.print("[bold yellow]Purpose:[/bold yellow]")
            self.console.print(f"  {explanation['purpose']}")

        # Key components
        if "key_components" in explanation:
            self.console.print()
            self.console.print("[bold yellow]Key Components:[/bold yellow]")
            for comp in explanation["key_components"][:10]:
                self.console.print(f"  • {comp}")

        # Detailed explanation
        if "explanation" in explanation:
            self.console.print()
            self.console.print("[bold yellow]Explanation:[/bold yellow]")
            md = Markdown(explanation["explanation"])
            self.console.print(md)


def command(
    file_path: Path = typer.Argument(..., help="File to explain"),
    mode: str = typer.Option(
        "detailed",
        "--mode",
        "-m",
        help="Explanation mode: simple, detailed, architecture",
    ),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
) -> None:
    """Explain code in a file.

    Examples:
        ingestforge code explain src/main.py
        ingestforge code explain app.py --mode simple
        ingestforge code explain module.py --mode architecture -o explanation.json
    """
    cmd = ExplainCommand()
    exit_code = cmd.execute(file_path, mode, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
