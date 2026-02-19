"""Init command - Initialize new IngestForge project.

Creates project directory structure and configuration file.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class InitCommand(IngestForgeCommand):
    """Initialize a new IngestForge project."""

    def execute(
        self,
        name: str,
        path: Optional[Path] = None,
        with_sample: bool = False,
        mobile: bool = False,
    ) -> int:
        """Create new project with configuration.

        Args:
            name: Project name
            path: Project directory (default: ./<name>)
            with_sample: Include and process sample document
            mobile: Enable mobile mode (JSONL storage, no vector DB)

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate project name (Commandment #7: Check inputs)
            self.validate_project_name(name)

            # Determine project path
            project_path = self._resolve_project_path(name, path)

            # Check if path already exists
            self._check_path_available(project_path)

            # Create project (Commandment #1: Simple flow)
            self._create_project_structure(project_path, name, mobile)

            # Optionally process sample
            if with_sample:
                self._process_sample_document(project_path)

            # Show success and next steps
            self._print_success_message(name, project_path)

            return 0

        except Exception as e:
            return self.handle_error(e, "Project initialization failed")

    def _resolve_project_path(self, name: str, path: Optional[Path]) -> Path:
        """Resolve project path from name and optional path.

        Args:
            name: Project name
            path: Optional path (if None, uses ./<name>)

        Returns:
            Resolved absolute path
        """
        if path is not None:
            return path.resolve()
        return (Path.cwd() / name).resolve()

    def _check_path_available(self, project_path: Path) -> None:
        """Check if project path is available for use.

        Args:
            project_path: Path to check

        Raises:
            ValueError: If path already exists and contains project
        """
        if not project_path.exists():
            return  # Path available

        # Path exists - check if it's already a project
        config_file = project_path / "ingestforge.yaml"
        if config_file.exists():
            raise ValueError(
                f"Directory already contains IngestForge project: {project_path}\n"
                "Use a different directory or remove existing project."
            )

        # Path exists but not a project - check if empty
        if any(project_path.iterdir()):
            raise ValueError(
                f"Directory not empty: {project_path}\n"
                "Please use an empty directory or specify a new path."
            )

    def _create_project_structure(
        self, project_path: Path, name: str, mobile: bool
    ) -> None:
        """Create project directories and configuration.

        Args:
            project_path: Project directory
            name: Project name
            mobile: Enable mobile mode
        """
        self.print_info(f"Creating project: {name}")

        # Create directory structure
        self._create_directories(project_path)
        self.print_success("Created directory structure")

        # Generate and write configuration
        config = self._generate_config(name, mobile)
        self._write_config(project_path, config)
        self.print_success("Generated configuration")

    def _create_directories(self, project_path: Path) -> None:
        """Create standard project directory structure.

        Args:
            project_path: Project root directory
        """
        directories = [
            project_path / "documents",
            project_path / "data",
            project_path / "exports",
        ]

        # Create all directories (Commandment #2: Bounded loop)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _generate_config(self, name: str, mobile: bool) -> str:
        """Generate project configuration YAML.

        Args:
            name: Project name
            mobile: Enable mobile mode

        Returns:
            YAML configuration string
        """
        import yaml

        # Build config as a dictionary
        config_dict = {
            "project": {
                "name": name,
                "version": "1.0.0",
                "mobile_mode": mobile,
            },
            "storage": {
                "backend": "chromadb",
                "chromadb": {
                    "persist_directory": ".data/chromadb",
                    "collection_name": f"{name}_chunks",
                },
            },
            "llm": {
                "default_provider": "llamacpp",
            },
            "chunking": {
                "strategy": "semantic",
                "chunk_size": 512,
                "chunk_overlap": 50,
            },
            "embedding": {
                "provider": "sentence-transformers",
                "model": "all-MiniLM-L6-v2",
            },
        }

        return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

    def _write_config(self, project_path: Path, config_yaml: str) -> None:
        """Write configuration to file.

        Args:
            project_path: Project directory
            config_yaml: YAML configuration string
        """
        config_path = project_path / "ingestforge.yaml"
        config_path.write_text(config_yaml, encoding="utf-8")

    def _process_sample_document(self, project_path: Path) -> None:
        """Process sample document for quickstart.

        Args:
            project_path: Project directory
        """
        from ingestforge.core.config_loaders import load_config
        from ingestforge.core.pipeline import Pipeline

        self.print_info("Processing sample document...")

        try:
            # Load config and create pipeline
            config = load_config(base_path=project_path)
            pipeline = Pipeline(config, project_path)

            # Get sample document path
            sample_path = self._get_sample_document_path()

            # Process with progress indicator
            ProgressManager.run_with_spinner(
                lambda: pipeline.process_file(sample_path),
                "Processing sample document...",
                "Sample document processed!",
            )

        except Exception as e:
            self.print_warning(f"Failed to process sample document: {e}")

    def _get_sample_document_path(self) -> Path:
        """Get path to sample document.

        Returns:
            Path to getting_started.md sample
        """
        # Sample document is in package samples directory
        samples_dir = Path(__file__).parent.parent / "samples"
        return samples_dir / "getting_started.md"

    def _print_success_message(self, name: str, project_path: Path) -> None:
        """Print success message with next steps.

        Args:
            name: Project name
            project_path: Project directory
        """
        self.console.print(
            "\n[bold green][OK] Project created successfully![/bold green]\n"
        )

        # Show next steps (Commandment #4: Keep function small)
        self.console.print("[bold]Next steps:[/bold]")
        self.console.print(f"  1. cd {project_path}")
        self.console.print("  2. Place documents in documents/")
        self.console.print("  3. ingestforge ingest documents/")
        self.console.print('  4. ingestforge query "your question"')
        self.console.print()
        self.console.print("[dim]For help: ingestforge --help[/dim]")


# Typer command wrapper
def command(
    name: str = typer.Argument(..., help="Project name"),
    path: Optional[Path] = typer.Option(
        None, "--path", help="Project directory (default: ./<name>)"
    ),
    with_sample: bool = typer.Option(
        False, "--with-sample", help="Include sample document"
    ),
    mobile: bool = typer.Option(
        False, "--mobile", help="Enable mobile mode (JSONL storage)"
    ),
) -> None:
    """Initialize a new IngestForge project.

    Creates directory structure, generates configuration, and optionally
    processes a sample document.

    Examples:
        # Create project with default settings
        ingestforge init my_project

        # Create project with sample document
        ingestforge init my_project --with-sample

        # Create project for mobile (no vector DB)
        ingestforge init my_project --mobile

        # Create project at specific location
        ingestforge init my_project --path /path/to/location
    """
    cmd = InitCommand()
    exit_code = cmd.execute(name, path, with_sample, mobile)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
