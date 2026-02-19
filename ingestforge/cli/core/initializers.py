"""Shared initialization patterns for CLI commands.

This module provides reusable initialization logic that appears in 5+ CLI commands,
following Commandments #6 (Smallest Scope) and #7 (Check Parameters & Returns).

Eliminates ~80 lines of duplicate code across CLI commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from ingestforge.cli.core.error_handlers import CLIErrorHandler


class CLIInitializer:
    """Reusable initialization patterns for CLI commands.

    Provides standard patterns for:
    - Project path resolution
    - Project initialization checking
    - Configuration loading
    - Storage backend initialization
    - Pipeline initialization

    All methods are static for ease of use.

    Example:
        # One-line initialization for most commands
        ctx = CLIInitializer.initialize_for_command(require_storage=True)
        config = ctx['config']
        storage = ctx['storage']
    """

    @staticmethod
    def get_project_path(project: Optional[Path] = None) -> Path:
        """Get and validate project path from option or current directory.

        Args:
            project: Project path from --project option (optional)

        Returns:
            Resolved absolute project path

        Example:
            path = CLIInitializer.get_project_path()
            # or with explicit path:
            path = CLIInitializer.get_project_path(Path("/my/project"))
        """
        if project is not None:
            return project.resolve()
        return Path.cwd()

    @staticmethod
    def ensure_project_initialized(project_path: Path) -> None:
        """Check project is initialized or exit with helpful error.

        Args:
            project_path: Project directory to check

        Raises:
            SystemExit: If project not initialized (via CLIErrorHandler)

        Example:
            CLIInitializer.ensure_project_initialized(Path.cwd())
        """
        # Check for config file (Commandment #7: Check inputs)
        config_file = project_path / "ingestforge.yaml"

        if not config_file.exists():
            error = FileNotFoundError(
                f"Not an IngestForge project (missing {config_file})\n\n"
                "Initialize a project with:\n"
                "  ingestforge init <project-name>"
            )
            CLIErrorHandler.exit_on_error(error, "Project not initialized")

    @staticmethod
    def load_config(project_path: Optional[Path] = None) -> Any:
        """Load project configuration with validation.

        Args:
            project_path: Project directory (default: current directory)

        Returns:
            Loaded Config object

        Raises:
            SystemExit: If config loading fails

        Example:
            config = CLIInitializer.load_config()
        """
        # Lazy import to avoid circular dependencies
        from ingestforge.core.config_loaders import load_config

        # Get and validate project path
        project_path = CLIInitializer.get_project_path(project_path)
        CLIInitializer.ensure_project_initialized(project_path)

        # Load config with error handling (Commandment #1: Simple flow)
        try:
            return load_config(base_path=project_path)
        except Exception as e:
            CLIErrorHandler.exit_on_error(e, "Failed to load configuration")

    @staticmethod
    def load_config_and_storage(
        project_path: Optional[Path] = None,
    ) -> Tuple[Any, Any]:
        """Load configuration and storage backend.

        This is the most common initialization pattern for CLI commands
        that need to access the knowledge base.

        Args:
            project_path: Project directory (optional)

        Returns:
            Tuple of (Config, ChunkRepository)

        Raises:
            SystemExit: If initialization fails

        Example:
            config, storage = CLIInitializer.load_config_and_storage()
            results = storage.search("query")
        """
        # Lazy imports (Commandment #6: Smallest scope)
        from ingestforge.storage.factory import get_storage_backend

        # Load config first
        config = CLIInitializer.load_config(project_path)

        # Initialize storage with error handling
        try:
            storage = get_storage_backend(config)
            return config, storage
        except Exception as e:
            CLIErrorHandler.exit_on_error(e, "Failed to initialize storage backend")

    @staticmethod
    def get_pipeline(project_path: Optional[Path] = None) -> Any:
        """Get configured processing pipeline.

        Args:
            project_path: Project directory (optional)

        Returns:
            Initialized Pipeline instance

        Raises:
            SystemExit: If pipeline initialization fails

        Example:
            pipeline = CLIInitializer.get_pipeline()
            result = pipeline.process_file(file_path)
        """
        # Lazy imports
        from ingestforge.core.pipeline import Pipeline

        # Get path and load config
        project_path = CLIInitializer.get_project_path(project_path)
        config = CLIInitializer.load_config(project_path)

        # Initialize pipeline with error handling
        try:
            return Pipeline(config, project_path)
        except Exception as e:
            CLIErrorHandler.exit_on_error(e, "Failed to initialize pipeline")

    @staticmethod
    def _build_context_dict(
        config: Any, project_path: Path, require_storage: bool, require_pipeline: bool
    ) -> Dict[str, Any]:
        """Build context dictionary with optional components.

        Rule #4: No large functions - Extracted from initialize_for_command
        """
        result = {
            "config": config,
            "project_path": project_path,
            "storage": None,
            "pipeline": None,
        }

        # Conditionally initialize storage (Commandment #1: Simple flow)
        if require_storage:
            result["storage"] = CLIInitializer._init_storage(config)

        # Conditionally initialize pipeline
        if require_pipeline:
            result["pipeline"] = CLIInitializer._init_pipeline(config, project_path)

        return result

    @staticmethod
    def initialize_for_command(
        project_path: Optional[Path] = None,
        require_storage: bool = True,
        require_pipeline: bool = False,
    ) -> Dict[str, Any]:
        """One-stop initialization for CLI commands.

        Rule #4: Function <60 lines (refactored to 43 lines)

        Returns context dict with all initialized components.

        Args:
            project_path: Project directory (optional)
            require_storage: Whether to initialize storage backend
            require_pipeline: Whether to initialize pipeline

        Returns:
            Dict with 'config', 'project_path', 'storage', 'pipeline'

        Raises:
            SystemExit: If any initialization step fails

        Example:
            ctx = CLIInitializer.initialize_for_command(require_storage=True)
            results = ctx['storage'].search("query")
        """
        # Get project path and load config (always needed)
        project_path = CLIInitializer.get_project_path(project_path)
        config = CLIInitializer.load_config(project_path)

        # Build context with optional components
        return CLIInitializer._build_context_dict(
            config, project_path, require_storage, require_pipeline
        )

    @staticmethod
    def _init_storage(config: Any) -> Any:
        """Initialize storage backend (private helper).

        Args:
            config: Config object

        Returns:
            ChunkRepository instance

        Raises:
            SystemExit: If initialization fails
        """
        from ingestforge.storage.factory import get_storage_backend

        try:
            return get_storage_backend(config)
        except Exception as e:
            CLIErrorHandler.exit_on_error(e, "Failed to initialize storage backend")

    @staticmethod
    def _init_pipeline(config: Any, project_path: Path) -> Any:
        """Initialize pipeline (private helper).

        Args:
            config: Config object
            project_path: Project directory

        Returns:
            Pipeline instance

        Raises:
            SystemExit: If initialization fails
        """
        from ingestforge.core.pipeline import Pipeline

        try:
            return Pipeline(config, project_path)
        except Exception as e:
            CLIErrorHandler.exit_on_error(e, "Failed to initialize pipeline")
