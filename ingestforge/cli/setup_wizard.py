"""Setup Wizard - Interactive configuration for IngestForge.

Guided CLI Setup Wizard
Enhanced Setup Wizard

Provides intelligent configuration with:
- Hardware detection and capability assessment ()
- Configuration presets based on system resources ()
- Embedding model download with progress ()
- Config file generation with validation ()
- Interactive prompts with rich UI ()
- Post-setup verification ()

Follows NASA JPL Power of Ten rules.
"""

from __future__ import annotations

import itertools
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table
from rich.markdown import Markdown

# =============================================================================
# CONSTANTS (JPL Rule #2: Bounded Data Structures)
# =============================================================================

MAX_PATH_LENGTH = 260  # Windows max path
MAX_API_KEY_LENGTH = 256  # Reasonable limit for API keys
MAX_MODEL_NAME_LENGTH = 128  # Model name limit
SUPPORTED_PROVIDERS = ["llamacpp", "ollama", "openai", "claude", "gemini"]
DEFAULT_DATA_DIR = ".data"
CONFIG_FILENAME = "config.yaml"

# Enhanced Setup Wizard Constants
MAX_HARDWARE_CHECKS = 5  # JPL Rule #2: Bounded hardware detection
MAX_DOWNLOAD_RETRIES = 3  # JPL Rule #2: Bounded model download attempts
MAX_VALIDATION_ATTEMPTS = 3  # JPL Rule #2: Bounded config validation
MAX_PROMPT_RETRIES = 5  # JPL Rule #2: Bounded user input attempts
MAX_VERIFICATION_CHECKS = 5  # JPL Rule #2: Bounded setup verification

# Hardware thresholds for presets
RAM_THRESHOLD_LIGHTWEIGHT = 12.0  # GB
RAM_THRESHOLD_PERFORMANCE = 28.0  # GB
CORES_THRESHOLD_LIGHTWEIGHT = 3
CORES_THRESHOLD_PERFORMANCE = 7

# JPL Rule #2: Bounded retry/check loops
MAX_HARDWARE_CHECKS = 5
MAX_DOWNLOAD_RETRIES = 3
MAX_VALIDATION_ATTEMPTS = 3
MAX_PROMPT_RETRIES = 5
MAX_VERIFICATION_CHECKS = 10
MAX_MODEL_FILES_CHECK = 100  # JPL Rule #2: Max files to check in model cache directory

# =============================================================================
# TYPE DEFINITIONS (AC, JPL Rule #9)
# =============================================================================


class HardwareSpec(TypedDict):
    """Hardware specifications detected from system.

    Hardware Detection
    JPL Rule #9: TypedDict for structured data.
    """

    cpu_cores: int
    ram_gb: float
    has_gpu: bool
    disk_free_gb: float
    platform: str


class PresetType(str, Enum):
    """Configuration preset types based on hardware.

    Configuration Presets
    - LIGHTWEIGHT: 8-12GB RAM, 2-3 cores
    - BALANCED: 12-28GB RAM, 4-7 cores [RECOMMENDED]
    - PERFORMANCE: 28GB+ RAM, 8+ cores
    """

    LIGHTWEIGHT = "lightweight"
    BALANCED = "balanced"
    PERFORMANCE = "performance"


class ConfigPreset(str, Enum):
    """Legacy configuration presets (backward compatibility).

    Maps to PresetType:
    - STANDARD → BALANCED
    - EXPERT → PERFORMANCE
    """

    STANDARD = "standard"
    EXPERT = "expert"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class LLMProviderSetup:
    """LLM provider configuration collected from user."""

    provider: str = "llamacpp"
    api_key: str = ""
    model_name: str = ""
    model_path: str = ""  # For local models
    ollama_url: str = "http://localhost:11434"


@dataclass
class SetupConfig:
    """Complete setup configuration from wizard."""

    project_name: str = "my_research"
    data_path: Path = field(default_factory=lambda: Path(DEFAULT_DATA_DIR))
    storage_backend: str = "chromadb"
    llm: LLMProviderSetup = field(default_factory=LLMProviderSetup)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    performance_mode: str = "balanced"
    enable_ocr: bool = False
    # Add preset field
    preset: PresetType = PresetType.BALANCED
    workers: int = 4
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    passed: bool
    message: str
    suggestion: Optional[str] = None


# =============================================================================
# HARDWARE DETECTION ()
# =============================================================================


def detect_hardware() -> HardwareSpec:
    """Detect system hardware capabilities.

    Hardware Detection
    - Detects CPU cores
    - Detects RAM
    - Detects GPU availability
    - Detects disk space

    JPL Compliance:
    - Rule #2: Bounded checks (MAX_HARDWARE_CHECKS)
    - Rule #4: <50 lines
    - Rule #5: Assertions for validation
    - Rule #9: Complete type hints

    Returns:
        HardwareSpec with detected capabilities
    """
    import psutil

    # Detect CPU cores (physical only)
    cpu_cores = psutil.cpu_count(logical=False) or 1
    # JPL Rule #5: Postcondition assertion
    assert cpu_cores > 0, "CPU cores must be positive"

    # Detect RAM
    ram_bytes = psutil.virtual_memory().total
    assert ram_bytes > 0, "RAM size must be positive"
    ram_gb = ram_bytes / (1024**3)

    # Detect GPU (graceful fallback if torch not installed)
    has_gpu = False
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:
        pass  # torch not installed, no GPU

    # Detect free disk space
    disk_usage = psutil.disk_usage(Path.cwd())
    disk_free_gb = disk_usage.free / (1024**3)

    spec: HardwareSpec = {
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb,
        "has_gpu": has_gpu,
        "disk_free_gb": disk_free_gb,
        "platform": platform.system().lower(),
    }

    # JPL Rule #5: Postcondition assertions
    assert spec["cpu_cores"] > 0, "Postcondition: CPU cores must be positive"
    assert spec["ram_gb"] > 0, "Postcondition: RAM must be positive"

    return spec


def recommend_preset(spec: HardwareSpec) -> PresetType:
    """Recommend configuration preset based on hardware.

    Configuration Presets
    Decision tree:
    - RAM >= 28GB AND cores >= 8 → PERFORMANCE
    - RAM >= 12GB AND cores >= 4 → BALANCED
    - Otherwise → LIGHTWEIGHT

    JPL Rule #4: <40 lines (decision logic)

    Args:
        spec: Detected hardware specifications

    Returns:
        Recommended preset type
    """
    ram = spec["ram_gb"]
    cores = spec["cpu_cores"]

    # High-end hardware
    if ram >= RAM_THRESHOLD_PERFORMANCE and cores >= CORES_THRESHOLD_PERFORMANCE:
        return PresetType.PERFORMANCE

    # Mid-range hardware
    if ram >= RAM_THRESHOLD_LIGHTWEIGHT and cores >= CORES_THRESHOLD_LIGHTWEIGHT:
        return PresetType.BALANCED

    # Low-end hardware
    return PresetType.LIGHTWEIGHT


def map_legacy_preset(legacy: ConfigPreset) -> PresetType:
    """Map legacy ConfigPreset to new PresetType.

    Backward compatibility mapping:
    - STANDARD → BALANCED
    - EXPERT → PERFORMANCE

    Args:
        legacy: Legacy preset

    Returns:
        New preset type
    """
    if legacy == ConfigPreset.STANDARD:
        return PresetType.BALANCED
    elif legacy == ConfigPreset.EXPERT:
        return PresetType.PERFORMANCE
    else:
        return PresetType.BALANCED  # Default


# =============================================================================
# MODEL DOWNLOAD ()
# =============================================================================


def download_embedding_model(
    model_name: str, cache_dir: Path, console: Console
) -> bool:
    """Download sentence-transformers embedding model.

    Model Download & Verification
    - Downloads with Rich progress bar
    - Retries MAX_DOWNLOAD_RETRIES times
    - Verifies model loads successfully
    - Caches in ~/.ingestforge/models/

    JPL Compliance:
    - Rule #2: Bounded retry loop (MAX_DOWNLOAD_RETRIES)
    - Rule #5: Precondition/postcondition assertions
    - Rule #7: Returns explicit success status and checks I/O
    - Rule #4: <60 lines

    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache model
        console: Rich console for output

    Returns:
        True if successful, False otherwise
    """
    # JPL Rule #5: Precondition assertions
    assert len(model_name) > 0, "Model name must not be empty"
    assert (
        len(model_name) <= MAX_MODEL_NAME_LENGTH
    ), f"Model name too long (max {MAX_MODEL_NAME_LENGTH})"

    cache_dir.mkdir(parents=True, exist_ok=True)
    # JPL Rule #7: Verify mkdir succeeded
    assert cache_dir.exists(), "Failed to create cache directory"

    # Check if model already cached
    if _model_exists(model_name, cache_dir):
        console.print(f"[green]✓[/green] Model already cached: {model_name}")
        return True

    # JPL Rule #2: Bounded retry loop
    for attempt in range(MAX_DOWNLOAD_RETRIES):
        try:
            console.print(
                f"[yellow]Downloading {model_name}[/yellow] "
                f"(attempt {attempt + 1}/{MAX_DOWNLOAD_RETRIES})..."
            )

            from sentence_transformers import SentenceTransformer

            # Download with console feedback
            with console.status(f"[bold green]Downloading {model_name}..."):
                model = SentenceTransformer(model_name, cache_folder=str(cache_dir))

            # Verify model loads
            test_embedding = model.encode("test")
            if test_embedding is not None:
                console.print(f"[green]✓[/green] Model {model_name} ready")
                return True

        except Exception as e:
            console.print(f"[red]Download failed:[/red] {str(e)[:100]}")
            if attempt < MAX_DOWNLOAD_RETRIES - 1:
                console.print("[yellow]Retrying...[/yellow]")

    console.print(f"[red]✗ Failed to download {model_name}[/red]")
    return False


def _model_exists(model_name: str, cache_dir: Path) -> bool:
    """Check if model already cached.

    JPL Rule #2: Bounded iteration using MAX_MODEL_FILES_CHECK.

    Args:
        model_name: Model name
        cache_dir: Cache directory

    Returns:
        True if model exists
    """
    # JPL Rule #5: Precondition assertions
    assert len(model_name) > 0, "Model name must not be empty"
    assert cache_dir is not None, "Cache directory must not be None"

    # sentence-transformers uses model name as subdirectory
    model_path = cache_dir / model_name.replace("/", "_")

    # JPL Rule #2: Bound directory iteration to prevent unbounded loops
    # Use itertools.islice to limit iteration to MAX_MODEL_FILES_CHECK files
    return model_path.exists() and any(
        itertools.islice(model_path.iterdir(), MAX_MODEL_FILES_CHECK)
    )


# =============================================================================
# CONFIGURATION GENERATION ()
# =============================================================================


def get_model_for_preset(preset: PresetType) -> str:
    """Get embedding model name for preset.

    Preset → Model mapping
    - LIGHTWEIGHT: all-MiniLM-L6-v2 (110MB, 384 dim)
    - BALANCED: all-MiniLM-L6-v2 (110MB, 384 dim)
    - PERFORMANCE: all-mpnet-base-v2 (420MB, 768 dim)

    Args:
        preset: Configuration preset

    Returns:
        HuggingFace model name
    """
    models = {
        PresetType.LIGHTWEIGHT: "sentence-transformers/all-MiniLM-L6-v2",
        PresetType.BALANCED: "sentence-transformers/all-MiniLM-L6-v2",
        PresetType.PERFORMANCE: "sentence-transformers/all-mpnet-base-v2",
    }
    return models.get(preset, "sentence-transformers/all-MiniLM-L6-v2")


def get_workers_for_preset(preset: PresetType, spec: HardwareSpec) -> int:
    """Calculate worker count for preset.

    Args:
        preset: Configuration preset
        spec: Hardware specifications

    Returns:
        Number of workers
    """
    if preset == PresetType.LIGHTWEIGHT:
        return 1
    elif preset == PresetType.BALANCED:
        return min(4, spec["cpu_cores"])
    else:  # PERFORMANCE
        return min(8, spec["cpu_cores"])


def get_batch_size_for_preset(preset: PresetType) -> int:
    """Get batch size for preset.

    Args:
        preset: Configuration preset

    Returns:
        Batch size
    """
    sizes = {
        PresetType.LIGHTWEIGHT: 16,
        PresetType.BALANCED: 32,
        PresetType.PERFORMANCE: 64,
    }
    return sizes.get(preset, 32)


# =============================================================================
# POST-SETUP VERIFICATION ()
# =============================================================================


def verify_setup(config_path: Path, console: Console) -> bool:
    """Verify setup completed successfully.

    Post-Setup Verification
    - Tests config file exists
    - Tests model cache exists
    - Tests storage directory accessible
    - Tests ChromaDB connection (optional)

    JPL Rule #2: Bounded verification checks (MAX_VERIFICATION_CHECKS)

    Args:
        config_path: Path to config file
        console: Rich console

    Returns:
        True if all checks pass
    """
    console.print("\n[bold cyan]Verifying Setup...[/bold cyan]\n")

    checks = [
        ("Config file", config_path.exists()),
        ("Model cache", (Path.home() / ".ingestforge" / "models").exists()),
        ("Storage dir", (Path.home() / ".ingestforge" / "storage").exists()),
        ("Data dir", (Path.home() / ".ingestforge" / "data").exists()),
        ("Logs dir", (Path.home() / ".ingestforge" / "logs").exists()),
    ]

    # JPL Rule #2: Bounded iteration
    all_passed = True
    for check_name, result in checks[:MAX_VERIFICATION_CHECKS]:
        status = "[green]✓[/green]" if result else "[red]✗[/red]"
        console.print(f"  {status} {check_name}")
        all_passed = all_passed and result

    # Optional: Test ChromaDB connection
    chroma_ok, chroma_msg = _test_chromadb_connection()
    status = "[green]✓[/green]" if chroma_ok else "[yellow]⚠[/yellow]"
    console.print(f"  {status} ChromaDB: {chroma_msg}")

    if all_passed:
        console.print("\n[bold green]✓ All verifications passed![/bold green]")
    else:
        console.print("\n[yellow]Some verifications failed[/yellow]")

    return all_passed


def _test_chromadb_connection() -> Tuple[bool, str]:
    """Test ChromaDB connection.

    Returns:
        (success: bool, message: str)
    """
    try:
        import chromadb

        client = chromadb.Client()
        # Try to create and delete test collection
        collection = client.create_collection("_test")
        client.delete_collection("_test")
        return (True, "Connection OK")
    except Exception as e:
        return (False, f"Not available ({str(e)[:30]}...)")


# =============================================================================
# WIZARD CLASS (Enhanced for )
# =============================================================================


class SetupWizard:
    """Interactive setup wizard for IngestForge configuration.

    Enhanced with features:
    - Hardware detection and display
    - Intelligent preset recommendation
    - Model download with progress
    - Post-setup verification

    JPL Rule #4: All methods <60 lines
    JPL Rule #5: Assert preconditions
    JPL Rule #7: Check all return values
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialize wizard.

        Args:
            console: Rich console for output
        """
        self.console = console or Console()
        self.config = SetupConfig()
        self.hardware_spec: Optional[HardwareSpec] = None

    def run(
        self,
        preset: Optional[ConfigPreset] = None,
        non_interactive: bool = False,
    ) -> Optional[Path]:
        """Run the complete setup wizard.

        Interactive Prompts with --non-interactive support

        Args:
            preset: Configuration preset (legacy, optional)
            non_interactive: Skip prompts, use defaults

        Returns:
            Path to generated config file, or None on failure
        """
        try:
            # Display welcome
            if not non_interactive:
                self._display_welcome()
                if not self._confirm_start():
                    self.console.print("\n[dim]Setup cancelled.[/dim]")
                    return None

            # Detect hardware
            self.hardware_spec = detect_hardware()
            if not non_interactive:
                self._display_hardware(self.hardware_spec)

            # Recommend preset
            recommended_preset = recommend_preset(self.hardware_spec)

            # Map legacy preset if provided
            if preset is not None:
                recommended_preset = map_legacy_preset(preset)

            # Get preset choice (or use recommended in non-interactive mode)
            if non_interactive:
                chosen_preset = recommended_preset
            else:
                chosen_preset = self._get_preset_choice(recommended_preset)

            self.config.preset = chosen_preset

            # Apply preset settings
            self._apply_preset(chosen_preset)

            # Step 1: Project basics
            if not non_interactive:
                self._step_project_basics()

            # Step 2: LLM configuration (skip in non-interactive)
            if not non_interactive:
                self._step_llm_configuration()

            # Download embedding model
            model_name = get_model_for_preset(chosen_preset)
            cache_dir = Path.home() / ".ingestforge" / "models"

            self.console.print(
                f"\n[bold cyan]Downloading Embedding Model ({chosen_preset.value})[/bold cyan]\n"
            )
            if not download_embedding_model(model_name, cache_dir, self.console):
                # Fallback to smaller model
                self.console.print(
                    "[yellow]Falling back to lightweight model...[/yellow]"
                )
                fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                if not download_embedding_model(
                    fallback_model, cache_dir, self.console
                ):
                    self.console.print(
                        "[red]Model download failed. Setup incomplete.[/red]"
                    )
                    return None
                model_name = fallback_model

            self.config.embedding_model = model_name

            # Step 3: Health check
            if not non_interactive:
                health_ok = self._step_health_check()
            else:
                health_ok = True

            # Step 4: Verify write permissions
            if not self._verify_write_permissions():
                return None

            # Generate config
            config_path = self._step_generate_config(non_interactive)

            # Verify setup
            if verify_setup(config_path, self.console):
                # Display completion
                if not non_interactive:
                    self._display_completion(config_path, health_ok)
                return config_path
            else:
                self.console.print("[yellow]Setup verification had warnings[/yellow]")
                return config_path

        except KeyboardInterrupt:
            self.console.print("\n\n[dim]Setup interrupted.[/dim]")
            return None

    # -------------------------------------------------------------------------
    # Hardware Detection Display
    # -------------------------------------------------------------------------

    def _display_hardware(self, spec: HardwareSpec) -> None:
        """Display detected hardware specifications.

        Hardware Detection Display

        Args:
            spec: Detected hardware specs
        """
        self.console.print("\n[bold cyan]Detected Hardware[/bold cyan]\n")

        table = Table(show_header=False)
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Value", style="green")

        table.add_row("CPU Cores", str(spec["cpu_cores"]))
        table.add_row("RAM", f"{spec['ram_gb']:.1f} GB")
        table.add_row("GPU", "Yes (CUDA)" if spec["has_gpu"] else "No")
        table.add_row("Free Disk", f"{spec['disk_free_gb']:.1f} GB")
        table.add_row("Platform", spec["platform"].capitalize())

        self.console.print(table)

    # -------------------------------------------------------------------------
    # Preset Selection
    # -------------------------------------------------------------------------

    def _get_preset_choice(self, recommended: PresetType) -> PresetType:
        """Get preset choice from user.

        & Interactive preset selection

        Args:
            recommended: Recommended preset

        Returns:
            Chosen preset
        """
        self.console.print(
            f"\n[bold green]Recommended preset: {recommended.value}[/bold green]"
        )

        self._display_preset_options()

        choice = Prompt.ask(
            "\n[cyan]Select configuration preset[/cyan]",
            choices=["lightweight", "balanced", "performance"],
            default=recommended.value,
        )

        return PresetType(choice)

    def _display_preset_options(self) -> None:
        """Display preset options."""
        self.console.print("\n[bold]Available Presets:[/bold]\n")

        presets_info = [
            (
                "lightweight",
                "8-12GB RAM, 2-3 cores",
                "Minimal resources, single worker",
            ),
            (
                "balanced",
                "12-28GB RAM, 4-7 cores",
                "Recommended for most users",
            ),
            (
                "performance",
                "28GB+ RAM, 8+ cores",
                "Maximum performance, parallel processing",
            ),
        ]

        for name, hw, desc in presets_info:
            self.console.print(f"  [cyan]{name:12}[/cyan] {hw:25} - {desc}")

    def _apply_preset(self, preset: PresetType) -> None:
        """Apply preset settings to config.

        Preset → Config mapping

        Args:
            preset: Selected preset
        """
        if self.hardware_spec is None:
            return

        self.config.workers = get_workers_for_preset(preset, self.hardware_spec)
        self.config.batch_size = get_batch_size_for_preset(preset)
        self.config.device = "cuda" if self.hardware_spec["has_gpu"] else "cpu"

        # Set performance mode name
        mode_map = {
            PresetType.LIGHTWEIGHT: "speed",
            PresetType.BALANCED: "balanced",
            PresetType.PERFORMANCE: "quality",
        }
        self.config.performance_mode = mode_map.get(preset, "balanced")

    # -------------------------------------------------------------------------
    # Display Methods
    # -------------------------------------------------------------------------

    def _display_welcome(self) -> None:
        """Display welcome message."""
        welcome_text = """
# IngestForge Setup Wizard

This wizard will detect your hardware and configure IngestForge optimally.

**What we'll set up:**
1. **Hardware Detection** - Analyze your system capabilities
2. **Configuration Preset** - Optimize settings for your hardware
3. **Embedding Models** - Download AI models for semantic search
4. **LLM Provider** - Local (llama.cpp, Ollama) or cloud (OpenAI, Claude)
5. **Verification** - Test your setup

Setup typically takes **3-5 minutes** including model download.
"""
        panel = Panel(
            Markdown(welcome_text),
            title="[bold cyan]Welcome[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

    def _confirm_start(self) -> bool:
        """Confirm user wants to proceed."""
        return Confirm.ask("\n[cyan]Ready to begin setup?[/cyan]", default=True)

    def _display_completion(self, config_path: Path, health_ok: bool) -> None:
        """Display completion message.

        Args:
            config_path: Path to generated config
            health_ok: Whether all health checks passed
        """
        status = "[green]✓[/green]" if health_ok else "[yellow]![/yellow]"

        completion_text = f"""
# Setup Complete! {status}

Configuration saved to: `{config_path}`

**Next Steps:**
1. Try demo: `ingestforge demo`
2. Upload docs: `ingestforge ingest <path>`
3. Configure LLM: Edit `~/.ingestforge/config.yaml`
4. Start querying: `ingestforge query "your question"`

**Preset Applied:** {self.config.preset.value}
- Workers: {self.config.workers}
- Batch Size: {self.config.batch_size}
- Device: {self.config.device}
- Model: {self.config.embedding_model.split('/')[-1]}
"""
        panel = Panel(
            Markdown(completion_text),
            title="[bold green]Setup Complete[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)

    # -------------------------------------------------------------------------
    # Step 1: Project Basics
    # -------------------------------------------------------------------------

    def _step_project_basics(self) -> None:
        """Configure project basics (name, data path)."""
        self.console.print("\n[bold cyan]Step 1: Project Basics[/bold cyan]\n")

        # Project name
        self.config.project_name = Prompt.ask(
            "[cyan]Project name[/cyan]",
            default="my_research",
        )

        # Data path
        default_path = str(Path.cwd() / DEFAULT_DATA_DIR)
        path_input = Prompt.ask(
            "[cyan]Data directory[/cyan]",
            default=default_path,
        )

        # Validate path length (JPL Rule #2)
        assert (
            len(path_input) <= MAX_PATH_LENGTH
        ), f"Path too long (max {MAX_PATH_LENGTH})"

        self.config.data_path = Path(path_input)

        self.console.print(
            f"[green]✓[/green] Data will be stored in: {self.config.data_path}"
        )

    # -------------------------------------------------------------------------
    # Step 2: LLM Configuration
    # -------------------------------------------------------------------------

    def _step_llm_configuration(self) -> None:
        """Configure LLM provider."""
        self.console.print("\n[bold cyan]Step 2: LLM Configuration[/bold cyan]\n")

        # Show provider options
        self._display_provider_options()

        # Get provider selection
        provider = Prompt.ask(
            "[cyan]Select LLM provider[/cyan]",
            choices=SUPPORTED_PROVIDERS,
            default="llamacpp",
        )

        self.config.llm.provider = provider

        # Provider-specific configuration
        if provider == "llamacpp":
            self._configure_llamacpp()
        elif provider == "ollama":
            self._configure_ollama()
        elif provider in ("openai", "claude", "gemini"):
            self._configure_cloud_provider(provider)

    def _display_provider_options(self) -> None:
        """Display LLM provider options."""
        table = Table(title="Available LLM Providers", show_header=True)
        table.add_column("Provider", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Notes")

        table.add_row("llamacpp", "Local", "Recommended. No API key needed")
        table.add_row("ollama", "Local", "Easy local server")
        table.add_row("openai", "Cloud", "GPT-4o (requires API key)")
        table.add_row("claude", "Cloud", "Claude 3.5 (requires API key)")
        table.add_row("gemini", "Cloud", "Gemini (requires API key)")

        self.console.print(table)
        self.console.print()

    def _configure_llamacpp(self) -> None:
        """Configure llama.cpp local model."""
        self.console.print("[dim]llama.cpp runs models locally.[/dim]\n")

        model_path = Prompt.ask(
            "[cyan]Path to GGUF model file[/cyan]",
            default=".data/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        )

        assert len(model_path) <= MAX_PATH_LENGTH, "Path too long"

        self.config.llm.model_path = model_path

        if not Path(model_path).exists():
            self.console.print("[yellow]Note: Model not found[/yellow]")

        self.console.print("[green]✓[/green] LLM: llama.cpp")

    def _configure_ollama(self) -> None:
        """Configure Ollama local server."""
        self.console.print("[dim]Ollama provides local model serving.[/dim]\n")

        url = Prompt.ask(
            "[cyan]Ollama server URL[/cyan]",
            default="http://localhost:11434",
        )
        self.config.llm.ollama_url = url

        model = Prompt.ask(
            "[cyan]Model name[/cyan]",
            default="qwen2.5:14b",
        )

        assert len(model) <= MAX_MODEL_NAME_LENGTH, "Model name too long"

        self.config.llm.model_name = model
        self.console.print(f"[green]✓[/green] LLM: Ollama with {model}")

    def _configure_cloud_provider(self, provider: str) -> None:
        """Configure cloud LLM provider.

        Args:
            provider: Provider name
        """
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }
        model_defaults = {
            "openai": "gpt-4o-mini",
            "claude": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-1.5-flash",
        }

        env_var = env_var_map.get(provider, "")
        default_model = model_defaults.get(provider, "")

        # Check for existing environment variable
        existing_key = os.environ.get(env_var, "")
        if existing_key:
            self.console.print(f"[green]✓[/green] Found {env_var}")
            self.config.llm.api_key = f"${{{env_var}}}"
        else:
            self.config.llm.api_key = f"${{{env_var}}}"

        # Model selection
        model = Prompt.ask(
            f"[cyan]{provider.capitalize()} model[/cyan]",
            default=default_model,
        )
        assert len(model) <= MAX_MODEL_NAME_LENGTH

        self.config.llm.model_name = model
        self.console.print(f"[green]✓[/green] LLM: {provider} with {model}")

    # -------------------------------------------------------------------------
    # Step 3: Health Check
    # -------------------------------------------------------------------------

    def _step_health_check(self) -> bool:
        """Run system health checks.

        Returns:
            True if all checks passed
        """
        self.console.print("\n[bold cyan]Step 3: System Health Check[/bold cyan]\n")

        checks: List[HealthCheckResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running health checks...", total=None)

            checks.append(self._check_python_version())
            checks.append(self._check_data_directory())

            progress.update(task, completed=True)

        return self._display_health_results(checks)

    def _check_python_version(self) -> HealthCheckResult:
        """Check Python version."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.minor >= 10:
            return HealthCheckResult(
                name="Python Version",
                passed=True,
                message=f"Python {version_str}",
            )
        return HealthCheckResult(
            name="Python Version",
            passed=False,
            message=f"Python {version_str} (3.10+ required)",
            suggestion="Upgrade to Python 3.10+",
        )

    def _check_data_directory(self) -> HealthCheckResult:
        """Check data directory is accessible."""
        try:
            self.config.data_path.mkdir(parents=True, exist_ok=True)
            test_file = self.config.data_path / ".setup_test"
            test_file.write_text("test")
            test_file.unlink()

            return HealthCheckResult(
                name="Data Directory",
                passed=True,
                message=f"Writable: {self.config.data_path}",
            )
        except Exception as e:
            return HealthCheckResult(
                name="Data Directory",
                passed=False,
                message=f"Cannot write to {self.config.data_path}",
                suggestion=str(e),
            )

    def _display_health_results(self, checks: List[HealthCheckResult]) -> bool:
        """Display health check results.

        Args:
            checks: List of check results

        Returns:
            True if all checks passed
        """
        table = Table(show_header=True)
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Details")

        all_passed = True

        for check in checks:
            if check.passed:
                status = "[green]✓ Pass[/green]"
            else:
                status = "[red]✗ Fail[/red]"
                all_passed = False

            details = check.message
            if check.suggestion:
                details += f"\n[dim]{check.suggestion}[/dim]"

            table.add_row(check.name, status, details)

        self.console.print(table)

        if all_passed:
            self.console.print("\n[green]All checks passed![/green]")
        else:
            self.console.print("\n[yellow]Some checks failed[/yellow]")

        return all_passed

    # -------------------------------------------------------------------------
    # Step 4: Verify Write Permissions (JPL Rule #7)
    # -------------------------------------------------------------------------

    def _verify_write_permissions(self) -> bool:
        """Verify write permissions for data directory.

        Returns:
            True if write permissions verified
        """
        try:
            self.config.data_path.mkdir(parents=True, exist_ok=True)
            test_file = self.config.data_path / ".permissions_test"
            test_file.write_text("test")

            if not test_file.exists():
                return False

            test_file.unlink()
            return True

        except PermissionError:
            self.console.print(
                f"[red]No write permission for {self.config.data_path}[/red]"
            )
            return False
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Step 5: Generate Config ()
    # -------------------------------------------------------------------------

    def _step_generate_config(self, non_interactive: bool) -> Path:
        """Generate configuration file.

        Config File Generation with validation

        Args:
            non_interactive: Skip overwrite prompts

        Returns:
            Path to generated config file
        """
        self.console.print("\n[bold cyan]Generating Configuration[/bold cyan]\n")

        config_yaml = self._build_config_yaml()
        config_path = Path.home() / ".ingestforge" / CONFIG_FILENAME

        # Backup existing config
        if config_path.exists() and not non_interactive:
            backup_path = config_path.with_suffix(".yaml.backup")
            shutil.copy(config_path, backup_path)
            self.console.print(
                f"[dim]Backed up existing config to {backup_path.name}[/dim]"
            )

        # Write config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_yaml, encoding="utf-8")

        # Verify write succeeded (JPL Rule #7)
        assert config_path.exists(), "Failed to write config"
        assert config_path.read_text() == config_yaml, "Config verification failed"

        self.console.print(f"[green]✓[/green] Configuration saved: {config_path}")

        return config_path

    def _build_config_yaml(self) -> str:
        """Build YAML configuration string.

        Config with comments

        Returns:
            YAML configuration content
        """
        import yaml

        config_dict: Dict[str, Any] = {
            "version": "1.0.0",
            "project": {
                "name": self.config.project_name,
                "data_dir": str(self.config.data_path),
            },
            "storage": {
                "type": self.config.storage_backend,
                "path": str(Path.home() / ".ingestforge" / "storage"),
            },
            "embeddings": {
                "model": self.config.embedding_model,
                "device": self.config.device,
                "cache_dir": str(Path.home() / ".ingestforge" / "models"),
            },
            "pipeline": {
                "workers": self.config.workers,
                "batch_size": self.config.batch_size,
                "checkpoint_interval": 100,
            },
            "llm": {
                "provider": "none",  # User configures later
                "model": "",
            },
            "performance_mode": self.config.performance_mode,
        }

        return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)


# =============================================================================
# CLI COMMAND ()
# =============================================================================


def setup_command(
    preset: str = typer.Option(
        "standard",
        "--preset",
        "-p",
        help="Configuration preset (standard/expert) - legacy",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Skip prompts, use hardware-recommended defaults",
    ),
) -> None:
    """Interactive setup wizard for IngestForge configuration.

    Enhanced Setup Wizard with hardware detection and intelligent
    configuration recommendations.

    **Features:**
    - Automatic hardware detection
    - Intelligent preset recommendation
    - Embedding model download
    - Post-setup verification

    **Usage:**
        # Interactive setup (recommended)
        ingestforge setup

        # Non-interactive (CI/CD)
        ingestforge setup --non-interactive

        # Legacy preset (maps to new presets)
        ingestforge setup --preset expert
    """
    # Validate preset (legacy)
    preset_enum = None
    if preset and preset != "standard":
        try:
            preset_enum = ConfigPreset(preset.lower())
        except ValueError:
            console = Console()
            console.print(f"[red]Invalid preset: {preset}[/red]")
            raise typer.Exit(1)

    # Run wizard
    wizard = SetupWizard()
    config_path = wizard.run(preset=preset_enum, non_interactive=non_interactive)

    if config_path is None:
        raise typer.Exit(1)


# Alias for backwards compatibility
command = setup_command


if __name__ == "__main__":
    typer.run(setup_command)
