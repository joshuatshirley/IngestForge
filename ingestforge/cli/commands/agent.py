"""CLI commands for autonomous agent.

Provides commands for running the ReAct agent with LLM integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ingestforge.agent.llm_adapter import create_llm_think_adapter
from ingestforge.agent.react_engine import (
    AgentResult,
    ReActEngine,
    create_engine,
)
from ingestforge.agent.tool_registry import (
    ToolRegistry,
    ToolCategory,
    create_registry,
    register_builtin_tools,
)
from ingestforge.agent.synthesis import (
    ReportFormat,
    ReportExporter,
    synthesize_report,
)
from ingestforge.cli.core import CLIInitializer
from ingestforge.core.logging import get_logger
from ingestforge.llm.base import GenerationConfig
from ingestforge.llm.factory import get_llm_client

logger = get_logger(__name__)
console = Console()
MAX_TASK_LENGTH = 2000
DEFAULT_MAX_STEPS = 10
MAX_PROVIDER_NAME = 50
MAX_MODEL_NAME = 100


@click.group(name="agent")
def agent_group() -> None:
    """Autonomous agent commands."""
    pass


@agent_group.command(name="run")
@click.argument("task")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["ollama", "claude", "openai", "gemini", "llamacpp"]),
    help="LLM provider (defaults to config)",
)
@click.option(
    "--model",
    "-M",
    help="Model name override",
)
@click.option(
    "--max-steps",
    "-m",
    default=DEFAULT_MAX_STEPS,
    help="Maximum agent steps",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output report file",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "html", "json"]),
    default="markdown",
    help="Report format",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    help="Project directory (defaults to current)",
)
@click.option(
    "--domain-aware",
    "-d",
    is_flag=True,
    help="Enable domain-aware tool filtering (STORY-30)",
)
@click.option(
    "--no-grammar",
    is_flag=True,
    help="Disable GBNF grammar constraint (for debugging)",
)
def run_agent(
    task: str,
    provider: Optional[str],
    model: Optional[str],
    max_steps: int,
    output: Optional[Path],
    format: str,
    quiet: bool,
    project: Optional[Path],
    domain_aware: bool,
    no_grammar: bool,
) -> None:
    """Run autonomous research agent with LLM.

    TASK is the research task or question.

    Examples:
        ingestforge agent run "What are the key findings about solar panels?"
        ingestforge agent run "Summarize recent documents" --provider ollama
        ingestforge agent run "Research topic X" --max-steps 20 --output report.md
    """
    if not task.strip():
        console.print("[red]Error: Empty task[/red]")
        raise SystemExit(1)

    task = task[:MAX_TASK_LENGTH]

    # Pre-flight resource check (80% threshold)
    from ingestforge.core.system import (
        ResourceExhaustedError,
        check_resources,
        is_safe_to_proceed,
    )

    if not is_safe_to_proceed():
        console.print(
            "[yellow]Warning: System resources are high. "
            "The agent may be slow or fail.[/yellow]"
        )
        status = check_resources()
        console.print(
            f"[dim]RAM: {status.ram_percent:.1f}% | "
            f"CPU: {status.cpu_percent:.1f}% | "
            f"VRAM: {status.vram_percent:.1f}%[/dim]"
            if status.vram_percent
            else f"[dim]RAM: {status.ram_percent:.1f}% | CPU: {status.cpu_percent:.1f}%[/dim]"
        )

    # Load configuration and initialize components
    try:
        ctx = _initialize_agent_context(project, provider, model)
    except Exception as e:
        console.print(f"[red]Initialization failed: {e}[/red]")
        raise SystemExit(1)

    # Setup registry and register tools
    registry = _setup_tool_registry(ctx)

    # STORY-30: Apply domain-aware tool filtering if enabled
    if domain_aware:
        registry = _apply_domain_filtering(registry, task, quiet)

    # Determine if grammar should be used (default: yes for llamacpp)
    use_grammar = _should_use_grammar(ctx, no_grammar, quiet)

    # Create engine with LLM think adapter
    engine = _create_llm_engine(ctx, registry, max_steps, use_grammar)

    # Run with progress display and resource monitoring
    try:
        result = _run_with_progress(engine, task, quiet)
    except ResourceExhaustedError as e:
        console.print(f"\n[red]Operation stopped: {e}[/red]")
        console.print(
            "[yellow]Tip: Close other applications or use a smaller model.[/yellow]"
        )
        raise SystemExit(1)

    # Display result
    _display_result(result)

    # Export report if requested
    if output:
        _export_report(result, output, format)


def _initialize_agent_context(
    project: Optional[Path],
    provider: Optional[str],
    model: Optional[str],
) -> dict[str, Any]:
    """Initialize agent context with config, storage, and LLM.

    Args:
        project: Project directory
        provider: LLM provider override
        model: Model name override

    Returns:
        Context dictionary with config, storage, pipeline, llm_client

    Raises:
        Exception: If initialization fails
    """
    # Load config and storage
    config, storage = CLIInitializer.load_config_and_storage(project)

    # Get pipeline for document ingestion tool
    pipeline = CLIInitializer.get_pipeline(project)

    # Create LLM client
    llm_client = _create_llm_client(config, provider, model)

    return {
        "config": config,
        "storage": storage,
        "pipeline": pipeline,
        "llm_client": llm_client,
    }


def _create_llm_client(
    config: Any,
    provider: Optional[str],
    model: Optional[str],
) -> Any:
    """Create LLM client with provider and model overrides.

    Args:
        config: IngestForge config
        provider: Provider override
        model: Model override

    Returns:
        LLM client instance

    Raises:
        ValueError: If credentials missing or provider unavailable
    """
    if provider:
        provider = provider[:MAX_PROVIDER_NAME]

    # Get client from factory
    llm_client = get_llm_client(config, provider)

    # Override model if specified
    if model:
        model = model[:MAX_MODEL_NAME]
        if hasattr(llm_client, "_model"):
            llm_client._model = model

    # Check credentials/availability
    if not llm_client.is_available():
        _handle_missing_credentials(provider or config.llm.default_provider)

    return llm_client


def _handle_missing_credentials(provider: str) -> None:
    """Handle missing credentials for cloud providers.

    Args:
        provider: Provider name

    Raises:
        SystemExit: Always exits with error
    """
    # Cloud providers and their env vars
    cloud_providers = {
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    if provider in cloud_providers:
        env_var = cloud_providers[provider]
        console.print(
            f"[red]Error: {provider} credentials not found[/red]\n"
            f"Set {env_var} environment variable or configure in ingestforge.yaml"
        )
    else:
        console.print(f"[red]Error: {provider} not available[/red]")

    raise SystemExit(1)


def _setup_tool_registry(ctx: dict[str, Any]) -> ToolRegistry:
    """Setup tool registry with built-in and knowledge tools.

    Args:
        ctx: Agent context with storage and pipeline

    Returns:
        Configured tool registry
    """
    from ingestforge.agent.knowledge_tools import register_knowledge_tools
    from ingestforge.agent.web_tools import register_web_tools
    from ingestforge.agent.vision_tools import register_vision_tools
    from ingestforge.agent.domain_tools import register_domain_tools
    from ingestforge.ingest.ocr.vlm_processor import VLMProcessor

    # Create registry
    registry = create_registry()

    # Register built-in tools
    register_builtin_tools(registry)

    # Register web research tools
    register_web_tools(registry)

    # Register vision research tools (Task 10.1.2)
    vlm = VLMProcessor(model_path=ctx["config"].llm.llamacpp.model_path)
    register_vision_tools(registry, ctx["storage"], vlm)

    # Register knowledge base tools
    register_knowledge_tools(
        registry=registry,
        storage=ctx["storage"],
        pipeline=ctx["pipeline"],
    )

    # STORY-30: Register domain-specific discovery tools
    register_domain_tools(registry)

    logger.info(f"Registered {registry.tool_count} tools")
    return registry


def _apply_domain_filtering(
    registry: ToolRegistry,
    task: str,
    quiet: bool,
) -> ToolRegistry:
    """Apply domain-aware tool filtering based on task content.

    STORY-30: Detects domains from task and activates relevant tools.

    Args:
        registry: Tool registry
        task: User's research task
        quiet: Suppress output

    Returns:
        Registry with domain-filtered tools
    """
    from ingestforge.query.domain_classifier import QueryDomainClassifier

    classifier = QueryDomainClassifier()
    detected_domains = classifier.classify_query(task)

    if detected_domains:
        enabled_count = registry.activate_for_domains(detected_domains)

        if not quiet:
            console.print(
                f"[blue]Detected domains: {', '.join(detected_domains)}[/blue]"
            )
            console.print(
                f"[blue]Activated {enabled_count} domain-relevant tools[/blue]\n"
            )
    else:
        if not quiet:
            console.print(
                "[yellow]No specific domain detected, using all tools[/yellow]\n"
            )

    return registry


def _should_use_grammar(
    ctx: dict[str, Any],
    no_grammar: bool,
    quiet: bool,
) -> bool:
    """Determine if GBNF grammar should be used.

    Grammar is enabled by default for llama-cpp to guarantee valid
    ReAct format output and eliminate parsing failures.

    Args:
        ctx: Agent context with llm_client
        no_grammar: User flag to disable grammar
        quiet: Suppress output

    Returns:
        True if grammar should be used
    """
    if no_grammar:
        return False

    # Check if using llama-cpp provider
    llm_client = ctx["llm_client"]
    is_llamacpp = type(llm_client).__name__ == "LlamaCppClient"

    if is_llamacpp and not quiet:
        console.print("[dim]Using GBNF grammar for constrained ReAct output[/dim]")

    return is_llamacpp


def _create_llm_engine(
    ctx: dict[str, Any],
    registry: ToolRegistry,
    max_steps: int,
    use_grammar: bool = False,
) -> ReActEngine:
    """Create ReAct engine with LLM think adapter.

    Args:
        ctx: Agent context with llm_client
        registry: Tool registry
        max_steps: Maximum agent steps
        use_grammar: Use GBNF grammar for constrained output

    Returns:
        Configured ReAct engine
    """
    # Create LLM think adapter
    gen_config = GenerationConfig(
        max_tokens=1000,
        temperature=0.7,
        stop_sequences=["Observation:"],
    )

    think_adapter = create_llm_think_adapter(
        llm_client=ctx["llm_client"],
        config=gen_config,
        use_grammar=use_grammar,
    )

    # Create engine with LLM client for audit/revision
    engine = create_engine(
        think_fn=think_adapter,
        max_iterations=max_steps,
        llm_client=ctx["llm_client"],
    )

    # Register tools from registry
    for name in registry.tool_names:
        tool = registry.get_as_protocol(name)
        if tool:
            engine.register_tool(tool)

    return engine


def _run_with_progress(
    engine: ReActEngine,
    task: str,
    quiet: bool,
) -> AgentResult:
    """Run agent with progress display.

    Args:
        engine: ReAct engine
        task: Research task
        quiet: Suppress output

    Returns:
        Agent result
    """
    if quiet:
        return engine.run(task)

    # Use ASCII-safe spinner for Windows compatibility
    with Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description="Running agent...", total=None)
        result = engine.run(task)

    return result


def _display_result(result: AgentResult) -> None:
    """Display agent result.

    Args:
        result: Agent result
    """
    # Status panel
    status = "[green]Success[/green]" if result.success else "[red]Failed[/red]"
    state = result.state.value

    panel = Panel(
        f"Status: {status}\nState: {state}\nIterations: {result.iterations}",
        title="Agent Result",
        border_style="blue",
    )
    console.print(panel)

    # Steps table
    if result.steps:
        table = Table(title="Execution Steps")
        table.add_column("Step", style="cyan")
        table.add_column("Action", style="green")
        table.add_column("Observation", style="yellow")

        for step in result.steps[:10]:
            action = step.action or "complete"
            obs = (
                step.observation[:50] + "..."
                if len(step.observation) > 50
                else step.observation
            )
            table.add_row(str(step.iteration), action, obs)

        console.print(table)

    # Final answer
    console.print(f"\n[bold]Answer:[/bold] {result.final_answer}")


def _export_report(
    result: AgentResult,
    output: Path,
    format_str: str,
) -> None:
    """Export report to file.

    Args:
        result: Agent result
        output: Output path
        format_str: Format string
    """
    format_map = {
        "markdown": ReportFormat.MARKDOWN,
        "html": ReportFormat.HTML,
        "json": ReportFormat.JSON,
    }
    report_format = format_map.get(format_str, ReportFormat.MARKDOWN)

    report = synthesize_report(result)
    exporter = ReportExporter()

    if exporter.export(report, output, report_format):
        console.print(f"[green]Report saved to {output}[/green]")
    else:
        console.print("[red]Failed to export report[/red]")


@agent_group.command(name="tools")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    help="Project directory (defaults to current)",
)
def list_tools(project: Optional[Path]) -> None:
    """List available agent tools.

    Shows built-in tools and knowledge base tools if project initialized.
    """
    registry = create_registry()
    register_builtin_tools(registry)

    # Try to add knowledge tools if in a project
    try:
        config, storage = CLIInitializer.load_config_and_storage(project)
        pipeline = CLIInitializer.get_pipeline(project)

        from ingestforge.agent.knowledge_tools import register_knowledge_tools

        register_knowledge_tools(registry, storage, pipeline)
        console.print("[green]Showing tools with knowledge base access[/green]\n")
    except Exception:
        console.print("[yellow]Showing built-in tools only (no project)[/yellow]\n")

    table = Table(title="Available Agent Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Description", style="yellow")

    for name in registry.tool_names:
        tool = registry.get(name)
        if tool:
            table.add_row(
                name,
                tool.metadata.category.value,
                tool.description[:60],
            )

    console.print(table)


@agent_group.command(name="status")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    help="Project directory (defaults to current)",
)
@click.option(
    "--provider",
    "-p",
    help="Test specific provider",
)
def agent_status(project: Optional[Path], provider: Optional[str]) -> None:
    """Show agent status and capabilities.

    Tests LLM connectivity and shows available tools.
    """
    # Basic status
    registry = create_registry()
    register_builtin_tools(registry)

    # Check for project
    has_project = False
    try:
        config, storage = CLIInitializer.load_config_and_storage(project)
        pipeline = CLIInitializer.get_pipeline(project)

        from ingestforge.agent.knowledge_tools import register_knowledge_tools

        register_knowledge_tools(registry, storage, pipeline)
        has_project = True
    except Exception:
        pass

    # Test LLM connectivity
    llm_status = "Not configured"
    llm_model = "N/A"

    if has_project:
        try:
            llm_client = _create_llm_client(config, provider, None)
            if llm_client.is_available():
                llm_status = "Available"
                llm_model = llm_client.model_name
            else:
                llm_status = "Unavailable"
        except Exception as e:
            llm_status = f"Error: {str(e)[:40]}"

    # Display status
    status_text = (
        f"Tools registered: {registry.tool_count}\n"
        f"Tool categories: {len(ToolCategory)}\n"
        f"Max steps: {DEFAULT_MAX_STEPS}\n"
        f"Project initialized: {has_project}\n"
        f"LLM status: {llm_status}\n"
        f"LLM model: {llm_model}"
    )

    console.print(
        Panel(
            status_text,
            title="Agent Status",
            border_style="green" if llm_status == "Available" else "yellow",
        )
    )


@agent_group.command(name="test-llm")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    help="Project directory (defaults to current)",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["ollama", "claude", "openai", "gemini", "llamacpp"]),
    help="LLM provider to test",
)
def test_llm(project: Optional[Path], provider: Optional[str]) -> None:
    """Test LLM connectivity and configuration.

    Verifies that the LLM provider is properly configured and can
    generate responses.

    Examples:
        ingestforge agent test-llm
        ingestforge agent test-llm --provider ollama
    """
    try:
        # Load config
        config, _ = CLIInitializer.load_config_and_storage(project)

        # Create LLM client
        console.print("[blue]Creating LLM client...[/blue]")
        llm_client = _create_llm_client(config, provider, None)

        provider_name = provider or config.llm.default_provider
        console.print(f"[green]Using provider: {provider_name}[/green]")
        console.print(f"[green]Model: {llm_client.model_name}[/green]\n")

        # Test basic generation
        console.print("[blue]Testing generation...[/blue]")
        test_prompt = "Say 'Hello from IngestForge' and nothing else."

        with Progress(
            SpinnerColumn(spinner_name="line"),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Generating...", total=None)
            response = llm_client.generate(test_prompt)

        # Display result
        console.print("\n[green]Success! LLM Response:[/green]")
        console.print(Panel(response, border_style="green"))

        # Show usage if available
        usage = llm_client.get_usage()
        if usage and usage.get("total_tokens", 0) > 0:
            console.print(
                f"\n[dim]Tokens used: {usage['total_tokens']} "
                f"(prompt: {usage['prompt_tokens']}, "
                f"completion: {usage['completion_tokens']})[/dim]"
            )

    except Exception as e:
        console.print(f"\n[red]LLM test failed: {e}[/red]")
        raise SystemExit(1)


# Export for CLI integration
agent_command = agent_group
