"""Shell command - Full-featured interactive REPL shell."""

from pathlib import Path

MAX_SHELL_ITERATIONS = 10000
from typing import Optional
import typer
from rich.panel import Panel
from rich.markdown import Markdown
from ingestforge.cli.core import IngestForgeCommand


class ShellCommand(IngestForgeCommand):
    """Full-featured interactive REPL shell."""

    def execute(
        self,
        project: Optional[Path] = None,
        history: bool = True,
    ) -> int:
        """Start interactive shell."""
        try:
            ctx = self.initialize_context(project, require_storage=True)
            llm_client = self.get_llm_client(ctx)

            if not llm_client:
                return 1

            # Display welcome message
            self._display_welcome()

            # Start REPL loop
            conversation_history = []
            return self._run_shell_loop(ctx, llm_client, conversation_history, history)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Shell session ended[/yellow]")
            return 0
        except Exception as e:
            return self.handle_error(e, "Interactive shell failed")

    def _display_welcome(self) -> None:
        """Display welcome message."""
        welcome = """# IngestForge Interactive Shell

Commands:
- Type your question to search the knowledge base
- `!help` - Show this help message
- `!clear` - Clear conversation history
- `!status` - Show current status
- `!quit` or `!exit` - Exit shell

Press Ctrl+C to exit at any time."""

        self.console.print()
        panel = Panel(Markdown(welcome), title="Welcome", border_style="cyan")
        self.console.print(panel)
        self.console.print()

    def _run_shell_loop(
        self,
        ctx: dict,
        llm_client: any,
        history: list,
        save_history: bool,
    ) -> int:
        """Run main shell loop (Rule #2: bounded)."""
        for _ in range(MAX_SHELL_ITERATIONS):
            try:
                should_continue = self._process_shell_iteration(
                    ctx, llm_client, history, save_history
                )
                if not should_continue:
                    break
            except KeyboardInterrupt:
                self.console.print()
                break
        else:
            raise AssertionError(
                f"Shell loop exceeded {MAX_SHELL_ITERATIONS} iterations"
            )

        return 0

    def _process_shell_iteration(
        self, ctx: dict, llm_client: any, history: list, save_history: bool
    ) -> bool:
        """Process single shell iteration.

        Returns:
            True to continue, False to exit
        """
        # Get user input
        user_input = self._get_user_input()

        if not user_input:
            return True

        # Handle shell commands
        if user_input.startswith("!"):
            return self._handle_shell_command(user_input, history)

        # Process query
        result = self._process_query(ctx, llm_client, user_input)

        # Display result
        self._display_result(result)

        # Add to history
        if save_history:
            history.append({"query": user_input, "answer": result})

        return True

    def _get_user_input(self) -> str:
        """Get user input."""
        return input("\n[You] > ").strip()

    def _handle_shell_command(self, command: str, history: list) -> bool:
        """Handle shell commands. Returns True to continue, False to exit."""
        cmd = command.lower()

        if cmd in ("!quit", "!exit"):
            return False

        if cmd == "!help":
            self._display_welcome()
        elif cmd == "!clear":
            history.clear()
            self.print_success("Conversation history cleared")
        elif cmd == "!status":
            self._display_status(history)
        else:
            self.print_warning(f"Unknown command: {command}")

        return True

    def _display_status(self, history: list) -> None:
        """Display current status."""
        self.console.print()
        self.console.print("[bold cyan]Status:[/bold cyan]")
        self.console.print(f"  • Conversation turns: {len(history)}")
        self.console.print()

    def _process_query(self, ctx: dict, llm_client: any, query: str) -> str:
        """Process user query."""
        # Search for relevant chunks
        chunks = self.search_context(ctx["storage"], query, k=5)

        # Build context
        context = "\n".join([getattr(c, "text", str(c))[:500] for c in chunks[:3]])

        # Generate answer
        prompt = f"""Answer this question based on the context:

Question: {query}

Context:
{context if context else "No relevant context found."}

Provide a clear, concise answer."""

        return self.generate_with_llm(llm_client, prompt, "answer")

    def _display_result(self, result: str) -> None:
        """Display query result."""
        self.console.print()
        self.console.print("[bold green][Assistant][/bold green]")
        md = Markdown(result)
        self.console.print(md)


def command(
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    history: bool = typer.Option(
        True, "--history/--no-history", help="Save conversation history"
    ),
) -> None:
    """Start interactive shell with command support.

    The shell provides a full-featured REPL with:
    - Natural language queries
    - Conversation history
    - Built-in commands (!help, !clear, !status)
    - Context-aware responses

    Examples:
        ingestforge interactive shell
        ingestforge interactive shell --no-history
    """
    cmd = ShellCommand()
    exit_code = cmd.execute(project, history)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
