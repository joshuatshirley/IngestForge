"""Ask command - Interactive REPL for conversational queries.

Provides an interactive read-eval-print loop for querying the knowledge base
with conversation history and context awareness.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.panel import Panel
from rich.markdown import Markdown

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class AskCommand(IngestForgeCommand):
    """Interactive REPL for conversational queries."""

    def __init__(self) -> None:
        """Initialize ask command."""
        super().__init__()
        self.conversation_history: List[Dict[str, str]] = []

    def execute(
        self,
        project: Optional[Path] = None,
        k: int = 5,
        no_history: bool = False,
    ) -> int:
        """Start interactive query session.

        Args:
            project: Project directory
            k: Number of chunks to retrieve per query
            no_history: Disable conversation history

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate parameters (Commandment #7)
            self.validate_k_value(k)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Get LLM client
            llm_client = self._get_llm_client(ctx)
            if llm_client is None:
                return 1

            # Display welcome message
            self._display_welcome()

            # Run interactive loop
            return self._run_interactive_loop(ctx["storage"], llm_client, k, no_history)

        except Exception as e:
            return self.handle_error(e, "Interactive session failed")

    def validate_k_value(self, k: int) -> None:
        """Validate k parameter.

        Args:
            k: Number of chunks to retrieve

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if k < 1:
            raise typer.BadParameter("k must be at least 1")

        if k > 50:
            raise typer.BadParameter("k cannot exceed 50")

    def _get_llm_client(self, ctx: dict) -> Optional[Any]:
        """Get LLM client for generation.

        Args:
            ctx: Context dictionary

        Returns:
            LLM client or None
        """
        try:
            from ingestforge.llm.factory import get_best_available_client

            client = get_best_available_client(ctx["config"])

            if client is None:
                self.print_warning(
                    "No LLM available. Install a provider:\n"
                    "  pip install anthropic  # For Claude\n"
                    "  pip install openai     # For OpenAI"
                )

            return client

        except Exception as e:
            self.print_warning(f"Failed to load LLM client: {e}")
            return None

    def _display_welcome(self) -> None:
        """Display welcome message."""
        welcome = Panel(
            "**Interactive Knowledge Base Query**\n\n"
            "Ask questions about your ingested documents.\n"
            "Conversation history is maintained for context.\n\n"
            "Commands:\n"
            "  - Type your question and press Enter\n"
            "  - `/history` - View conversation history\n"
            "  - `/clear` - Clear conversation history\n"
            "  - `/exit` or `/quit` - Exit interactive mode\n"
            "  - `Ctrl+C` - Exit",
            title="[bold cyan]IngestForge Interactive Mode[/bold cyan]",
            border_style="cyan",
        )

        self.console.print()
        self.console.print(welcome)
        self.console.print()

    def _get_user_input(self, history_obj: Any) -> str:
        """
        Get user input from prompt.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            history_obj: InMemoryHistory object

        Returns:
            User input string (stripped)
        """
        from prompt_toolkit import prompt as pt_prompt

        assert history_obj is not None, "History object cannot be None"

        question = pt_prompt("You: ", history=history_obj).strip()
        return question

    def _process_user_input(
        self,
        question: str,
        storage: Any,
        llm_client: Optional[Any],
        k: int,
        no_history: bool,
    ) -> bool:
        """
        Process user input (command or question).

        Rule #1: Early returns eliminate nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            question: User input
            storage: Storage instance
            llm_client: LLM client
            k: Number of chunks
            no_history: Disable history flag

        Returns:
            True if should exit, False otherwise
        """
        assert question is not None, "Question cannot be None"
        if not question:
            return False
        if question.startswith("/"):
            return self._handle_command(question)
        self._process_question(question, storage, llm_client, k, no_history)
        return False

    def _initialize_interactive_session(
        self, storage: Any, llm_client: Any, k: int
    ) -> tuple[int, Any]:
        """Initialize interactive session with validation.

        Rule #4: No large functions - Extracted from _run_interactive_loop
        Rule #7: Parameter validation

        Returns:
            Tuple of (max_iterations, history_object)
        """
        from prompt_toolkit.history import InMemoryHistory

        assert storage is not None, "Storage cannot be None"
        assert llm_client is not None, "LLM client cannot be None"
        assert k > 0, "k must be positive"
        MAX_ITERATIONS: int = 10_000  # Hard limit
        history_obj = InMemoryHistory()

        return MAX_ITERATIONS, history_obj

    def _execute_loop_iteration(
        self, history_obj: Any, storage: Any, llm_client: Any, k: int, no_history: bool
    ) -> bool:
        """Execute one loop iteration.

        Rule #4: No large functions - Extracted from _run_interactive_loop

        Returns:
            True if should exit, False otherwise
        """
        try:
            question = self._get_user_input(history_obj)
            should_exit = self._process_user_input(
                question, storage, llm_client, k, no_history
            )

            return should_exit

        except KeyboardInterrupt:
            self.print_info("\nExiting interactive mode...")
            return True

        except EOFError:
            return True

        except Exception as e:
            self.print_error(f"Error: {e}")
            return False

    def _run_interactive_loop(
        self, storage: Any, llm_client: Any, k: int, no_history: bool
    ) -> int:
        """
        Run the interactive query loop.

        Rule #1: Zero nesting - helpers extract all nested logic
        Rule #2: Fixed loop bound (max iterations for safety)
        Rule #4: Function <60 lines (refactored to 28 lines)
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            storage: Storage instance
            llm_client: LLM client
            k: Number of chunks to retrieve
            no_history: Whether to disable history

        Returns:
            Exit code (always 0)
        """
        # Initialize session with validation
        MAX_ITERATIONS, history_obj = self._initialize_interactive_session(
            storage, llm_client, k
        )

        iterations: int = 0
        while True:
            iterations += 1
            if iterations > MAX_ITERATIONS:
                self.print_warning(f"Safety limit: {MAX_ITERATIONS} iterations reached")
                break

            # Execute one iteration
            should_exit = self._execute_loop_iteration(
                history_obj, storage, llm_client, k, no_history
            )

            if should_exit:
                break

        return 0

    def _handle_exit_command(self) -> bool:
        """
        Handle exit command.

        Rule #1: Extracted helper reduces complexity
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Returns:
            True (always exits)
        """
        self.print_info("Goodbye!")
        return True

    def _handle_unknown_command(self, command: str) -> bool:
        """
        Handle unknown command.

        Rule #1: Extracted helper reduces complexity
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            command: Unknown command string

        Returns:
            False (don't exit)
        """
        assert command is not None, "Command cannot be None"

        self.print_warning(f"Unknown command: {command}")
        self.print_info("Type /help for available commands")
        return False

    def _handle_command(self, command: str) -> bool:
        """
        Handle special commands.

        Rule #1: Dictionary dispatch eliminates if/elif chain
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            command: Command string (e.g., "/help", "/exit")

        Returns:
            True if should exit, False otherwise
        """
        assert command is not None, "Command cannot be None"
        assert isinstance(command, str), "Command must be string"
        assert command.startswith("/"), "Command must start with /"
        command_lower = command.lower()
        exit_commands = {"/exit", "/quit", "/q"}
        if command_lower in exit_commands:
            return self._handle_exit_command()
        command_handlers = {
            "/history": self._display_history,
            "/clear": self._clear_history,
            "/help": self._display_help,
            "/?": self._display_help,
        }
        handler = command_handlers.get(command_lower)
        if handler is None:
            return self._handle_unknown_command(command)

        # Execute handler (void return, so return False)
        handler()
        return False

    def _display_history(self) -> None:
        """Display conversation history."""
        if not self.conversation_history:
            self.print_info("No conversation history yet")
            return

        self.console.print()
        self.console.print("[bold cyan]Conversation History[/bold cyan]")
        self.console.print()

        for idx, entry in enumerate(self.conversation_history, 1):
            self.console.print(f"[yellow]Q{idx}:[/yellow] {entry['question']}")
            self.console.print(f"[green]A{idx}:[/green] {entry['answer'][:100]}...")
            self.console.print()

    def _clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.print_success("Conversation history cleared")

    def _display_help(self) -> None:
        """Display help message."""
        help_text = Panel(
            "**Available Commands:**\n\n"
            "`/history` - View conversation history\n"
            "`/clear` - Clear conversation history\n"
            "`/help` - Show this help message\n"
            "`/exit` or `/quit` - Exit interactive mode\n\n"
            "Just type your question to query the knowledge base.",
            title="[bold cyan]Help[/bold cyan]",
            border_style="cyan",
        )

        self.console.print()
        self.console.print(help_text)
        self.console.print()

    def _process_question(
        self,
        question: str,
        storage: Any,
        llm_client: Any,
        k: int,
        no_history: bool,
    ) -> None:
        """Process a question and generate answer.

        Args:
            question: User question
            storage: Storage instance
            llm_client: LLM client
            k: Number of chunks to retrieve
            no_history: Whether to disable history
        """
        # Search for relevant chunks
        chunks = ProgressManager.run_with_spinner(
            lambda: storage.search(question, k=k),
            "Searching knowledge base...",
            "",
        )

        if not chunks:
            self.print_warning("No relevant information found")
            return

        # Generate answer
        answer = self._generate_answer(question, chunks, llm_client, no_history)

        # Display answer
        self._display_answer(answer)

        # Update history
        if not no_history:
            self.conversation_history.append({"question": question, "answer": answer})

    def _generate_answer(
        self,
        question: str,
        chunks: list,
        llm_client: Any,
        no_history: bool,
    ) -> str:
        """Generate answer using LLM.

        Args:
            question: User question
            chunks: Retrieved chunks
            llm_client: LLM client
            no_history: Whether to disable history

        Returns:
            Generated answer
        """
        # Build context
        context = self._format_context(chunks)

        # Build prompt with optional history
        prompt = self._build_prompt(question, context, no_history)

        # Generate answer
        return ProgressManager.run_with_spinner(
            lambda: self._generate_with_llm(llm_client, prompt),
            "Generating answer...",
            "",
        )

    def _format_context(self, chunks: list) -> str:
        """Format chunks as context.

        Args:
            chunks: List of chunks

        Returns:
            Formatted context string
        """
        context_parts = []

        for idx, chunk in enumerate(chunks, 1):
            # Extract text from chunk
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
            elif hasattr(chunk, "text"):
                text = chunk.text
            else:
                text = str(chunk)

            context_parts.append(f"[{idx}] {text}")

        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str, no_history: bool) -> str:
        """Build prompt for LLM.

        Args:
            question: User question
            context: Context from chunks
            no_history: Whether to disable history

        Returns:
            Formatted prompt
        """
        prompt_parts = []

        # Add conversation history if enabled
        if not no_history and self.conversation_history:
            prompt_parts.append("Previous conversation:\n")
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                prompt_parts.append(f"Q: {entry['question']}\n")
                prompt_parts.append(f"A: {entry['answer'][:200]}...\n\n")

        # Add context and question
        prompt_parts.append("Context from knowledge base:\n")
        prompt_parts.append(context)
        prompt_parts.append("\n\n")
        prompt_parts.append(f"Question: {question}\n\n")
        prompt_parts.append(
            "Provide a clear, concise answer based on the context. "
            "If the context doesn't contain enough information, say so."
        )

        return "".join(prompt_parts)

    def _generate_with_llm(self, llm_client: Any, prompt: str) -> str:
        """Generate text with LLM.

        Args:
            llm_client: LLM client
            prompt: Prompt text

        Returns:
            Generated response
        """
        if hasattr(llm_client, "generate"):
            return llm_client.generate(prompt)
        elif hasattr(llm_client, "complete"):
            return llm_client.complete(prompt)
        elif callable(llm_client):
            return llm_client(prompt)
        else:
            raise TypeError(f"Unknown LLM client type: {type(llm_client)}")

    def _display_answer(self, answer: str) -> None:
        """Display generated answer.

        Args:
            answer: Answer text
        """
        self.console.print()

        panel = Panel(
            Markdown(answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        )

        self.console.print(panel)
        self.console.print()


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    k: int = typer.Option(
        5, "--k", "-k", help="Number of chunks to retrieve per query"
    ),
    no_history: bool = typer.Option(
        False, "--no-history", help="Disable conversation history"
    ),
) -> None:
    """Start interactive query session (REPL mode).

    Provides a conversational interface for querying your
    knowledge base with maintained conversation history.

    Features:
    - Conversational context across questions
    - Real-time search and answer generation
    - Command support (/history, /clear, etc.)
    - Easy navigation and exploration

    Examples:
        # Start interactive session
        ingestforge interactive ask

        # Disable conversation history
        ingestforge interactive ask --no-history

        # Retrieve more context per query
        ingestforge interactive ask --k 10

        # Specific project
        ingestforge interactive ask -p /path/to/project

    In the REPL:
        You: What is machine learning?
        [Answer displayed]

        You: How does it differ from AI?
        [Answer using conversation context]
    """
    cmd = AskCommand()
    exit_code = cmd.execute(project, k, no_history)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
