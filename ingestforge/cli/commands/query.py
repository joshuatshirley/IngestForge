"""Query command - Search the knowledge base and generate answers.

Searches indexed documents and generates AI-powered answers with citations.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Any
import typer

from ingestforge.cli.core import (
    IngestForgeCommand,
    ProgressManager,
)


class QueryCommand(IngestForgeCommand):
    """Search knowledge base and generate answers."""

    def execute(
        self,
        question: str,
        project: Optional[Path] = None,
        k: int = 5,
        no_llm: bool = False,
        hybrid: bool = True,
        library: Optional[str] = None,
        unread_only: bool = False,
        tag: Optional[str] = None,
        clarify: bool = False,
    ) -> int:
        """Execute knowledge base query.

        Args:
            question: Question to answer
            project: Project directory
            k: Number of results to retrieve (1-100)
            no_llm: Skip LLM answer generation (just show results)
            hybrid: Use hybrid search (semantic + keyword) for better retrieval
            library: Filter search to specific library
            unread_only: If True, filter out chunks marked as read (ORG-001)
            tag: If provided, filter to chunks with this tag (ORG-002)
            clarify: If True, check query clarity before search ()

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs (Commandment #7: Check parameters)
            self.validate_non_empty_string(question, "question")
            self.validate_k_value(k)

            # Initialize with storage
            ctx = self.initialize_context(project, require_storage=True)

            # Query clarification if requested
            if clarify:
                question = self._clarify_query(question)

            # Show active filters
            if library:
                self.print_info(f"Searching library: {library}")
            if unread_only:
                self.print_info("Filtering to unread chunks only")
            if tag:
                self.print_info(f"Filtering by tag: {tag}")

            # Search with progress indicator
            if hybrid:
                results = self._search_hybrid(
                    ctx, question, k, library, unread_only, tag
                )
            else:
                results = self._search_knowledge_base(
                    ctx["storage"], question, k, library, unread_only, tag
                )

            # Display results
            if not results:
                self._handle_no_results(question)
                return 0

            self._display_search_results(results, k)

            # Generate answer if LLM available
            if not no_llm:
                self._generate_and_display_answer(question, results, ctx)

            return 0

        except Exception as e:
            return self.handle_error(e, "Query failed")

    def _search_hybrid(
        self,
        ctx: dict,
        question: str,
        k: int,
        library: Optional[str] = None,
        unread_only: bool = False,
        tag: Optional[str] = None,
    ) -> List[Any]:
        """Search using hybrid retrieval (semantic + BM25).

        Args:
            ctx: Context with config and storage
            question: User question
            k: Number of results
            library: Filter to specific library
            unread_only: If True, filter out chunks marked as read
            tag: If provided, filter to chunks with this tag (ORG-002)

        Returns:
            List of search results
        """
        try:
            from ingestforge.retrieval.hybrid import HybridRetriever

            retriever = HybridRetriever(
                config=ctx["config"],
                storage=ctx["storage"],
            )

            # Get more results if filtering, then filter and trim
            fetch_k = k * 3 if (unread_only or tag) else k

            results = ProgressManager.run_with_spinner(
                lambda: retriever.search(
                    question, top_k=fetch_k, library_filter=library, tag_filter=tag
                ),
                f"Hybrid search (semantic + keyword, k={k})...",
                f"Found {k} relevant chunks",
            )

            # Apply unread filter if requested
            if unread_only:
                results = self._filter_unread(results, ctx["storage"])[:k]

            return results
        except Exception as e:
            # Fall back to semantic-only search
            self.print_warning(f"Hybrid search failed, using semantic: {e}")
            return self._search_knowledge_base(
                ctx["storage"], question, k, library, unread_only, tag
            )

    def _search_knowledge_base(
        self,
        storage: Any,
        question: str,
        k: int,
        library: Optional[str] = None,
        unread_only: bool = False,
        tag: Optional[str] = None,
    ) -> List[Any]:
        """Search knowledge base for relevant chunks.

        Uses query expansion to improve retrieval quality.

        Args:
            storage: ChunkRepository instance
            question: User question
            k: Number of results
            library: Filter to specific library
            unread_only: If True, filter out chunks marked as read
            tag: If provided, filter to chunks with this tag (ORG-002)

        Returns:
            List of retrieved chunks
        """
        # Expand query for better retrieval
        expanded_query = self._expand_query(question)

        # Get more results if filtering, then filter and trim
        fetch_k = k * 3 if (unread_only or tag) else k

        results = ProgressManager.run_with_spinner(
            lambda: storage.search(
                expanded_query, k=fetch_k, library_filter=library, tag_filter=tag
            ),
            f"Searching knowledge base (k={k})...",
            f"Found {k} relevant chunks",
        )

        # Apply unread filter if requested
        if unread_only:
            results = self._filter_unread(results, storage)[:k]

        return results

    def _filter_unread(self, results: List[Any], storage: Any) -> List[Any]:
        """Filter results to only include unread chunks.

        Args:
            results: Search results to filter
            storage: ChunkRepository for checking read status

        Returns:
            Filtered results containing only unread chunks
        """
        filtered = []
        for result in results:
            chunk_id = getattr(result, "chunk_id", None)
            if chunk_id is None:
                continue

            # Check if chunk is marked as read
            chunk = storage.get_chunk(chunk_id)
            if chunk and not getattr(chunk, "is_read", False):
                filtered.append(result)

        return filtered

    def _expand_query(self, question: str) -> str:
        """Expand query with key terms for better retrieval.

        Extracts nouns and key phrases to improve semantic matching.

        Rule #2: Bounded iteration over words.

        Args:
            question: Original user question

        Returns:
            Expanded query string
        """
        # Extract key terms (simple approach - remove stop words, keep nouns)
        stop_words = {
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "can",
            "could",
            "would",
            "should",
            "will",
            "does",
            "do",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "the",
            "a",
            "an",
            "for",
            "to",
            "of",
            "in",
            "on",
            "at",
            "and",
            "or",
            "but",
            "if",
            "then",
            "so",
            "as",
            "by",
            "with",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "up",
            "down",
            "out",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "they",
            "their",
            "them",
            "used",
            "use",
            "using",
        }

        # Tokenize and filter
        words = question.lower().replace("?", "").replace(",", "").split()

        # JPL Rule #2: Bound words list BEFORE iteration (max 200 words)
        bounded_words = words[:200]

        key_terms = [w for w in bounded_words if w not in stop_words and len(w) > 2]

        # Build expanded query: original question + key terms
        if key_terms:
            return f"{question} {' '.join(key_terms)}"

        return question

    def _handle_no_results(self, question: str) -> None:
        """Handle case where no results found.

        Args:
            question: User question
        """
        self.print_warning("No relevant documents found")
        self.print_info(
            "Try:\n"
            "  - Ingesting more documents\n"
            "  - Rephrasing your question\n"
            "  - Using broader search terms"
        )

    def _display_search_results(self, results: List[Any], k: int) -> None:
        """Display search results with scores and sources.

        Args:
            results: Retrieved chunks
            k: Number of results requested
        """
        self.console.print(
            f"\n[bold cyan]Top {min(len(results), k)} Results:[/bold cyan]\n"
        )

        # Display each result (Commandment #2: Bounded loop)
        for i, result in enumerate(results[:k], 1):
            self._display_single_result(i, result)

    def _display_single_result(self, index: int, result: Any) -> None:
        """Display a single search result.

        Args:
            index: Result number (1-based)
            result: Chunk object
        """
        # Extract metadata safely (Commandment #5: Defensive)
        score = getattr(result, "score", 0.0)
        source = getattr(result, "source_file", getattr(result, "source", "Unknown"))
        # Try content first, then text, then fallback
        text = getattr(result, "content", getattr(result, "text", ""))
        if not text:
            text = str(result)

        # Extract author attribution (TICKET-301)
        author_name = self._extract_author_name(result)

        # Clean text for display (remove problematic characters)
        display_text = text[:200].encode("ascii", errors="replace").decode("ascii")

        # Format and display
        self.console.print(
            f"[bold]{index}. [cyan]{source}[/cyan] (score: {score:.3f})[/bold]"
        )

        # Display author attribution if present (TICKET-301)
        if author_name:
            self.console.print(f"   [dim]Contributed by: {author_name}[/dim]")

        self.console.print(f"   {display_text}...")
        self.console.print()

    def _extract_author_name(self, result: Any) -> Optional[str]:
        """Extract author name from result for attribution display (TICKET-301).

        Tries multiple sources for author information:
        1. Direct author_name attribute on result
        2. Nested metadata.author_name
        3. ContributorIdentity if present

        Args:
            result: Search result or chunk object

        Returns:
            Author name string if present, None otherwise

        Rule #1: Early returns for simple control flow
        Rule #7: Defensive attribute access
        """
        # Try direct attribute first
        author_name = getattr(result, "author_name", None)
        if author_name:
            return author_name

        # Try nested in metadata dict
        metadata = getattr(result, "metadata", None)
        if metadata and isinstance(metadata, dict):
            author_name = metadata.get("author_name")
            if author_name:
                return author_name

        return None

    def _generate_and_display_answer(
        self, question: str, results: List[Any], ctx: dict[str, Any]
    ) -> None:
        """Generate and display LLM answer with citations.

        Args:
            question: User question
            results: Retrieved chunks
            ctx: Context dict with config
        """
        try:
            # Try to get LLM client
            llm_client = self._get_llm_client(ctx["config"])

            if llm_client is None:
                self.print_warning(
                    "No LLM available - install a provider:\n"
                    "  pip install anthropic  # For Claude\n"
                    "  pip install openai     # For OpenAI"
                )
                return

            # Generate answer with progress
            answer = self._generate_answer(llm_client, question, results)

            # Display answer
            self._display_answer(answer)

        except Exception as e:
            self.print_warning(f"Answer generation failed: {e}")

    def _get_llm_client(self, config: Any) -> Optional[Any]:
        """Get LLM client from config.

        Args:
            config: Project configuration

        Returns:
            LLM client or None if unavailable
        """
        try:
            from ingestforge.llm.factory import get_best_available_client

            return get_best_available_client(config)
        except Exception:
            return None

    def _generate_answer(
        self, llm_client: Any, question: str, results: List[Any]
    ) -> str:
        """Generate answer using LLM.

        Args:
            llm_client: LLM provider instance
            question: User question
            results: Retrieved chunks

        Returns:
            Generated answer text
        """
        # Build context from results (Commandment #4: Small function)
        # Pass question for keyword-based result reordering
        context = self._build_context(results, question=question)

        # System prompt for RAG
        system_prompt = (
            "You are a helpful research assistant. Answer questions based on the "
            "provided context. Cite sources using [number] notation. Be concise."
        )

        # Use chat API if available (better for instruction-tuned models)
        if hasattr(llm_client, "generate_with_context"):
            return ProgressManager.run_with_spinner(
                lambda: llm_client.generate_with_context(
                    system_prompt=system_prompt,
                    user_prompt=question,
                    context=context,
                ),
                "Generating answer...",
                "Answer generated!",
            )

        # Fallback to basic generate for providers without chat API
        prompt = self._build_prompt(question, context)
        return ProgressManager.run_with_spinner(
            lambda: llm_client.generate(prompt),
            "Generating answer...",
            "Answer generated!",
        )

    def _build_context(
        self, results: List[Any], max_chars: int = 4000, question: str = ""
    ) -> str:
        """Build context string from search results with length limit.

        Reorders results to prioritize those containing question keywords.

        Rule #2: All loops bounded by MAX constants.

        Args:
            results: Retrieved chunks
            max_chars: Maximum context length in characters
            question: Original question for keyword matching

        Returns:
            Formatted context string (truncated if needed)
        """
        # Extract keywords from question for prioritization
        question_words_list = question.lower().split()

        # JPL Rule #2: Bound question words (max 200 words)
        bounded_question_words = question_words_list[:200]
        question_words = set(bounded_question_words)

        stop_words = {
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "can",
            "is",
            "are",
            "the",
            "a",
            "an",
            "for",
            "to",
            "of",
            "be",
            "used",
        }
        keywords = question_words - stop_words

        # JPL Rule #2: Bound results BEFORE iteration (max 100 results)
        bounded_results = results[:100]

        # Score and reorder results by keyword relevance
        scored_results = []
        for i, result in enumerate(bounded_results):
            text = getattr(result, "content", getattr(result, "text", str(result)))
            text_lower = text.lower()

            # Count keyword matches
            keyword_score = sum(1 for kw in keywords if kw in text_lower)
            scored_results.append((keyword_score, i, result))

        # Sort by keyword score (descending), then original order
        scored_results.sort(key=lambda x: (-x[0], x[1]))

        context_parts = []
        current_length = 0

        # JPL Rule #2: Loop is now bounded (scored_results derived from bounded_results)
        for _, orig_idx, result in scored_results:
            text = getattr(result, "content", getattr(result, "text", str(result)))
            source = getattr(result, "source", f"Source {orig_idx + 1}")
            part = f"[{orig_idx + 1}] {source}:\n{text}"

            # Check if adding this part would exceed limit
            if current_length + len(part) > max_chars:
                # Truncate this part to fit remaining space
                remaining = max_chars - current_length - 50  # Leave room for ellipsis
                if remaining > 100:
                    part = part[:remaining] + "..."
                    context_parts.append(part)
                break

            context_parts.append(part)
            current_length += len(part) + 2  # +2 for separator

        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build LLM prompt with question and context.

        Args:
            question: User question
            context: Context from search results

        Returns:
            Formatted prompt
        """
        return (
            f"Answer the following question based on the provided context. "
            f"Cite sources using [number] notation.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:"
        )

    def _display_answer(self, answer: str) -> None:
        """Display generated answer in formatted panel.

        Args:
            answer: Generated answer text
        """
        self.console.print()
        self.console.print("[bold green]=== Answer ===[/bold green]")
        self.console.print()
        self.console.print(answer)
        self.console.print()
        self.console.print("[green]" + "=" * 50 + "[/green]")

    def _clarify_query(self, query: str) -> str:
        """
        Clarify query if ambiguous.

        Interactive clarification workflow.
        Rule #4: Under 60 lines.

        Args:
            query: Original user query.

        Returns:
            Refined query (or original if clear/skipped).
        """
        from ingestforge.query.clarifier import create_clarifier
        from rich.prompt import Prompt, Confirm

        try:
            # Create clarifier
            clarifier = create_clarifier(threshold=0.7, use_llm=False)

            # Evaluate query
            artifact = clarifier.evaluate(query)

            # If clear, return as-is
            if not artifact.needs_clarification:
                self.print_success(
                    f"Query is clear (score: {artifact.clarity_score.score:.2f})"
                )
                return query

            # Show clarity score and reason
            self.console.print(
                f"\n[yellow]⚠️  Query clarity: {artifact.clarity_score.score:.2f}/1.0[/yellow]"
            )
            self.console.print(f"[yellow]Reason: {artifact.reason}[/yellow]\n")

            # Show ambiguity report if available
            if artifact.ambiguity_report and artifact.ambiguity_report.questions:
                self.console.print("[cyan]Detected ambiguities:[/cyan]")
                for question in artifact.ambiguity_report.questions:
                    self.console.print(f"  • {question.question}")
                self.console.print()

            # Show suggestions
            if artifact.suggestions:
                self.console.print("[cyan]Suggestions to improve clarity:[/cyan]")
                for i, suggestion in enumerate(artifact.suggestions, 1):
                    self.console.print(f"  {i}. {suggestion}")
                self.console.print()

            # Ask if user wants to refine
            if not Confirm.ask(
                "[bold]Would you like to refine your query?[/bold]", default=True
            ):
                return query

            # Get refined query
            refined = Prompt.ask("[bold]Enter refined query[/bold]", default=query)

            self.print_success(f"Using refined query: {refined}")
            return refined

        except Exception as e:
            self.print_warning(f"Clarification failed: {e}. Using original query.")
            return query


# Typer command wrapper
def command(
    question: str = typer.Argument(..., help="Question to answer"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    k: int = typer.Option(10, "--k", "-k", help="Number of results (1-100)"),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Skip answer generation (just show results)"
    ),
    hybrid: bool = typer.Option(
        True, "--hybrid/--no-hybrid", help="Use hybrid search (semantic + keyword)"
    ),
    library: Optional[str] = typer.Option(
        None, "--library", "-l", help="Filter search to specific library"
    ),
    unread_only: bool = typer.Option(
        False,
        "--unread-only",
        "-u",
        help="Only return unread chunks (filter out read content)",
    ),
    tag: Optional[str] = typer.Option(
        None, "--tag", "-t", help="Filter search to chunks with this tag (ORG-002)"
    ),
    clarify: bool = typer.Option(
        False, "--clarify", "-c", help="Check query clarity before search ()"
    ),
) -> None:
    """Search knowledge base and generate AI-powered answers.

    Searches indexed documents for relevant content, then generates
    an answer with citations using an LLM.

    By default uses hybrid search (semantic + BM25 keyword matching)
    for better retrieval of exact terms and concepts.

    Examples:
        # Basic query (uses hybrid search, retrieves 10 results by default)
        ingestforge query "What is IngestForge?"

        # Retrieve fewer results for faster responses
        ingestforge query "How does chunking work?" --k 5

        # Just show search results
        ingestforge query "What are embeddings?" --no-llm

        # Search only in a specific library
        ingestforge query "citizenship requirements" --library regulations

        # Use semantic-only search
        ingestforge query "Explain concepts" --no-hybrid

        # Query specific project
        ingestforge query "Explain the architecture" --project /path/to/project

        # Show only unread content
        ingestforge query "What is RAG?" --unread-only

        # Filter by tag
        ingestforge query "machine learning basics" --tag important
    """
    cmd = QueryCommand()
    exit_code = cmd.execute(
        question, project, k, no_llm, hybrid, library, unread_only, tag, clarify
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
