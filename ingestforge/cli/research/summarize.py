"""Summarize command - Multi-agent paper summarization.

Summarizes academic papers using three specialized agents:
- Abstract Agent: Extracts main thesis and contribution
- Methodology Agent: Identifies research methods
- Critique Agent: Provides critical analysis

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any

import typer
from rich.markdown import Markdown

from ingestforge.cli.research.base import ResearchCommand
from ingestforge.cli.core import ProgressManager


class SummarizeCommand(ResearchCommand):
    """Multi-agent paper summarization command."""

    def execute(
        self,
        document_id: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        output_format: str = "markdown",
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> int:
        """Summarize a paper using multi-agent approach.

        Args:
            document_id: Document ID or chunk ID to summarize
            project: Project directory
            output: Output file path (optional)
            output_format: Output format (markdown or json)
            provider: LLM provider override
            model: Model name override

        Returns:
            0 on success, 1 on error
        """
        try:
            # Initialize context with storage
            ctx = self.initialize_context(project, require_storage=True)

            # Get document content
            document_text, title, source_chunks = self._get_document_content(
                ctx["storage"], document_id
            )

            if not document_text:
                self.print_error(f"Document not found: {document_id}")
                return 1

            # Create LLM client
            llm_client = self._get_llm_client(ctx["config"], provider, model)

            if not llm_client:
                return 1

            # Run summarization
            summary = self._run_summarization(
                llm_client, document_text, title, source_chunks
            )

            # Display or save results
            if output:
                self._save_summary(output, summary, output_format)
            else:
                self._display_summary(summary, output_format)

            return 0

        except Exception as e:
            return self.handle_error(e, "Summarization failed")

    def _get_document_content(
        self,
        storage: Any,
        document_id: str,
    ) -> tuple[str, str, list[str]]:
        """Get document content from storage.

        Args:
            storage: ChunkRepository instance
            document_id: Document or chunk ID

        Returns:
            Tuple of (document_text, title, source_chunk_ids)
        """
        return ProgressManager.run_with_spinner(
            lambda: self._retrieve_document(storage, document_id),
            "Retrieving document...",
            "Document retrieved",
        )

    def _retrieve_document(
        self,
        storage: Any,
        document_id: str,
    ) -> tuple[str, str, list[str]]:
        """Retrieve document content (internal helper).

        Args:
            storage: ChunkRepository instance
            document_id: Document or chunk ID

        Returns:
            Tuple of (document_text, title, source_chunk_ids)
        """
        # Try to get by document source
        chunks = self._get_chunks_by_source(storage, document_id)

        if not chunks:
            # Try to get single chunk by ID
            chunk = self._get_chunk_by_id(storage, document_id)
            if chunk:
                chunks = [chunk]

        if not chunks:
            return "", "", []

        # Combine chunks into document
        text_parts = []
        source_ids = []

        for chunk in chunks:
            text = self._extract_chunk_text(chunk)
            text_parts.append(text)
            source_ids.append(self._get_chunk_id(chunk))

        # Extract title from first chunk or document ID
        title = self._extract_title(chunks[0], document_id)

        return "\n\n".join(text_parts), title, source_ids

    def _get_chunks_by_source(self, storage: Any, source: str) -> list[Any]:
        """Get all chunks from a source document.

        Args:
            storage: ChunkRepository instance
            source: Source document identifier

        Returns:
            List of chunks
        """
        if hasattr(storage, "get_by_source"):
            return storage.get_by_source(source)
        elif hasattr(storage, "search"):
            # Search and filter by source
            results = storage.search(source, k=100)
            return [r for r in results if self._get_chunk_source(r) == source]
        else:
            return []

    def _get_chunk_by_id(self, storage: Any, chunk_id: str) -> Optional[Any]:
        """Get a single chunk by ID.

        Args:
            storage: ChunkRepository instance
            chunk_id: Chunk identifier

        Returns:
            Chunk or None
        """
        if hasattr(storage, "get"):
            return storage.get(chunk_id)
        elif hasattr(storage, "get_chunk"):
            return storage.get_chunk(chunk_id)
        else:
            return None

    def _extract_chunk_text(self, chunk: Any) -> str:
        """Extract text from chunk.

        Args:
            chunk: Chunk object or dict

        Returns:
            Text content
        """
        if isinstance(chunk, dict):
            return chunk.get("text", "")
        elif hasattr(chunk, "text"):
            return chunk.text
        else:
            return str(chunk)

    def _get_chunk_id(self, chunk: Any) -> str:
        """Get chunk ID.

        Args:
            chunk: Chunk object or dict

        Returns:
            Chunk ID
        """
        if isinstance(chunk, dict):
            return chunk.get("id", chunk.get("chunk_id", "unknown"))
        elif hasattr(chunk, "id"):
            return chunk.id
        elif hasattr(chunk, "chunk_id"):
            return chunk.chunk_id
        else:
            return "unknown"

    def _extract_title(self, chunk: Any, fallback: str) -> str:
        """Extract title from chunk metadata.

        Args:
            chunk: Chunk object or dict
            fallback: Fallback title if not found

        Returns:
            Title string

        Rule #1: Max 3 nesting levels.
        """
        if isinstance(chunk, dict):
            metadata = chunk.get("metadata", {})
            return metadata.get("title", metadata.get("source", fallback))

        if hasattr(chunk, "metadata"):
            return self._extract_title_from_metadata(chunk.metadata, fallback)

        return fallback

    def _extract_title_from_metadata(self, metadata: Any, fallback: str) -> str:
        """Extract title from metadata object.

        Helper for _extract_title to reduce nesting.

        Args:
            metadata: Metadata object
            fallback: Fallback title

        Returns:
            Title string
        """
        if hasattr(metadata, "title") and metadata.title:
            return metadata.title
        if hasattr(metadata, "source") and metadata.source:
            return metadata.source
        return fallback

    def _get_llm_client(
        self,
        config: Any,
        provider: Optional[str],
        model: Optional[str],
    ) -> Optional[Any]:
        """Get LLM client for summarization.

        Args:
            config: IngestForge config
            provider: Provider override
            model: Model override

        Returns:
            LLM client or None on error
        """
        try:
            from ingestforge.llm.factory import get_llm_client

            llm_client = get_llm_client(config, provider)

            # Override model if specified
            if model and hasattr(llm_client, "_model"):
                llm_client._model = model

            # Check availability
            if not llm_client.is_available():
                self._handle_missing_credentials(
                    provider or config.llm.default_provider
                )
                return None

            return llm_client

        except Exception as e:
            self.print_error(f"Failed to create LLM client: {e}")
            return None

    def _handle_missing_credentials(self, provider: str) -> None:
        """Handle missing credentials for provider.

        Args:
            provider: Provider name
        """
        cloud_providers = {
            "claude": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }

        if provider in cloud_providers:
            env_var = cloud_providers[provider]
            self.print_error(
                f"{provider} credentials not found.\n"
                f"Set {env_var} environment variable or configure in ingestforge.yaml"
            )
        else:
            self.print_error(f"{provider} not available")

    def _run_summarization(
        self,
        llm_client: Any,
        document_text: str,
        title: str,
        source_chunks: list[str],
    ) -> Any:
        """Run multi-agent summarization.

        Args:
            llm_client: LLM client instance
            document_text: Document content
            title: Paper title
            source_chunks: Source chunk IDs

        Returns:
            PaperSummary object
        """

        return ProgressManager.run_with_spinner(
            lambda: self._execute_summarization(
                llm_client, document_text, title, source_chunks
            ),
            "Analyzing paper with multi-agent summarizer...",
            "Analysis complete",
        )

    def _execute_summarization(
        self,
        llm_client: Any,
        document_text: str,
        title: str,
        source_chunks: list[str],
    ) -> Any:
        """Execute summarization (internal helper).

        Args:
            llm_client: LLM client instance
            document_text: Document content
            title: Paper title
            source_chunks: Source chunk IDs

        Returns:
            PaperSummary object
        """
        from ingestforge.agent.paper_summarizer import create_paper_summarizer

        summarizer = create_paper_summarizer(llm_client)
        return summarizer.summarize(document_text, title, source_chunks)

    def _display_summary(self, summary: Any, output_format: str) -> None:
        """Display summary to console.

        Args:
            summary: PaperSummary object
            output_format: Output format (markdown or json)
        """
        self.console.print()

        if output_format == "json":
            self._display_json_summary(summary)
        else:
            self._display_markdown_summary(summary)

    def _display_markdown_summary(self, summary: Any) -> None:
        """Display summary in markdown format.

        Args:
            summary: PaperSummary object
        """
        md_content = summary.to_markdown()
        self.console.print(Markdown(md_content))

    def _display_json_summary(self, summary: Any) -> None:
        """Display summary in JSON format.

        Args:
            summary: PaperSummary object
        """
        json_content = json.dumps(summary.to_dict(), indent=2)
        self.console.print(json_content)

    def _save_summary(
        self,
        output: Path,
        summary: Any,
        output_format: str,
    ) -> None:
        """Save summary to file.

        Args:
            output: Output file path
            summary: PaperSummary object
            output_format: Output format
        """
        try:
            if output_format == "json":
                content = json.dumps(summary.to_dict(), indent=2)
            else:
                content = summary.to_markdown()

            output.write_text(content, encoding="utf-8")
            self.print_success(f"Summary saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save summary: {e}")


# Typer command wrapper
def command(
    document_id: str = typer.Argument(
        ...,
        help="Document ID or chunk ID to summarize",
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    output_format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format (markdown or json)",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="LLM provider (ollama, claude, openai, gemini, llamacpp)",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-M", help="Model name override"
    ),
) -> None:
    """Summarize a paper using multi-agent approach.

    Uses three specialized AI agents to analyze academic papers:
    - Abstract Agent: Extracts main thesis and contribution
    - Methodology Agent: Identifies research methods
    - Critique Agent: Provides critical analysis and limitations

    The output includes:
    - Abstract & Thesis summary
    - Methodology explanation
    - Critical analysis
    - Key findings
    - Identified limitations

    Examples:
        # Summarize by document source
        ingestforge research summarize paper.pdf

        # Summarize specific chunk
        ingestforge research summarize chunk_abc123

        # Save to markdown file
        ingestforge research summarize paper.pdf -o summary.md

        # Output as JSON
        ingestforge research summarize paper.pdf -f json

        # Use specific provider
        ingestforge research summarize paper.pdf --provider claude
    """
    cmd = SummarizeCommand()
    exit_code = cmd.execute(
        document_id=document_id,
        project=project,
        output=output,
        output_format=output_format,
        provider=provider,
        model=model,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
