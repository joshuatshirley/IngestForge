"""LLM-optimized output formatters.

Copy-Paste Ready CLI Interfaces
Epic: EP-08 (Structured Data Foundry)
Feature: FE-08-04 (Copy-Paste Ready CLI Interfaces)

Provides template-based formatters for different LLM interfaces
(ChatGPT, Claude, plain text) to maximize usability when
copy-pasting context into LLM chats.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_CHUNKS, MAX_CONTEXT_LENGTH)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# JPL Rule #2: Fixed upper bounds
MAX_CHUNKS = 50  # Maximum chunks to format
MAX_CONTEXT_LENGTH = 50_000  # Maximum output length (chars)
MAX_CITATION_LENGTH = 200  # Maximum citation string length
MAX_TITLE_LENGTH = 100  # Maximum title/source length


class LLMFormat(Enum):
    """Available LLM output formats."""

    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    PLAIN = "plain"
    MARKDOWN = "markdown"


@dataclass
class ContextChunk:
    """A chunk of context with metadata.

    Attributes:
        text: The chunk content.
        source: Source document/file name.
        rank: Relevance rank (1 = most relevant).
        score: Relevance score (0.0 - 1.0).
        page: Optional page number.
        section: Optional section name.
    """

    text: str
    source: str = "Unknown"
    rank: int = 1
    score: float = 1.0
    page: Optional[int] = None
    section: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate chunk fields."""
        # JPL Rule #5: Assert preconditions
        assert self.text is not None, "text cannot be None"
        assert self.rank >= 1, "rank must be >= 1"
        assert 0.0 <= self.score <= 1.0, "score must be between 0 and 1"


@dataclass
class FormatterContext:
    """Context for LLM formatting.

    Attributes:
        query: The user's query.
        chunks: List of context chunks.
        include_citations: Whether to include citations.
        include_scores: Whether to include relevance scores.
        max_chunks: Maximum chunks to include.
    """

    query: str
    chunks: list[ContextChunk] = field(default_factory=list)
    include_citations: bool = True
    include_scores: bool = False
    max_chunks: int = MAX_CHUNKS

    def __post_init__(self) -> None:
        """Validate context fields."""
        # JPL Rule #5: Assert preconditions
        assert self.query is not None, "query cannot be None"
        assert self.max_chunks > 0, "max_chunks must be positive"
        # JPL Rule #2: Enforce bounds
        self.max_chunks = min(self.max_chunks, MAX_CHUNKS)


@dataclass
class FormattedOutput:
    """Result of formatting operation.

    Attributes:
        content: The formatted content.
        format: Format that was used.
        chunk_count: Number of chunks included.
        truncated: Whether content was truncated.
        char_count: Number of characters in output.
    """

    content: str
    format: LLMFormat
    chunk_count: int = 0
    truncated: bool = False
    char_count: int = 0

    def __post_init__(self) -> None:
        """Calculate char count if not set."""
        if self.char_count == 0:
            self.char_count = len(self.content)


class LLMFormatterBase(ABC):
    """Abstract base class for LLM formatters.

    Each formatter implements format-specific output generation
    optimized for a particular LLM interface.
    """

    @property
    @abstractmethod
    def format_type(self) -> LLMFormat:
        """Return the format type this formatter produces."""
        pass

    @abstractmethod
    def format(self, context: FormatterContext) -> FormattedOutput:
        """Format context for target LLM.

        Args:
            context: Context to format.

        Returns:
            Formatted output ready for copy-paste.
        """
        pass

    def _truncate_text(self, text: str, max_length: int) -> tuple[str, bool]:
        """Truncate text to max length.

        Args:
            text: Text to truncate.
            max_length: Maximum length.

        Returns:
            Tuple of (truncated_text, was_truncated).
        """
        if len(text) <= max_length:
            return text, False
        return text[:max_length] + "...", True

    def _format_citation(self, chunk: ContextChunk) -> str:
        """Format a citation string for a chunk.

        Args:
            chunk: Chunk to create citation for.

        Returns:
            Citation string.
        """
        parts = [chunk.source[:MAX_TITLE_LENGTH]]

        if chunk.page is not None:
            parts.append(f"p. {chunk.page}")

        if chunk.section:
            parts.append(chunk.section[:50])

        citation = ", ".join(parts)
        return citation[:MAX_CITATION_LENGTH]


class ChatGPTFormatter(LLMFormatterBase):
    """Formatter optimized for ChatGPT.

    Uses markdown with clear headers, code blocks for
    technical content, and structured sections.
    """

    @property
    def format_type(self) -> LLMFormat:
        return LLMFormat.CHATGPT

    def format(self, context: FormatterContext) -> FormattedOutput:
        """Format context for ChatGPT.

        Args:
            context: Context to format.

        Returns:
            Markdown-formatted output optimized for ChatGPT.
        """
        # JPL Rule #5: Assert preconditions
        assert context is not None, "context cannot be None"

        lines: list[str] = []
        truncated = False

        # Header
        lines.append("## Context for Your Query")
        lines.append("")
        lines.append(f"**Query:** {context.query}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Format chunks (bounded by max_chunks)
        chunk_count = 0
        for chunk in context.chunks[: context.max_chunks]:
            chunk_count += 1

            # Source header
            if context.include_citations:
                citation = self._format_citation(chunk)
                lines.append(f"### Source {chunk.rank}: {citation}")
            else:
                lines.append(f"### Context {chunk.rank}")

            if context.include_scores:
                lines.append(f"*Relevance: {chunk.score:.2f}*")

            lines.append("")
            lines.append(chunk.text)
            lines.append("")

        # Citations section
        if context.include_citations and context.chunks:
            lines.append("---")
            lines.append("")
            lines.append("### Sources")
            lines.append("")
            for i, chunk in enumerate(context.chunks[: context.max_chunks], 1):
                citation = self._format_citation(chunk)
                lines.append(f"{i}. {citation}")
            lines.append("")

        content = "\n".join(lines)

        # Truncate if needed
        content, truncated = self._truncate_text(content, MAX_CONTEXT_LENGTH)

        return FormattedOutput(
            content=content,
            format=self.format_type,
            chunk_count=chunk_count,
            truncated=truncated,
        )


class ClaudeFormatter(LLMFormatterBase):
    """Formatter optimized for Claude.

    Uses clean formatting with XML-style tags for clear
    structure and minimal markdown for readability.
    """

    @property
    def format_type(self) -> LLMFormat:
        return LLMFormat.CLAUDE

    def format(self, context: FormatterContext) -> FormattedOutput:
        """Format context for Claude.

        Args:
            context: Context to format.

        Returns:
            Clean formatted output optimized for Claude.
        """
        # JPL Rule #5: Assert preconditions
        assert context is not None, "context cannot be None"

        lines: list[str] = []
        truncated = False

        # Header with XML-style structure
        lines.append("<context>")
        lines.append(f"<query>{context.query}</query>")
        lines.append("")

        # Format chunks
        chunk_count = 0
        for chunk in context.chunks[: context.max_chunks]:
            chunk_count += 1

            if context.include_citations:
                citation = self._format_citation(chunk)
                lines.append(f'<source rank="{chunk.rank}" ref="{citation}">')
            else:
                lines.append(f'<source rank="{chunk.rank}">')

            lines.append(chunk.text)
            lines.append("</source>")
            lines.append("")

        lines.append("</context>")

        # Citations as a separate section
        if context.include_citations and context.chunks:
            lines.append("")
            lines.append("<references>")
            for i, chunk in enumerate(context.chunks[: context.max_chunks], 1):
                citation = self._format_citation(chunk)
                lines.append(f"[{i}] {citation}")
            lines.append("</references>")

        content = "\n".join(lines)

        # Truncate if needed
        content, truncated = self._truncate_text(content, MAX_CONTEXT_LENGTH)

        return FormattedOutput(
            content=content,
            format=self.format_type,
            chunk_count=chunk_count,
            truncated=truncated,
        )


class PlainTextFormatter(LLMFormatterBase):
    """Formatter for plain text output.

    Minimal formatting, suitable for any text-based interface.
    """

    @property
    def format_type(self) -> LLMFormat:
        return LLMFormat.PLAIN

    def format(self, context: FormatterContext) -> FormattedOutput:
        """Format context as plain text.

        Args:
            context: Context to format.

        Returns:
            Plain text output.
        """
        # JPL Rule #5: Assert preconditions
        assert context is not None, "context cannot be None"

        lines: list[str] = []
        truncated = False

        lines.append(f"Query: {context.query}")
        lines.append("")
        lines.append("=" * 40)
        lines.append("")

        # Format chunks
        chunk_count = 0
        for chunk in context.chunks[: context.max_chunks]:
            chunk_count += 1

            if context.include_citations:
                citation = self._format_citation(chunk)
                lines.append(f"[{chunk.rank}] {citation}")
            else:
                lines.append(f"[{chunk.rank}]")

            lines.append("")
            lines.append(chunk.text)
            lines.append("")
            lines.append("-" * 40)
            lines.append("")

        content = "\n".join(lines)

        # Truncate if needed
        content, truncated = self._truncate_text(content, MAX_CONTEXT_LENGTH)

        return FormattedOutput(
            content=content,
            format=self.format_type,
            chunk_count=chunk_count,
            truncated=truncated,
        )


class MarkdownFormatter(LLMFormatterBase):
    """Formatter for generic markdown output.

    Standard markdown compatible with most platforms.
    """

    @property
    def format_type(self) -> LLMFormat:
        return LLMFormat.MARKDOWN

    def format(self, context: FormatterContext) -> FormattedOutput:
        """Format context as markdown.

        Args:
            context: Context to format.

        Returns:
            Markdown formatted output.
        """
        # JPL Rule #5: Assert preconditions
        assert context is not None, "context cannot be None"

        lines: list[str] = []
        truncated = False

        lines.append("# Retrieved Context")
        lines.append("")
        lines.append(f"> **Query:** {context.query}")
        lines.append("")

        # Format chunks
        chunk_count = 0
        for chunk in context.chunks[: context.max_chunks]:
            chunk_count += 1

            if context.include_citations:
                citation = self._format_citation(chunk)
                lines.append(f"## {chunk.rank}. {citation}")
            else:
                lines.append(f"## Context {chunk.rank}")

            lines.append("")
            lines.append(chunk.text)
            lines.append("")

        # References section
        if context.include_citations and context.chunks:
            lines.append("## References")
            lines.append("")
            for i, chunk in enumerate(context.chunks[: context.max_chunks], 1):
                citation = self._format_citation(chunk)
                lines.append(f"- [{i}] {citation}")
            lines.append("")

        content = "\n".join(lines)

        # Truncate if needed
        content, truncated = self._truncate_text(content, MAX_CONTEXT_LENGTH)

        return FormattedOutput(
            content=content,
            format=self.format_type,
            chunk_count=chunk_count,
            truncated=truncated,
        )


# Formatter registry
_FORMATTERS: dict[LLMFormat, type[LLMFormatterBase]] = {
    LLMFormat.CHATGPT: ChatGPTFormatter,
    LLMFormat.CLAUDE: ClaudeFormatter,
    LLMFormat.PLAIN: PlainTextFormatter,
    LLMFormat.MARKDOWN: MarkdownFormatter,
}


def get_formatter(format_type: LLMFormat) -> LLMFormatterBase:
    """Get formatter instance for format type.

    Args:
        format_type: Desired output format.

    Returns:
        Formatter instance.

    Raises:
        ValueError: If format type is not supported.
    """
    formatter_class = _FORMATTERS.get(format_type)
    if formatter_class is None:
        raise ValueError(f"Unsupported format: {format_type}")
    return formatter_class()


def format_context(
    query: str,
    chunks: list[dict[str, Any]],
    format_type: LLMFormat = LLMFormat.MARKDOWN,
    include_citations: bool = True,
) -> FormattedOutput:
    """Convenience function to format context.

    Args:
        query: User query.
        chunks: List of chunk dictionaries.
        format_type: Output format.
        include_citations: Whether to include citations.

    Returns:
        Formatted output.
    """
    # Convert dict chunks to ContextChunk objects
    context_chunks: list[ContextChunk] = []
    for i, chunk_data in enumerate(chunks[:MAX_CHUNKS], 1):
        context_chunks.append(
            ContextChunk(
                text=chunk_data.get("text", str(chunk_data)),
                source=chunk_data.get("source", "Unknown"),
                rank=i,
                score=chunk_data.get("score", 1.0),
                page=chunk_data.get("page"),
                section=chunk_data.get("section"),
            )
        )

    context = FormatterContext(
        query=query,
        chunks=context_chunks,
        include_citations=include_citations,
    )

    formatter = get_formatter(format_type)
    return formatter.format(context)
