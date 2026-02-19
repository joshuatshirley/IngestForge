"""Symbols command - Identify and analyze symbolic patterns.

Provides symbol detection and analysis:
- detect: Find recurring symbolic elements
- analyze: Analyze specific symbol's meaning and development"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from ingestforge.cli.literary.base import LiteraryCommand
from ingestforge.cli.literary.models import Symbol, SymbolAnalysis
from ingestforge.cli.core import ProgressManager
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Common Symbol Patterns
# ============================================================================

# Common symbolic elements and their potential meanings
COMMON_SYMBOLS = {
    # Nature
    "water": ["purification", "life", "death", "rebirth", "emotion"],
    "fire": ["destruction", "passion", "transformation", "anger", "warmth"],
    "earth": ["stability", "fertility", "death", "foundation"],
    "air": ["freedom", "spirit", "thought", "change"],
    "light": ["hope", "knowledge", "goodness", "truth", "divinity"],
    "darkness": ["evil", "ignorance", "death", "mystery", "fear"],
    "storm": ["turmoil", "change", "divine wrath", "cleansing"],
    "rain": ["renewal", "sadness", "blessing", "fertility"],
    # Colors
    "white": ["purity", "innocence", "death", "emptiness"],
    "black": ["death", "evil", "mystery", "power", "elegance"],
    "red": ["passion", "blood", "anger", "love", "danger"],
    "green": ["nature", "envy", "growth", "decay"],
    "blue": ["sadness", "calm", "truth", "royalty"],
    "gold": ["wealth", "divinity", "corruption", "value"],
    # Objects
    "mirror": ["self-reflection", "vanity", "truth", "duality"],
    "door": ["opportunity", "transition", "mystery", "choice"],
    "window": ["perspective", "hope", "barrier", "vision"],
    "key": ["knowledge", "freedom", "secrets", "power"],
    "clock": ["mortality", "time passing", "urgency", "fate"],
    "chain": ["bondage", "connection", "oppression", "strength"],
    "crown": ["authority", "ambition", "burden", "divinity"],
    "mask": ["deception", "identity", "protection", "hiding"],
    "blood": ["life", "death", "sacrifice", "guilt", "family"],
    "heart": ["love", "courage", "emotion", "life"],
    "eye": ["perception", "knowledge", "surveillance", "truth"],
    "hand": ["power", "help", "creation", "violence"],
    "road": ["journey", "life path", "choice", "destiny"],
    "bridge": ["transition", "connection", "danger", "opportunity"],
    "wall": ["barrier", "protection", "division", "imprisonment"],
    "garden": ["paradise", "cultivation", "nature", "temptation"],
    "forest": ["unknown", "danger", "freedom", "nature"],
    "mountain": ["obstacle", "achievement", "spirituality"],
    "river": ["time", "journey", "boundary", "life"],
    "sea": ["unconscious", "infinity", "chaos", "freedom"],
    "bird": ["freedom", "soul", "transcendence", "omen"],
    "snake": ["evil", "temptation", "wisdom", "renewal"],
    "rose": ["love", "beauty", "secrecy", "mortality"],
    "tree": ["life", "knowledge", "family", "growth"],
    "sun": ["life", "power", "divinity", "enlightenment"],
    "moon": ["femininity", "change", "mystery", "madness"],
    "star": ["hope", "destiny", "guidance", "divinity"],
}

# ============================================================================
# Symbol Detector Class
# ============================================================================


class SymbolDetector:
    """Detect and analyze symbolic patterns in text.

    Identifies recurring symbolic elements and analyzes their meaning.
    Follows Rule #4: All methods <60 lines.
    """

    def __init__(
        self, llm_client: Optional[Any] = None, min_occurrences: int = 3
    ) -> None:
        """Initialize symbol detector.

        Args:
            llm_client: Optional LLM for enhanced analysis
            min_occurrences: Minimum occurrences to detect a symbol
        """
        self.llm_client = llm_client
        self.min_occurrences = min_occurrences
        self._symbol_patterns = COMMON_SYMBOLS

    def detect_symbols(self, chunks: List[Any]) -> List[Symbol]:
        """Detect recurring symbolic elements.

        Args:
            chunks: Text chunks to analyze

        Returns:
            List of Symbol objects

        Rule #1: Early return for empty input
        """
        if not chunks:
            return []

        # Count symbol occurrences
        symbol_counts = self._count_symbols(chunks)

        # Filter by minimum occurrences
        filtered = {
            name: count
            for name, count in symbol_counts.items()
            if count >= self.min_occurrences
        }

        # Create Symbol objects
        symbols = self._create_symbol_objects(filtered, chunks)

        # Sort by occurrence count
        return sorted(symbols, key=lambda s: -s.occurrences)

    def _count_symbols(self, chunks: List[Any]) -> Dict[str, int]:
        """Count symbol occurrences in chunks.

        Args:
            chunks: Text chunks

        Returns:
            Dict of symbol name to count
        """
        counts: Dict[str, int] = defaultdict(int)

        for chunk in chunks:
            content = getattr(chunk, "content", str(chunk)).lower()

            for symbol_name in self._symbol_patterns:
                if symbol_name in content:
                    counts[symbol_name] += content.count(symbol_name)

        return dict(counts)

    def _create_symbol_objects(
        self, counts: Dict[str, int], chunks: List[Any]
    ) -> List[Symbol]:
        """Create Symbol objects from counts.

        Args:
            counts: Symbol occurrence counts
            chunks: Original chunks for context

        Returns:
            List of Symbol objects
        """
        symbols = []

        for name, count in counts.items():
            # Find first appearance
            first_appearance = self._find_first_appearance(name, chunks)

            # Get sample contexts
            contexts = self._get_contexts(name, chunks, max_contexts=3)

            # Get associated themes from our knowledge
            associated = self._symbol_patterns.get(name, [])

            symbol = Symbol(
                name=name.title(),
                occurrences=count,
                first_appearance=first_appearance,
                contexts=contexts,
                associated_themes=associated[:5],
            )
            symbols.append(symbol)

        return symbols

    def _find_first_appearance(self, symbol: str, chunks: List[Any]) -> int:
        """Find first chunk where symbol appears.

        Args:
            symbol: Symbol name
            chunks: Text chunks

        Returns:
            Chunk index of first appearance
        """
        for i, chunk in enumerate(chunks):
            content = getattr(chunk, "content", str(chunk)).lower()
            if symbol.lower() in content:
                return i
        return 0

    def _get_contexts(
        self, symbol: str, chunks: List[Any], max_contexts: int = 3
    ) -> List[str]:
        """Get context excerpts for symbol.

        Args:
            symbol: Symbol name
            chunks: Text chunks
            max_contexts: Maximum contexts to return

        Returns:
            List of context strings
        """
        contexts = []
        symbol_lower = symbol.lower()

        for chunk in chunks:
            content = getattr(chunk, "content", str(chunk))

            if symbol_lower not in content.lower():
                continue

            # Extract context around symbol
            context = self._extract_context(content, symbol)
            contexts.append(context)

            if len(contexts) >= max_contexts:
                break

        return contexts

    def _extract_context(self, text: str, symbol: str, window: int = 100) -> str:
        """Extract context around symbol.

        Args:
            text: Full text
            symbol: Symbol to find
            window: Context window size

        Returns:
            Context string
        """
        lower_text = text.lower()
        idx = lower_text.find(symbol.lower())

        if idx == -1:
            return text[:200]

        start = max(0, idx - window)
        end = min(len(text), idx + len(symbol) + window)

        context = text[start:end].strip()
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."

        return context

    def analyze_symbolism(self, symbol: str, chunks: List[Any]) -> SymbolAnalysis:
        """Analyze a specific symbol in depth.

        Args:
            symbol: Symbol name to analyze
            chunks: Text chunks for context

        Returns:
            SymbolAnalysis with full analysis
        """
        symbol_lower = symbol.lower()

        # Count occurrences
        count = sum(
            getattr(c, "content", str(c)).lower().count(symbol_lower) for c in chunks
        )

        # Get contexts
        contexts = self._get_contexts(symbol_lower, chunks, max_contexts=5)

        # Create base symbol
        symbol_obj = Symbol(
            name=symbol.title(),
            occurrences=count,
            contexts=contexts,
            associated_themes=self._symbol_patterns.get(symbol_lower, []),
        )

        # Build analysis
        analysis = SymbolAnalysis(symbol=symbol_obj)

        # Get known meanings
        known_meanings = self._symbol_patterns.get(symbol_lower, [])
        if known_meanings:
            analysis.symbolic_meaning = (
                f"Common symbolic meanings: {', '.join(known_meanings[:5])}"
            )

        # Use LLM for enhanced analysis
        if self.llm_client:
            analysis = self._enhance_with_llm(analysis, chunks)

        return analysis

    def _enhance_with_llm(
        self, analysis: SymbolAnalysis, chunks: List[Any]
    ) -> SymbolAnalysis:
        """Enhance symbol analysis using LLM.

        Args:
            analysis: Base analysis to enhance
            chunks: Text chunks

        Returns:
            Enhanced SymbolAnalysis
        """
        # Build context from relevant chunks
        symbol = analysis.symbol.name.lower()
        relevant_contexts = []

        for chunk in chunks:
            content = getattr(chunk, "content", str(chunk))
            if symbol in content.lower():
                relevant_contexts.append(content[:300])
            if len(relevant_contexts) >= 5:
                break

        context = "\n---\n".join(relevant_contexts)

        prompt = (
            f"Analyze the symbol '{analysis.symbol.name}' in these excerpts:\n\n"
            f"{context}\n\n"
            "Provide:\n"
            "1. Literal meaning in the text (1 sentence)\n"
            "2. Symbolic/metaphorical meaning (1-2 sentences)\n"
            "3. Significance to the work (1 sentence)\n"
            "4. How it develops through the narrative (1 sentence)\n\n"
            "Format as JSON with keys: literal, symbolic, significance, development"
        )

        try:
            response = self.llm_client.generate(prompt)
            data = json.loads(response)

            analysis.literal_meaning = data.get("literal", "")
            analysis.symbolic_meaning = data.get("symbolic", analysis.symbolic_meaning)
            analysis.significance = data.get("significance", "")
            analysis.development = data.get("development", "")

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return analysis


# ============================================================================
# Symbols Command Class
# ============================================================================


class SymbolsCommand(LiteraryCommand):
    """Identify and analyze symbolic patterns."""

    def execute(
        self,
        work: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        min_occurrences: int = 3,
        symbol: Optional[str] = None,
    ) -> int:
        """Analyze symbols and motifs in a literary work.

        Args:
            work: Name of the literary work
            project: Project directory
            output: Output file for analysis (optional)
            min_occurrences: Minimum occurrences for symbol
            symbol: Specific symbol to analyze

        Returns:
            0 on success, 1 on error
        """
        try:
            self.validate_work_name(work)

            ctx = self.initialize_context(project, require_storage=True)

            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            chunks = self._search_for_symbols(ctx["storage"], work)

            if not chunks:
                self._handle_no_context(work)
                return 0

            detector = SymbolDetector(llm_client, min_occurrences)

            if symbol:
                return self._analyze_single_symbol(
                    work, symbol, chunks, detector, output
                )

            return self._detect_all_symbols(work, chunks, detector, output)

        except Exception as e:
            return self.handle_error(e, "Symbols analysis failed")

    def _analyze_single_symbol(
        self,
        work: str,
        symbol: str,
        chunks: List[Any],
        detector: SymbolDetector,
        output: Optional[Path],
    ) -> int:
        """Analyze a specific symbol.

        Args:
            work: Literary work name
            symbol: Symbol to analyze
            chunks: Context chunks
            detector: SymbolDetector instance
            output: Output path

        Returns:
            Exit code
        """
        analysis = ProgressManager.run_with_spinner(
            lambda: detector.analyze_symbolism(symbol, chunks),
            f"Analyzing '{symbol}'...",
            "Analysis complete",
        )

        self._display_symbol_analysis(work, analysis)

        if output:
            self._save_symbol_analysis(output, work, analysis)

        return 0

    def _detect_all_symbols(
        self,
        work: str,
        chunks: List[Any],
        detector: SymbolDetector,
        output: Optional[Path],
    ) -> int:
        """Detect all significant symbols.

        Args:
            work: Literary work name
            chunks: Context chunks
            detector: SymbolDetector instance
            output: Output path

        Returns:
            Exit code
        """
        symbols = ProgressManager.run_with_spinner(
            lambda: detector.detect_symbols(chunks),
            "Detecting symbols...",
            "Detection complete",
        )

        if not symbols:
            self.print_warning(
                f"No symbols found with {detector.min_occurrences}+ occurrences"
            )
            return 0

        self._display_symbols(work, symbols)

        if output:
            self._save_symbols(output, work, symbols)

        return 0

    def _search_for_symbols(self, storage: Any, work: str) -> List[Any]:
        """Search for symbolic context."""
        return ProgressManager.run_with_spinner(
            lambda: storage.search(f"{work} symbols imagery motifs", k=20),
            f"Searching for symbolic patterns in '{work}'...",
            "Context retrieved",
        )

    def _handle_no_context(self, work: str) -> None:
        """Handle case where no context found."""
        self.print_warning(f"No context found for '{work}'")
        self.print_info(
            "Try:\n"
            f"  1. Ingesting documents about {work}\n"
            "  2. Using 'lit gather' to fetch Wikipedia pages\n"
            "  3. Checking the work name spelling"
        )

    def _display_symbol_analysis(self, work: str, analysis: SymbolAnalysis) -> None:
        """Display symbol analysis."""
        self.console.print()

        lines = [
            f"## {analysis.symbol.name}",
            f"**Occurrences:** {analysis.symbol.occurrences}",
            "",
        ]

        if analysis.literal_meaning:
            lines.extend(
                [
                    "### Literal Meaning",
                    analysis.literal_meaning,
                    "",
                ]
            )

        if analysis.symbolic_meaning:
            lines.extend(
                [
                    "### Symbolic Meaning",
                    analysis.symbolic_meaning,
                    "",
                ]
            )

        if analysis.significance:
            lines.extend(
                [
                    "### Significance",
                    analysis.significance,
                    "",
                ]
            )

        if analysis.development:
            lines.extend(
                [
                    "### Development",
                    analysis.development,
                    "",
                ]
            )

        if analysis.symbol.associated_themes:
            lines.extend(
                [
                    "### Associated Themes",
                    ", ".join(analysis.symbol.associated_themes),
                    "",
                ]
            )

        if analysis.symbol.contexts:
            lines.append("### Sample Contexts")
            for i, ctx in enumerate(analysis.symbol.contexts[:3], 1):
                lines.append(f'{i}. "{ctx}"')
            lines.append("")

        panel = Panel(
            Markdown("\n".join(lines)),
            title=f"[bold magenta]Symbol Analysis: {analysis.symbol.name} in {work}[/bold magenta]",
            border_style="magenta",
        )
        self.console.print(panel)

    def _display_symbols(self, work: str, symbols: List[Symbol]) -> None:
        """Display detected symbols."""
        self.console.print()

        table = Table(title=f"Symbols in {work}")
        table.add_column("Symbol", style="cyan")
        table.add_column("Occurrences", style="magenta")
        table.add_column("First Appearance", style="dim")
        table.add_column("Associated Themes", style="green")

        for symbol in symbols[:20]:
            themes = ", ".join(symbol.associated_themes[:3])
            table.add_row(
                symbol.name,
                str(symbol.occurrences),
                f"Chunk {symbol.first_appearance}",
                themes or "-",
            )

        self.console.print(table)

    def _save_symbol_analysis(
        self, output: Path, work: str, analysis: SymbolAnalysis
    ) -> None:
        """Save symbol analysis to file."""
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if output.suffix == ".json":
                data = {
                    "work": work,
                    "generated": timestamp,
                    "analysis": analysis.to_dict(),
                }
                output.write_text(json.dumps(data, indent=2), encoding="utf-8")
            else:
                lines = [
                    f"# Symbol Analysis: {analysis.symbol.name}",
                    f"## {work}",
                    "",
                    f"Generated: {timestamp}",
                    "",
                    "---",
                    "",
                    f"## Occurrences: {analysis.symbol.occurrences}",
                    "",
                ]

                if analysis.literal_meaning:
                    lines.extend(["## Literal Meaning", analysis.literal_meaning, ""])

                if analysis.symbolic_meaning:
                    lines.extend(["## Symbolic Meaning", analysis.symbolic_meaning, ""])

                if analysis.significance:
                    lines.extend(["## Significance", analysis.significance, ""])

                if analysis.development:
                    lines.extend(["## Development", analysis.development, ""])

                output.write_text("\n".join(lines), encoding="utf-8")

            self.print_success(f"Analysis saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save: {e}")

    def _save_symbols(self, output: Path, work: str, symbols: List[Symbol]) -> None:
        """Save symbols to file.

        Rule #1: Max 3 nesting levels via helper extraction.
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if output.suffix == ".json":
                data = {
                    "work": work,
                    "generated": timestamp,
                    "symbols": [s.to_dict() for s in symbols],
                }
                output.write_text(json.dumps(data, indent=2), encoding="utf-8")
            else:
                content = self._format_symbols_markdown(work, timestamp, symbols)
                output.write_text(content, encoding="utf-8")

            self.print_success(f"Symbols saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save: {e}")

    def _format_symbols_markdown(
        self, work: str, timestamp: str, symbols: List[Symbol]
    ) -> str:
        """Format symbols as markdown.

        Rule #1: Extracted to reduce nesting in _save_symbols.

        Args:
            work: Work name
            timestamp: Generation timestamp
            symbols: Symbols to format

        Returns:
            Markdown formatted string
        """
        lines = [
            f"# Symbols in {work}",
            "",
            f"Generated: {timestamp}",
            "",
            "---",
            "",
        ]

        for symbol in symbols:
            lines.append(f"## {symbol.name}")
            lines.append(f"- Occurrences: {symbol.occurrences}")
            lines.append(f"- First appearance: Chunk {symbol.first_appearance}")
            if symbol.associated_themes:
                lines.append(f"- Themes: {', '.join(symbol.associated_themes)}")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# Typer Command Wrapper
# ============================================================================


def command(
    work: str = typer.Argument(..., help="Name of the literary work"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for analysis"
    ),
    min_occurrences: int = typer.Option(
        3, "--min-occurrences", "-m", help="Minimum occurrences to detect"
    ),
    symbol: Optional[str] = typer.Option(
        None, "--symbol", "-s", help="Specific symbol to analyze"
    ),
) -> None:
    """Identify and analyze symbolic patterns in a literary work.

    Analyzes symbols, motifs, and recurring imagery to understand
    their significance and impact on the work's meaning.

    Requires documents about the work to be ingested first.

    Examples:
        # Detect symbols
        ingestforge lit symbols "The Great Gatsby"

        # Analyze specific symbol
        ingestforge lit symbols "Moby Dick" --symbol "whale"

        # Adjust detection threshold
        ingestforge lit symbols "Lord of the Flies" -m 5

        # Save to file
        ingestforge lit symbols "1984" -o symbols.json
    """
    cmd = SymbolsCommand()
    exit_code = cmd.execute(work, project, output, min_occurrences, symbol)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# Subcommand for detection
def detect_command(
    work: str = typer.Argument(..., help="Name of the literary work"),
    min_occurrences: int = typer.Option(
        3, "--min-occurrences", "-m", help="Minimum occurrences"
    ),
    project: Optional[Path] = typer.Option(None, "--project", "-p"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Detect symbolic elements in a literary work.

    Examples:
        ingestforge lit symbols detect "The Scarlet Letter" --min-occurrences 5
    """
    cmd = SymbolsCommand()
    exit_code = cmd.execute(work, project, output, min_occurrences, None)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# Subcommand for analysis
def analyze_command(
    symbol: str = typer.Argument(..., help="Symbol to analyze"),
    work: str = typer.Option(..., "--work", "-w", help="Literary work"),
    project: Optional[Path] = typer.Option(None, "--project", "-p"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Analyze a specific symbol in depth.

    Examples:
        ingestforge lit symbols analyze "green light" --work "Great Gatsby"
    """
    cmd = SymbolsCommand()
    exit_code = cmd.execute(work, project, output, 1, symbol)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
