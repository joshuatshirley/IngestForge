"""Themes command - Extract and analyze literary themes.

Provides theme detection, evidence finding, and theme development tracking:
- detect: Detect major themes in a work
- analyze: Analyze a specific theme with evidence
- compare: Compare multiple themes"""

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
from ingestforge.cli.literary.models import (
    Evidence,
    Theme,
    ThemeArc,
    ThemeComparison,
    ThemePoint,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# Common literary themes for detection
COMMON_THEMES = [
    "love",
    "death",
    "redemption",
    "power",
    "identity",
    "justice",
    "freedom",
    "betrayal",
    "revenge",
    "sacrifice",
    "isolation",
    "corruption",
    "innocence",
    "fate",
    "ambition",
    "morality",
    "nature",
    "war",
    "family",
    "society",
    "truth",
    "deception",
]

# ============================================================================
# Theme Detector Class
# ============================================================================


class ThemeDetector:
    """Detect and analyze themes in literary text.

    Uses keyword matching and LLM for theme identification.
    Follows Rule #4: All methods <60 lines.
    """

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        """Initialize theme detector.

        Args:
            llm_client: Optional LLM client for enhanced detection
        """
        self.llm_client = llm_client
        self._theme_keywords = self._build_theme_keywords()

    def _build_theme_keywords(self) -> Dict[str, List[str]]:
        """Build theme keyword mappings.

        Returns:
            Dict mapping theme names to related keywords
        """
        return {
            "love": ["love", "heart", "passion", "beloved", "affection", "romance"],
            "death": ["death", "die", "dying", "dead", "mortal", "grave", "funeral"],
            "redemption": ["redeem", "forgive", "salvation", "atone", "repent"],
            "power": ["power", "control", "authority", "rule", "dominate", "throne"],
            "identity": ["identity", "self", "who am i", "belong", "true self"],
            "justice": ["justice", "fair", "right", "wrong", "trial", "judge"],
            "freedom": ["freedom", "free", "liberty", "escape", "chains", "prison"],
            "betrayal": ["betray", "traitor", "trust", "deceive", "backstab"],
            "revenge": ["revenge", "vengeance", "avenge", "retribution", "payback"],
            "sacrifice": ["sacrifice", "give up", "selfless", "martyr", "cost"],
            "isolation": ["alone", "lonely", "isolated", "solitude", "outcast"],
            "corruption": ["corrupt", "decay", "rotten", "moral", "fallen"],
            "innocence": ["innocent", "pure", "naive", "child", "untainted"],
            "fate": ["fate", "destiny", "fortune", "doom", "inevitable"],
            "ambition": ["ambition", "aspire", "strive", "goal", "success"],
            "morality": ["moral", "ethics", "right", "wrong", "conscience"],
            "nature": ["nature", "natural", "earth", "wild", "environment"],
            "war": ["war", "battle", "fight", "conflict", "soldier", "army"],
            "family": ["family", "father", "mother", "son", "daughter", "blood"],
            "society": ["society", "social", "class", "status", "community"],
            "truth": ["truth", "honest", "lie", "real", "genuine", "authentic"],
            "deception": ["deceive", "trick", "false", "pretend", "mask", "hide"],
        }

    def detect_themes(self, chunks: List[Any], top_n: int = 5) -> List[Theme]:
        """Detect major themes in text chunks.

        Args:
            chunks: List of ChunkRecord objects
            top_n: Number of top themes to return

        Returns:
            List of detected Theme objects

        Rule #1: Early return for empty input
        """
        if not chunks:
            return []

        # Combine chunk content
        combined_text = " ".join(getattr(c, "content", str(c)).lower() for c in chunks)

        # Score themes by keyword frequency
        theme_scores = self._score_themes_by_keywords(combined_text)

        # Use LLM for enhanced detection if available
        if self.llm_client:
            llm_themes = self._detect_with_llm(chunks, top_n)
            theme_scores = self._merge_theme_scores(theme_scores, llm_themes)

        # Convert to Theme objects
        themes = self._create_theme_objects(theme_scores, combined_text, top_n)

        return themes

    def _score_themes_by_keywords(self, text: str) -> Dict[str, float]:
        """Score themes based on keyword frequency.

        Args:
            text: Combined text content

        Returns:
            Dict of theme names to scores
        """
        scores: Dict[str, float] = defaultdict(float)
        text_lower = text.lower()
        total_words = len(text_lower.split())

        for theme, keywords in self._theme_keywords.items():
            for keyword in keywords:
                count = text_lower.count(keyword)
                # Normalize by text length and keyword count
                scores[theme] += count / max(total_words, 1) * 1000

        return dict(scores)

    def _detect_with_llm(self, chunks: List[Any], top_n: int) -> Dict[str, float]:
        """Detect themes using LLM.

        Args:
            chunks: Text chunks
            top_n: Number of themes to request

        Returns:
            Dict of theme names to confidence scores
        """
        context = "\n---\n".join(
            getattr(c, "content", str(c))[:500] for c in chunks[:5]
        )

        prompt = (
            f"Analyze these text excerpts and identify the {top_n} major themes:\n\n"
            f"{context}\n\n"
            "Return as JSON array of objects with 'name' and 'confidence' (0-1).\n"
            'Example: [{"name": "love", "confidence": 0.85}]'
        )

        try:
            response = self.llm_client.generate(prompt)
            data = json.loads(response)

            return {
                item.get("name", "").lower(): item.get("confidence", 0.5)
                for item in data
                if isinstance(item, dict)
            }

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM theme detection failed: {e}")
            return {}

    def _merge_theme_scores(
        self, keyword_scores: Dict[str, float], llm_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Merge keyword and LLM theme scores.

        Args:
            keyword_scores: Scores from keyword matching
            llm_scores: Scores from LLM analysis

        Returns:
            Merged scores
        """
        merged = dict(keyword_scores)

        for theme, score in llm_scores.items():
            if theme in merged:
                merged[theme] = (merged[theme] + score * 10) / 2
            else:
                merged[theme] = score * 10

        return merged

    def _create_theme_objects(
        self, scores: Dict[str, float], text: str, top_n: int
    ) -> List[Theme]:
        """Create Theme objects from scores.

        Args:
            scores: Theme name to score mapping
            text: Full text for evidence counting
            top_n: Number of themes to return

        Returns:
            List of Theme objects
        """
        sorted_themes = sorted(scores.items(), key=lambda x: -x[1])[:top_n]

        max_score = max(scores.values()) if scores else 1

        themes = []
        for name, score in sorted_themes:
            if score <= 0:
                continue

            confidence = min(0.95, score / max_score)
            keywords = self._theme_keywords.get(name, [])

            # Count evidence occurrences
            evidence_count = sum(text.count(kw) for kw in keywords)

            theme = Theme(
                name=name.title(),
                confidence=round(confidence, 2),
                keywords=keywords[:5],
                evidence_count=evidence_count,
                description=f"Theme of {name} detected with {evidence_count} supporting instances",
            )
            themes.append(theme)

        return themes

    def find_evidence(
        self, theme: str, chunks: List[Any], max_evidence: int = 10
    ) -> List[Evidence]:
        """Find evidence supporting a theme.

        Args:
            theme: Theme name to find evidence for
            chunks: Text chunks to search
            max_evidence: Maximum evidence items to return

        Returns:
            List of Evidence objects
        """
        theme_lower = theme.lower()
        keywords = self._theme_keywords.get(theme_lower, [theme_lower])

        evidence_list: List[Evidence] = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            content = getattr(chunk, "content", str(chunk))
            chunk_id = getattr(chunk, "chunk_id", f"chunk_{i}")

            for keyword in keywords:
                if keyword.lower() not in content.lower():
                    continue

                quote = self._extract_quote(content, keyword)
                position = i / max(total_chunks - 1, 1)

                evidence = Evidence(
                    quote=quote,
                    chunk_id=chunk_id,
                    explanation=f"Contains theme keyword '{keyword}'",
                    position=position,
                )
                evidence_list.append(evidence)

                if len(evidence_list) >= max_evidence:
                    return evidence_list

        return evidence_list

    def _extract_quote(self, text: str, keyword: str, window: int = 150) -> str:
        """Extract quote around keyword.

        Args:
            text: Full text
            keyword: Keyword to center on
            window: Context window size

        Returns:
            Quote string
        """
        lower_text = text.lower()
        idx = lower_text.find(keyword.lower())

        if idx == -1:
            return text[:300] if len(text) > 300 else text

        start = max(0, idx - window)
        end = min(len(text), idx + len(keyword) + window)

        quote = text[start:end].strip()

        if start > 0:
            quote = "..." + quote
        if end < len(text):
            quote = quote + "..."

        return quote

    def track_theme_development(self, theme: str, chunks: List[Any]) -> ThemeArc:
        """Track theme development across narrative.

        Args:
            theme: Theme to track
            chunks: Ordered text chunks

        Returns:
            ThemeArc with development data
        """
        theme_obj = Theme(name=theme.title())
        keywords = self._theme_keywords.get(theme.lower(), [theme.lower()])

        development: List[ThemePoint] = []
        peak_moments: List[str] = []
        max_intensity = 0

        for i, chunk in enumerate(chunks):
            content = getattr(chunk, "content", str(chunk)).lower()
            chunk_id = getattr(chunk, "chunk_id", f"chunk_{i}")
            position = i / max(len(chunks) - 1, 1)

            # Calculate intensity based on keyword frequency
            intensity = sum(content.count(kw) for kw in keywords)
            normalized = min(1.0, intensity / 10)

            point = ThemePoint(
                position=round(position, 3),
                intensity=round(normalized, 3),
                chunk_id=chunk_id,
            )
            development.append(point)

            # Track peaks
            if intensity > max_intensity:
                max_intensity = intensity
                peak_moments.append(f"Peak at position {position:.1%}: {chunk_id}")

        return ThemeArc(
            theme=theme_obj,
            development=development,
            peak_moments=peak_moments[-5:],  # Keep top 5 peaks
        )

    def compare_themes(self, themes: List[Theme]) -> ThemeComparison:
        """Compare multiple themes.

        Args:
            themes: Themes to compare

        Returns:
            ThemeComparison with analysis
        """
        comparison = ThemeComparison(themes=themes)

        # Build simple correlation (based on keyword overlap)
        for i, theme1 in enumerate(themes):
            comparison.correlations[theme1.name] = {}

            for j, theme2 in enumerate(themes):
                if i == j:
                    comparison.correlations[theme1.name][theme2.name] = 1.0
                    continue

                # Calculate overlap in keywords
                kw1 = set(self._theme_keywords.get(theme1.name.lower(), []))
                kw2 = set(self._theme_keywords.get(theme2.name.lower(), []))

                overlap = len(kw1 & kw2)
                total = len(kw1 | kw2) or 1

                comparison.correlations[theme1.name][theme2.name] = overlap / total

        comparison.summary = self._generate_comparison_summary(themes)
        return comparison

    def _generate_comparison_summary(self, themes: List[Theme]) -> str:
        """Generate comparison summary.

        Args:
            themes: Themes to summarize

        Returns:
            Summary string
        """
        if not themes:
            return "No themes to compare"

        sorted_themes = sorted(themes, key=lambda t: -t.confidence)
        primary = sorted_themes[0]

        summary = (
            f"Primary theme: {primary.name} (confidence: {primary.confidence:.0%}). "
        )

        if len(themes) > 1:
            others = ", ".join(t.name for t in sorted_themes[1:3])
            summary += f"Supporting themes: {others}."

        return summary


# ============================================================================
# Themes Command Class
# ============================================================================


class ThemesCommand(LiteraryCommand):
    """Extract and analyze literary themes."""

    def execute(
        self,
        work: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        top_n: int = 5,
        theme: Optional[str] = None,
        evidence: bool = False,
        store: bool = False,
    ) -> int:
        """Analyze themes in a literary work.

        Args:
            work: Name of the literary work
            project: Project directory
            output: Output file for analysis (optional)
            top_n: Number of top themes to detect
            theme: Specific theme to analyze
            evidence: Include evidence for themes
            store: Store analysis in vector database

        Returns:
            0 on success, 1 on error
        """
        try:
            self.validate_work_name(work)

            ctx = self.initialize_context(project, require_storage=True)

            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            chunks = self.search_literary_context(ctx["storage"], work)

            if not chunks:
                self._handle_no_context(work)
                return 0

            detector = ThemeDetector(llm_client)

            if theme:
                return self._analyze_single_theme(
                    work, theme, chunks, detector, output, evidence, store, ctx
                )

            return self._detect_all_themes(
                work, chunks, detector, output, top_n, evidence, store, ctx
            )

        except Exception as e:
            return self.handle_error(e, "Themes analysis failed")

    def _analyze_single_theme(
        self,
        work: str,
        theme: str,
        chunks: List[Any],
        detector: ThemeDetector,
        output: Optional[Path],
        show_evidence: bool,
        store: bool = False,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Analyze a single theme in depth.

        Args:
            work: Literary work name
            theme: Theme to analyze
            chunks: Context chunks
            detector: ThemeDetector instance
            output: Output path
            show_evidence: Show supporting evidence
            store: Store analysis in vector database
            ctx: Context dict with config

        Returns:
            Exit code
        """
        # Get theme development
        arc = detector.track_theme_development(theme, chunks)

        # Find evidence
        evidence_list: List[Evidence] = []
        if show_evidence:
            evidence_list = detector.find_evidence(theme, chunks)

        self._display_theme_analysis(work, arc, evidence_list)

        if output:
            self._save_theme_analysis(output, work, arc, evidence_list)

        # Store analysis if requested
        if store and ctx:
            self._store_theme_analysis(work, arc, evidence_list, chunks, ctx)

        return 0

    def _detect_all_themes(
        self,
        work: str,
        chunks: List[Any],
        detector: ThemeDetector,
        output: Optional[Path],
        top_n: int,
        show_evidence: bool,
        store: bool = False,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Detect and display all major themes.

        Args:
            work: Literary work name
            chunks: Context chunks
            detector: ThemeDetector instance
            output: Output path
            top_n: Number of themes
            show_evidence: Show evidence
            store: Store analysis in vector database
            ctx: Context dict with config

        Returns:
            Exit code
        """
        themes = detector.detect_themes(chunks, top_n)

        if not themes:
            self.print_warning("No themes detected")
            return 0

        # Get evidence if requested
        theme_evidence: Dict[str, List[Evidence]] = {}
        if show_evidence:
            for theme in themes:
                evidence_list = detector.find_evidence(
                    theme.name, chunks, max_evidence=3
                )
                theme_evidence[theme.name] = evidence_list

        self._display_themes(work, themes, theme_evidence)

        if output:
            self._save_themes(output, work, themes, theme_evidence)

        # Store analysis if requested
        if store and ctx:
            self._store_themes_analysis(work, themes, theme_evidence, chunks, ctx)

        return 0

    def _handle_no_context(self, work: str) -> None:
        """Handle case where no context found."""
        self.print_warning(f"No context found for '{work}'")
        self.print_info(
            "Try:\n"
            f"  1. Ingesting documents about {work}\n"
            "  2. Using the 'lit gather' command to fetch Wikipedia pages\n"
            "  3. Checking the work name spelling"
        )

    def _display_theme_analysis(
        self, work: str, arc: ThemeArc, evidence: List[Evidence]
    ) -> None:
        """Display single theme analysis."""
        self.console.print()

        lines = [
            f"## {arc.theme.name}",
            "",
            "### Development",
            f"The theme develops across {len(arc.development)} narrative points.",
            "",
        ]

        if arc.peak_moments:
            lines.append("### Peak Moments")
            for moment in arc.peak_moments:
                lines.append(f"- {moment}")
            lines.append("")

        if evidence:
            lines.append("### Supporting Evidence")
            for i, ev in enumerate(evidence[:5], 1):
                lines.append(f'**{i}.** "{ev.quote}"')
                lines.append(f"   - {ev.explanation}")
                lines.append("")

        # Show ASCII tension curve
        if arc.development:
            lines.append("### Intensity Curve")
            lines.append("```")
            intensities = [p.intensity for p in arc.development]
            lines.append(self._mini_ascii_chart(intensities))
            lines.append("```")

        panel = Panel(
            Markdown("\n".join(lines)),
            title=f"[bold cyan]Theme Analysis: {arc.theme.name} in {work}[/bold cyan]",
            border_style="cyan",
        )
        self.console.print(panel)

    def _mini_ascii_chart(self, values: List[float], width: int = 40) -> str:
        """Create mini ASCII chart of values.

        Args:
            values: Values to chart
            width: Chart width

        Returns:
            ASCII chart string
        """
        if not values:
            return "No data"

        # Sample values to fit width
        step = max(1, len(values) // width)
        sampled = [values[i] for i in range(0, len(values), step)][:width]

        max_val = max(sampled) if sampled else 1
        chars = " _.-=+*#"

        result = ""
        for val in sampled:
            idx = int((val / max_val) * (len(chars) - 1))
            result += chars[idx]

        return f"[{result}]"

    def _display_themes(
        self,
        work: str,
        themes: List[Theme],
        evidence: Dict[str, List[Evidence]],
    ) -> None:
        """Display detected themes."""
        self.console.print()

        table = Table(title=f"Themes in {work}")
        table.add_column("Theme", style="cyan")
        table.add_column("Confidence", style="magenta")
        table.add_column("Evidence", style="green")
        table.add_column("Keywords", style="dim")

        for theme in themes:
            keywords = ", ".join(theme.keywords[:3])
            table.add_row(
                theme.name,
                f"{theme.confidence:.0%}",
                str(theme.evidence_count),
                keywords,
            )

        self.console.print(table)

        # Show evidence if available
        if evidence:
            self.console.print()
            self.console.print("[bold]Supporting Evidence:[/bold]")

            for theme_name, ev_list in evidence.items():
                if not ev_list:
                    continue
                self.console.print(f"\n[cyan]{theme_name}:[/cyan]")
                for ev in ev_list[:2]:
                    self.console.print(f'  - "{ev.quote[:100]}..."')

    def _save_theme_analysis(
        self,
        output: Path,
        work: str,
        arc: ThemeArc,
        evidence: List[Evidence],
    ) -> None:
        """Save theme analysis to file."""
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if output.suffix == ".json":
                data = {
                    "work": work,
                    "generated": timestamp,
                    "theme_arc": arc.to_dict(),
                    "evidence": [
                        {"quote": e.quote, "position": e.position} for e in evidence
                    ],
                }
                output.write_text(json.dumps(data, indent=2), encoding="utf-8")
            else:
                content = self._format_theme_markdown(work, arc, evidence, timestamp)
                output.write_text(content, encoding="utf-8")

            self.print_success(f"Analysis saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save: {e}")

    def _format_theme_markdown(
        self,
        work: str,
        arc: ThemeArc,
        evidence: List[Evidence],
        timestamp: str,
    ) -> str:
        """Format theme analysis as markdown."""
        lines = [
            f"# Theme Analysis: {arc.theme.name}",
            f"## {work}",
            "",
            f"Generated: {timestamp}",
            "",
            "---",
            "",
            "## Development",
            f"Theme tracked across {len(arc.development)} points.",
            "",
        ]

        if arc.peak_moments:
            lines.append("## Peak Moments")
            for moment in arc.peak_moments:
                lines.append(f"- {moment}")
            lines.append("")

        if evidence:
            lines.append("## Evidence")
            for i, ev in enumerate(evidence, 1):
                lines.append(f'{i}. "{ev.quote}"')
            lines.append("")

        return "\n".join(lines)

    def _save_themes(
        self,
        output: Path,
        work: str,
        themes: List[Theme],
        evidence: Dict[str, List[Evidence]],
    ) -> None:
        """Save themes to file."""
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if output.suffix == ".json":
                data = {
                    "work": work,
                    "generated": timestamp,
                    "themes": [t.to_dict() for t in themes],
                    "evidence": {
                        name: [{"quote": e.quote} for e in evs]
                        for name, evs in evidence.items()
                    },
                }
                output.write_text(json.dumps(data, indent=2), encoding="utf-8")
            else:
                lines = [
                    f"# Themes in {work}",
                    "",
                    f"Generated: {timestamp}",
                    "",
                    "---",
                    "",
                ]

                for theme in themes:
                    lines.append(f"## {theme.name}")
                    lines.append(f"- Confidence: {theme.confidence:.0%}")
                    lines.append(f"- Evidence count: {theme.evidence_count}")
                    lines.append(f"- Keywords: {', '.join(theme.keywords)}")
                    lines.append("")

                output.write_text("\n".join(lines), encoding="utf-8")

            self.print_success(f"Themes saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save: {e}")

    def _store_theme_analysis(
        self,
        work: str,
        arc: ThemeArc,
        evidence: List[Evidence],
        chunks: List[Any],
        ctx: Dict[str, Any],
    ) -> None:
        """Store single theme analysis in vector database."""
        try:
            from ingestforge.storage.analysis import (
                AnalysisRecord,
                get_analysis_storage,
            )

            # Build analysis content
            content_parts = [
                f"Theme Analysis: {arc.theme.name} in {work}",
                "",
                f"Theme: {arc.theme.name}",
                f"Confidence: {arc.theme.confidence:.0%}",
                f"Development tracked across {len(arc.development)} narrative points.",
                "",
            ]

            if arc.peak_moments:
                content_parts.append("Peak Moments:")
                for moment in arc.peak_moments:
                    content_parts.append(f"  - {moment}")
                content_parts.append("")

            if evidence:
                content_parts.append("Supporting Evidence:")
                for ev in evidence[:5]:
                    content_parts.append(f'  - "{ev.quote[:200]}..."')

            content = "\n".join(content_parts)

            # Get source chunks IDs
            source_chunks = [
                getattr(c, "chunk_id", f"chunk_{i}") for i, c in enumerate(chunks[:10])
            ]

            # Create record
            record = AnalysisRecord(
                analysis_id=AnalysisRecord.generate_id(),
                analysis_type="theme",
                content=content,
                source_document=work,
                source_chunks=source_chunks,
                confidence=arc.theme.confidence,
                metadata={
                    "theme_name": arc.theme.name,
                    "peak_moments": arc.peak_moments,
                    "keywords": arc.theme.keywords,
                    "evidence_count": len(evidence),
                },
                title=f"Theme: {arc.theme.name} in {work}",
            )

            # Store
            storage = get_analysis_storage(ctx.get("persist_directory"))
            analysis_id = storage.store_analysis(record)

            self.print_success(f"Analysis stored: {analysis_id}")

        except Exception as e:
            self.print_warning(f"Failed to store analysis: {e}")
            logger.error(f"Store analysis failed: {e}")

    def _store_themes_analysis(
        self,
        work: str,
        themes: List[Theme],
        evidence: Dict[str, List[Evidence]],
        chunks: List[Any],
        ctx: Dict[str, Any],
    ) -> None:
        """Store multiple themes analysis in vector database.

        Rule #1: Max 3 nesting levels.
        Rule #4: Functions <60 lines.
        """
        try:
            from ingestforge.storage.analysis import (
                get_analysis_storage,
            )

            # Build analysis content
            content = self._build_themes_content(work, themes, evidence)

            # Create and store record
            record = self._create_themes_record(work, themes, chunks, content)
            storage = get_analysis_storage(ctx.get("persist_directory"))
            analysis_id = storage.store_analysis(record)

            self.print_success(f"Analysis stored: {analysis_id}")

        except Exception as e:
            self.print_warning(f"Failed to store analysis: {e}")
            logger.error(f"Store analysis failed: {e}")

    def _build_themes_content(
        self,
        work: str,
        themes: List[Theme],
        evidence: Dict[str, List[Evidence]],
    ) -> str:
        """Build content string for themes analysis.

        Helper for _store_themes_analysis.

        Args:
            work: Work name
            themes: List of themes
            evidence: Evidence dict

        Returns:
            Content string
        """
        content_parts = [
            f"Theme Detection: {work}",
            "",
            f"Detected {len(themes)} major themes:",
            "",
        ]

        for theme in themes:
            self._add_theme_to_content(theme, evidence, content_parts)

        return "\n".join(content_parts)

    def _create_themes_record(
        self,
        work: str,
        themes: List[Theme],
        chunks: List[Any],
        content: str,
    ) -> "AnalysisRecord":
        """Create analysis record for themes.

        Helper for _store_themes_analysis.

        Args:
            work: Work name
            themes: List of themes
            chunks: Source chunks
            content: Content string

        Returns:
            AnalysisRecord instance
        """
        from ingestforge.storage.analysis import AnalysisRecord

        # Get source chunks IDs
        source_chunks = [
            getattr(c, "chunk_id", f"chunk_{i}") for i, c in enumerate(chunks[:10])
        ]

        # Calculate average confidence
        avg_confidence = (
            sum(t.confidence for t in themes) / len(themes) if themes else 0.5
        )

        # Create record
        return AnalysisRecord(
            analysis_id=AnalysisRecord.generate_id(),
            analysis_type="theme",
            content=content,
            source_document=work,
            source_chunks=source_chunks,
            confidence=avg_confidence,
            metadata={
                "themes": [t.to_dict() for t in themes],
                "theme_count": len(themes),
            },
            title=f"Themes in {work}",
        )

    def _add_theme_to_content(
        self,
        theme: Theme,
        evidence: Dict[str, List[Evidence]],
        content_parts: List[str],
    ) -> None:
        """Add theme information to content parts.

        Helper for _store_themes_analysis to reduce nesting.

        Args:
            theme: Theme object
            evidence: Evidence dict
            content_parts: List to append content to
        """
        content_parts.append(f"## {theme.name}")
        content_parts.append(f"- Confidence: {theme.confidence:.0%}")
        content_parts.append(f"- Evidence count: {theme.evidence_count}")
        content_parts.append(f"- Keywords: {', '.join(theme.keywords)}")
        content_parts.append("")

        if theme.name not in evidence or not evidence[theme.name]:
            return

        content_parts.append("Evidence:")
        for ev in evidence[theme.name][:2]:
            content_parts.append(f'  - "{ev.quote[:150]}..."')
        content_parts.append("")


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
    top: int = typer.Option(5, "--top", "-n", help="Number of top themes to detect"),
    theme: Optional[str] = typer.Option(
        None, "--theme", "-t", help="Specific theme to analyze"
    ),
    evidence: bool = typer.Option(
        False, "--evidence", "-e", help="Include supporting evidence"
    ),
    store: bool = typer.Option(
        False, "--store", "-s", help="Store analysis in vector database for searching"
    ),
) -> None:
    """Extract and analyze major themes in a literary work.

    Analyzes a literary work to identify:
    - Major themes and motifs
    - Theme development across narrative
    - Supporting evidence

    Requires documents about the work to be ingested first.

    Examples:
        # Detect top themes
        ingestforge lit themes "Hamlet"

        # Analyze specific theme
        ingestforge lit themes "Hamlet" --theme "revenge"

        # Include evidence
        ingestforge lit themes "1984" --evidence

        # Save to file
        ingestforge lit themes "Hamlet" -o hamlet_themes.json

        # Store analysis for later searching
        ingestforge lit themes "1984" --store
    """
    cmd = ThemesCommand()
    exit_code = cmd.execute(work, project, output, top, theme, evidence, store)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# Subcommand for theme detection
def detect_command(
    work: str = typer.Argument(..., help="Name of the literary work"),
    top: int = typer.Option(5, "--top", "-n", help="Number of themes"),
    project: Optional[Path] = typer.Option(None, "--project", "-p"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Detect major themes in a literary work.

    Examples:
        ingestforge lit themes detect "Great Gatsby" --top 5
    """
    cmd = ThemesCommand()
    exit_code = cmd.execute(work, project, output, top, None, False)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# Subcommand for theme analysis with evidence
def analyze_command(
    theme: str = typer.Argument(..., help="Theme to analyze"),
    work: str = typer.Option(..., "--work", "-w", help="Literary work"),
    project: Optional[Path] = typer.Option(None, "--project", "-p"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Analyze a specific theme with evidence.

    Examples:
        ingestforge lit themes analyze "redemption" --work "Crime and Punishment"
    """
    cmd = ThemesCommand()
    exit_code = cmd.execute(work, project, output, 5, theme, True)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
