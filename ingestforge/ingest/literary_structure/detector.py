"""
Literary structure detection for prose, poetry, and dramatic works.

Provides LiteraryStructureDetector for analyzing text structure.
"""

import re
from typing import Any, List, Set

from ingestforge.ingest.literary_structure.models import (
    DRAMATIC_DIALOGUE_RE,
    PROSE_DIALOGUE_RE,
    PROSE_DIALOGUE_REV_RE,
    SECTION_PATTERNS,
    STAGE_DIRECTION_RE,
    DialogueLine,
    LiterarySection,
    LiteraryStructure,
)


class LiteraryStructureDetector:
    """Analyzes text to detect literary structure elements."""

    def analyze(self, text: str) -> LiteraryStructure:
        """Perform full structural analysis of a text.

        Args:
            text: The full text to analyze.

        Returns:
            LiteraryStructure with sections, dialogue, verse/dramatic flags,
            and estimated form.
        """
        if not text or not text.strip():
            return LiteraryStructure(
                sections=[],
                dialogue=[],
                is_verse=False,
                is_dramatic=False,
                estimated_form="short_story",
            )

        lines = text.split("\n")

        # Detect structural sections
        raw_sections = self._detect_sections(lines)
        sections = self._build_hierarchy(raw_sections)

        # Detect dialogue
        dialogue = self._detect_dialogue(text, lines)

        # Detect verse and dramatic form
        is_verse = self._detect_verse(lines)
        is_dramatic = self._detect_dramatic(text, lines)

        # Estimate literary form
        estimated_form = self._estimate_form(
            text, lines, sections, dialogue, is_verse, is_dramatic
        )

        return LiteraryStructure(
            sections=sections,
            dialogue=dialogue,
            is_verse=is_verse,
            is_dramatic=is_dramatic,
            estimated_form=estimated_form,
        )

    def get_speakers(self, text: str) -> List[str]:
        """Extract unique speaker names from text, sorted alphabetically.

        Args:
            text: Full text to scan for speakers.

        Returns:
            Sorted list of unique speaker names.
        """
        lines = text.split("\n")
        dialogue = self._detect_dialogue(text, lines)
        speakers = sorted(set(d.speaker for d in dialogue))
        return speakers

    def get_section_hierarchy(self, text: str, pos: int) -> List[str]:
        """Get the section hierarchy for a character position.

        Args:
            text: Full text.
            pos: Character offset into the text.

        Returns:
            List of section labels from outermost to innermost.
        """
        lines = text.split("\n")
        raw_sections = self._detect_sections(lines)
        hierarchy = self._build_hierarchy(raw_sections)

        # Convert character position to line number
        line_number = text[:pos].count("\n")

        # Walk the hierarchy to find containing sections
        result: list[Any] = []
        self._find_containing_sections(hierarchy, line_number, result)
        return result

    def _find_containing_sections(
        self,
        sections: List[LiterarySection],
        line_number: int,
        result: List[str],
    ) -> None:
        """Recursively find sections that contain the given line."""
        for section in sections:
            if section.start_line <= line_number <= section.end_line:
                label = section.label
                if section.title:
                    label = f"{section.label}: {section.title}"
                result.append(label)
                self._find_containing_sections(section.children, line_number, result)
                break

    # ------------------------------------------------------------------
    # Section detection
    # ------------------------------------------------------------------

    def _extract_section_title(self, match: Any, stype: str, pattern: Any) -> str:
        """
        Extract section title from regex match.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            match: Regex match object
            stype: Section type
            pattern: Compiled regex pattern

        Returns:
            Extracted title string
        """
        if not match.lastindex or match.lastindex < 1:
            return ""

        captured: str = match.group(1) or ""
        # (group(1) is the numeral itself, not a title)
        if stype == "chapter" and pattern.pattern == r"^([IVXLCDM]+)$":
            return ""

        return captured.strip() if captured else ""

    def _refine_section_label(self, line: str, title: str) -> str:
        """
        Refine section label from line and title.

        Rule #1: Early returns eliminate nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            line: Full line text
            title: Extracted title

        Returns:
            Refined label
        """
        label = line
        if not title:
            return label

        # For "Part One: The Beginning", label is just "Part One"
        if ":" in line:
            label = line.split(":")[0].strip()
            return label if label else line

        # For "Part One - The Beginning", label is just "Part One"
        if " - " in line:
            label = line.split(" - ")[0].strip()
            return label if label else line

        return label

    def _create_section_from_match(
        self,
        line: str,
        line_index: int,
        stype: str,
        level: int,
        match: Any,
        pattern: Any,
        total_lines: int,
    ) -> LiterarySection:
        """
        Create LiterarySection from pattern match.

        Rule #1: Helper function eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            line: Matched line text
            line_index: Line index in document
            stype: Section type
            level: Section level
            match: Regex match object
            pattern: Compiled regex pattern
            total_lines: Total number of lines

        Returns:
            Created LiterarySection
        """
        title = self._extract_section_title(match, stype, pattern)
        label = self._refine_section_label(line, title)

        return LiterarySection(
            section_type=stype,
            label=label,
            title=title,
            start_line=line_index,
            end_line=total_lines - 1,  # Will be refined later
            level=level,
        )

    def _try_match_section_pattern(
        self,
        line: str,
        line_index: int,
        total_lines: int,
        seen_start_lines: Set[int],
        sections: List[LiterarySection],
    ) -> bool:
        """
        Try to match line against section patterns.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            line: Line to match
            line_index: Line index in document
            total_lines: Total number of lines
            seen_start_lines: Set of already processed line indices
            sections: List to append section to (mutated)

        Returns:
            True if match found and section created
        """
        if line_index in seen_start_lines:
            return False
        for pattern, stype, level in SECTION_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue
            if self._is_false_positive(line, stype):
                continue

            # Valid section found - create and add
            section = self._create_section_from_match(
                line, line_index, stype, level, match, pattern, total_lines
            )
            sections.append(section)
            seen_start_lines.add(line_index)
            return True

        return False

    def _refine_section_end_lines(self, sections: List[LiterarySection]) -> None:
        """
        Refine end_line for each section.

        Rule #1: Simple nested loops (max 2 levels)
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Each section ends where the next one at the same or
        higher level begins (or at end of text).

        Args:
            sections: List of sections to refine (mutated)
        """
        for idx in range(len(sections)):
            for jdx in range(idx + 1, len(sections)):
                if sections[jdx].level <= sections[idx].level:
                    sections[idx].end_line = sections[jdx].start_line - 1
                    break

    def _detect_sections(self, lines: List[str]) -> List[LiterarySection]:
        """
        Detect structural section markers in text lines.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            lines: List of text lines to analyze

        Returns:
            List of detected literary sections
        """
        sections: List[LiterarySection] = []
        seen_start_lines: Set[int] = set()
        total_lines = len(lines)
        for i, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            self._try_match_section_pattern(
                line, i, total_lines, seen_start_lines, sections
            )
        self._refine_section_end_lines(sections)

        return sections

    def _is_false_positive(self, line: str, stype: str) -> bool:
        """Check if a detected section marker is a false positive."""
        lower = line.lower()

        # "chapter of accidents", "part of the problem", etc.
        false_phrases = [
            "chapter of",
            "part of",
            "act now",
            "act on",
            "scene we",
            "scene of",
            "scene that",
            "scene in",
            "book of",
            "book that",
        ]
        for phrase in false_phrases:
            if phrase in lower:
                return True

        # If the line is very long, it's probably prose, not a heading.
        if len(line) > 80:
            return True

        return False

    # ------------------------------------------------------------------
    # Hierarchy building
    # ------------------------------------------------------------------

    def _build_hierarchy(
        self, sections: List[LiterarySection]
    ) -> List[LiterarySection]:
        """Arrange flat sections into a nested hierarchy.

        Higher-level sections (lower level number) become parents of
        lower-level sections that fall within their line range.
        """
        if not sections:
            return []

        # Deep copy to avoid mutating originals
        nodes = [
            LiterarySection(
                section_type=s.section_type,
                label=s.label,
                title=s.title,
                start_line=s.start_line,
                end_line=s.end_line,
                level=s.level,
                children=[],
            )
            for s in sections
        ]

        # Build hierarchy using a stack-based approach
        root: List[LiterarySection] = []
        stack: List[LiterarySection] = []

        for node in nodes:
            # Pop stack until we find a parent with lower level
            while stack and stack[-1].level >= node.level:
                stack.pop()

            if stack:
                stack[-1].children.append(node)
            else:
                root.append(node)

            stack.append(node)

        return root

    # ------------------------------------------------------------------
    # Dialogue detection
    # ------------------------------------------------------------------

    def _detect_dialogue(self, text: str, lines: List[str]) -> List[DialogueLine]:
        """Detect dialogue in both prose and dramatic forms."""
        dialogue: List[DialogueLine] = []
        seen: set[tuple[str, int]] = set()

        # Dramatic dialogue (SPEAKER. text)
        for m in DRAMATIC_DIALOGUE_RE.finditer(text):
            speaker = m.group(1).strip().title()
            raw_text = m.group(2).strip()
            # Remove stage directions
            cleaned = STAGE_DIRECTION_RE.sub("", raw_text).strip()
            line_num = text[: m.start()].count("\n")

            key = (speaker, line_num)
            if key not in seen and cleaned:
                dialogue.append(
                    DialogueLine(
                        speaker=speaker,
                        text=cleaned,
                        line_number=line_num,
                    )
                )
                seen.add(key)

        # Prose dialogue: "text," said Speaker.
        for m in PROSE_DIALOGUE_RE.finditer(text):
            raw_text = m.group(1).strip()
            speaker = m.group(2).strip().rstrip(".")
            cleaned = STAGE_DIRECTION_RE.sub("", raw_text).strip()
            line_num = text[: m.start()].count("\n")

            key = (speaker, line_num)
            if key not in seen and cleaned:
                dialogue.append(
                    DialogueLine(
                        speaker=speaker,
                        text=cleaned,
                        line_number=line_num,
                    )
                )
                seen.add(key)

        # Reverse prose: Speaker replied, "text"
        for m in PROSE_DIALOGUE_REV_RE.finditer(text):
            speaker = m.group(1).strip().rstrip(".")
            raw_text = m.group(2).strip()
            cleaned = STAGE_DIRECTION_RE.sub("", raw_text).strip()
            line_num = text[: m.start()].count("\n")

            key = (speaker, line_num)
            if key not in seen and cleaned:
                dialogue.append(
                    DialogueLine(
                        speaker=speaker,
                        text=cleaned,
                        line_number=line_num,
                    )
                )
                seen.add(key)

        # Sort by line number
        dialogue.sort(key=lambda d: d.line_number)
        return dialogue

    # ------------------------------------------------------------------
    # Verse / dramatic detection
    # ------------------------------------------------------------------

    def _detect_verse(self, lines: List[str]) -> bool:
        """Detect if the text is verse (poetry) based on line characteristics."""
        non_empty = [l for l in lines if l.strip()]
        if len(non_empty) < 3:
            return False

        # Filter out structural markers (chapter headings, etc.)
        content_lines = [
            l
            for l in non_empty
            if not re.match(
                r"^(Chapter|Part|Act|Scene|Book|Canto)\s",
                l.strip(),
                re.IGNORECASE,
            )
            and len(l.strip()) > 2
        ]
        if len(content_lines) < 3:
            return False

        # Verse heuristics:
        # 1. Short average line length (< 55 chars)
        # 2. Many lines don't end with sentence-ending punctuation
        # 3. Lines with dialogue quotes reduce verse confidence

        avg_len = sum(len(l.strip()) for l in content_lines) / len(content_lines)
        lines_without_period = sum(
            1
            for l in content_lines
            if not l.strip().endswith(
                (".", "!", "?", '."', '!"', '?"', '."', '!"', '?"')
            )
        )
        period_ratio = lines_without_period / len(content_lines)

        # Check for prose indicators: dialogue quotes, long sentences
        has_dialogue_quotes = sum(
            1 for l in content_lines if '"' in l or '"' in l or '"' in l
        )
        dialogue_ratio = has_dialogue_quotes / len(content_lines)

        # Long lines (> 70 chars) are prose indicators
        long_lines = sum(1 for l in content_lines if len(l.strip()) > 70)
        long_line_ratio = long_lines / len(content_lines)

        # Score-based detection
        score = 0.0
        if avg_len < 55:
            score += 0.35
        if avg_len < 40:
            score += 0.2
        if period_ratio > 0.5:
            score += 0.25

        # Prose penalties
        if dialogue_ratio > 0.2:
            score -= 0.3
        if long_line_ratio > 0.1:
            score -= 0.3

        return score >= 0.6

    def _detect_dramatic(self, text: str, lines: List[str]) -> bool:
        """Detect if the text is dramatic (play) form."""
        # Count lines matching SPEAKER. text pattern
        dramatic_count = 0
        non_empty = [l for l in lines if l.strip()]

        for line in non_empty:
            stripped = line.strip()
            if DRAMATIC_DIALOGUE_RE.match(stripped):
                dramatic_count += 1

        if not non_empty:
            return False

        # Check for act/scene markers too
        has_acts = bool(
            re.search(r"^Act\s+[IVXLCDMivxlcdm\d]+", text, re.MULTILINE | re.IGNORECASE)
        )

        ratio = dramatic_count / len(non_empty)

        # If high ratio of dramatic lines or acts + some dramatic lines
        return ratio > 0.2 or (has_acts and dramatic_count >= 2)

    # ------------------------------------------------------------------
    # Form estimation
    # ------------------------------------------------------------------

    def _estimate_form(
        self,
        text: str,
        lines: List[str],
        sections: List[LiterarySection],
        dialogue: List[DialogueLine],
        is_verse: bool,
        is_dramatic: bool,
    ) -> str:
        """Estimate the literary form of the text."""
        word_count = len(text.split())

        if is_dramatic:
            return "play"

        if is_verse:
            if word_count > 5000:
                return "epic_poem"
            return "poem"

        # Prose length thresholds
        if word_count > 10000:
            return "novel"
        elif word_count > 3000:
            return "novella"
        else:
            return "short_story"
