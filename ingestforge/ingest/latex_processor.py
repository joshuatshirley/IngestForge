"""LaTeX document processor.

Processes LaTeX (.tex) files, extracting text, structure, equations,
citations, and figures."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Equation:
    """Represents a mathematical equation in LaTeX."""

    latex: str
    label: Optional[str] = None
    position: int = 0
    is_inline: bool = False
    environment: str = "equation"


@dataclass
class Citation:
    """Represents a citation reference."""

    key: str
    position: int = 0
    command: str = "cite"  # cite, citep, citet, etc.


@dataclass
class Figure:
    """Represents a figure in LaTeX."""

    caption: str
    label: Optional[str] = None
    filename: Optional[str] = None
    position: int = 0


@dataclass
class BibEntry:
    """Represents a bibliography entry."""

    key: str
    entry_type: str = "misc"
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[str] = None
    raw: str = ""


@dataclass
class Section:
    """Represents a document section."""

    level: str
    title: str
    position: int = 0


@dataclass
class LaTeXDocument:
    """Represents a processed LaTeX document."""

    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    date: str = ""
    document_class: str = ""
    sections: List[Section] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    bibliography: List[BibEntry] = field(default_factory=list)
    packages: List[str] = field(default_factory=list)
    text: str = ""


# ============================================================================
# Processor Class
# ============================================================================


class LaTeXProcessor:
    """Process LaTeX document files.

    Extracts text, metadata, structure, equations, citations, and figures
    from LaTeX source files.
    """

    # Regex patterns compiled at class level for performance
    TITLE_PATTERN = re.compile(r"\\title\{([^}]+)\}")
    AUTHOR_PATTERN = re.compile(r"\\author\{([^}]+)\}")
    DATE_PATTERN = re.compile(r"\\date\{([^}]+)\}")
    DOCCLASS_PATTERN = re.compile(r"\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}")
    PACKAGE_PATTERN = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}")
    ABSTRACT_PATTERN = re.compile(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL
    )
    DOCUMENT_PATTERN = re.compile(
        r"\\begin\{document\}(.*?)\\end\{document\}", re.DOTALL
    )

    def process(self, file_path: Path) -> Dict[str, Any]:
        """Process LaTeX file.

        Args:
            file_path: Path to .tex file

        Returns:
            Dictionary with extracted content and metadata
        """
        latex_content = self._read_file(file_path)

        # Build document structure
        document = self._build_document(latex_content, file_path)

        return {
            "text": document.text,
            "metadata": self._build_metadata(document, file_path),
            "structure": [
                {"level": s.level, "title": s.title, "position": s.position}
                for s in document.sections
            ],
            "equations": [
                {"latex": e.latex, "label": e.label, "is_inline": e.is_inline}
                for e in document.equations
            ],
            "citations": [c.key for c in document.citations],
            "figures": [
                {"caption": f.caption, "label": f.label, "filename": f.filename}
                for f in document.figures
            ],
            "type": "latex",
            "source": str(file_path.name),
        }

    def process_to_document(self, file_path: Path) -> LaTeXDocument:
        """Process LaTeX file to structured document.

        Args:
            file_path: Path to .tex file

        Returns:
            LaTeXDocument with all extracted content
        """
        latex_content = self._read_file(file_path)
        return self._build_document(latex_content, file_path)

    def _read_file(self, file_path: Path) -> str:
        """Read file with proper encoding handling."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except OSError as e:
            logger.error(f"Failed to read LaTeX file {file_path}: {e}")
            raise

    def _build_document(self, latex_content: str, file_path: Path) -> LaTeXDocument:
        """Build complete document structure.

        Rule #4: Orchestrator function - delegates to extractors
        """
        document = LaTeXDocument()

        # Extract metadata
        self._extract_document_metadata(latex_content, document)

        # Extract packages
        document.packages = self._extract_packages(latex_content)

        # Extract abstract
        document.abstract = self._extract_abstract(latex_content)

        # Extract structure and content
        document.sections = self._extract_structure(latex_content)
        document.equations = self._extract_equations(latex_content)
        document.citations = self._extract_citations(latex_content)
        document.figures = self._extract_figures(latex_content)
        document.bibliography = self._extract_bibliography(latex_content)

        # Extract text content
        document.text = self._extract_text(latex_content)

        return document

    def _build_metadata(
        self, document: LaTeXDocument, file_path: Path
    ) -> Dict[str, Any]:
        """Build metadata dictionary from document."""
        metadata: Dict[str, Any] = {"filename": file_path.name}

        if document.title:
            metadata["title"] = document.title
        if document.authors:
            metadata["author"] = ", ".join(document.authors)
        if document.date:
            metadata["date"] = document.date
        if document.document_class:
            metadata["document_class"] = document.document_class
        if document.abstract:
            metadata["abstract"] = document.abstract
        if document.packages:
            metadata["packages"] = document.packages

        return metadata

    # ========================================================================
    # Metadata Extraction
    # ========================================================================

    def _extract_document_metadata(
        self, latex_content: str, document: LaTeXDocument
    ) -> None:
        """Extract title, author, date, and document class.

        Rule #1: Early returns not needed - simple assignments
        """
        title_match = self.TITLE_PATTERN.search(latex_content)
        if title_match:
            document.title = self._clean_latex_text(title_match.group(1))

        author_match = self.AUTHOR_PATTERN.search(latex_content)
        if author_match:
            document.authors = self._parse_authors(author_match.group(1))

        date_match = self.DATE_PATTERN.search(latex_content)
        if date_match:
            document.date = self._clean_latex_text(date_match.group(1))

        docclass_match = self.DOCCLASS_PATTERN.search(latex_content)
        if docclass_match:
            document.document_class = docclass_match.group(1)

    def _parse_authors(self, author_text: str) -> List[str]:
        """Parse author text into list of authors.

        Handles various formats: \\and separator, comma-separated.
        """
        cleaned = self._clean_latex_text(author_text)

        # Split by \and first
        if "\\and" in author_text:
            parts = re.split(r"\\and", author_text)
            return [self._clean_latex_text(p).strip() for p in parts if p.strip()]

        # Try comma separation if single author doesn't look right
        if "," in cleaned and len(cleaned) > 50:
            return [p.strip() for p in cleaned.split(",") if p.strip()]

        return [cleaned] if cleaned else []

    def _extract_packages(self, latex_content: str) -> List[str]:
        """Extract used packages from preamble."""
        packages: List[str] = []

        for match in self.PACKAGE_PATTERN.finditer(latex_content):
            package_list = match.group(1)
            # Handle comma-separated packages
            for pkg in package_list.split(","):
                pkg = pkg.strip()
                if pkg:
                    packages.append(pkg)

        return packages

    def _extract_abstract(self, latex_content: str) -> str:
        """Extract abstract from document."""
        match = self.ABSTRACT_PATTERN.search(latex_content)
        if not match:
            return ""

        return self._clean_latex_text(match.group(1))

    # ========================================================================
    # Content Extraction
    # ========================================================================

    def _extract_metadata(self, latex_content: str, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from LaTeX (legacy compatibility).

        Args:
            latex_content: LaTeX source code
            file_path: Path to file

        Returns:
            Metadata dictionary
        """
        metadata: Dict[str, Any] = {"filename": file_path.name}

        title_match = self.TITLE_PATTERN.search(latex_content)
        if title_match:
            metadata["title"] = self._clean_latex_text(title_match.group(1))

        author_match = self.AUTHOR_PATTERN.search(latex_content)
        if author_match:
            metadata["author"] = self._clean_latex_text(author_match.group(1))

        date_match = self.DATE_PATTERN.search(latex_content)
        if date_match:
            metadata["date"] = self._clean_latex_text(date_match.group(1))

        docclass_match = self.DOCCLASS_PATTERN.search(latex_content)
        if docclass_match:
            metadata["document_class"] = docclass_match.group(1)

        return metadata

    def _extract_text(self, latex_content: str) -> str:
        """Extract text content from LaTeX.

        Args:
            latex_content: LaTeX source code

        Returns:
            Plain text content
        """
        doc_match = self.DOCUMENT_PATTERN.search(latex_content)
        body = doc_match.group(1) if doc_match else latex_content

        # Remove environments and commands
        text = self._remove_latex_environments(body)
        text = self._remove_latex_commands(text)
        text = self._clean_latex_text(text)

        return text

    # ========================================================================
    # Equation Extraction
    # ========================================================================

    def _extract_equations(self, latex_content: str) -> List[Equation]:
        """Extract all equations from LaTeX content.

        Handles both display and inline math.
        """
        equations: List[Equation] = []

        # Extract display equations
        equations.extend(self._extract_display_equations(latex_content))

        # Extract inline equations
        equations.extend(self._extract_inline_equations(latex_content))

        # Sort by position
        return sorted(equations, key=lambda e: e.position)

    def _extract_display_equations(self, latex_content: str) -> List[Equation]:
        """Extract display math environments."""
        equations: List[Equation] = []

        # Pattern for numbered environments
        env_pattern = re.compile(
            r"\\begin\{(equation|align|gather|multline)\*?\}"
            r"(.*?)"
            r"\\end\{\1\*?\}",
            re.DOTALL,
        )

        for match in env_pattern.finditer(latex_content):
            env_name = match.group(1)
            content = match.group(2).strip()

            # Extract label if present
            label = self._extract_label(content)

            equations.append(
                Equation(
                    latex=content,
                    label=label,
                    position=match.start(),
                    is_inline=False,
                    environment=env_name,
                )
            )

        # Add \[ \] display math
        display_pattern = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
        for match in display_pattern.finditer(latex_content):
            equations.append(
                Equation(
                    latex=match.group(1).strip(),
                    position=match.start(),
                    is_inline=False,
                    environment="displaymath",
                )
            )

        return equations

    def _extract_inline_equations(self, latex_content: str) -> List[Equation]:
        """Extract inline math ($...$)."""
        equations: List[Equation] = []

        # Match $...$ but not $$...$$
        inline_pattern = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

        for match in inline_pattern.finditer(latex_content):
            content = match.group(1).strip()
            if content:  # Skip empty matches
                equations.append(
                    Equation(
                        latex=content,
                        position=match.start(),
                        is_inline=True,
                        environment="inline",
                    )
                )

        return equations

    def _extract_label(self, content: str) -> Optional[str]:
        """Extract label from equation content."""
        label_match = re.search(r"\\label\{([^}]+)\}", content)
        return label_match.group(1) if label_match else None

    # ========================================================================
    # Citation Extraction
    # ========================================================================

    def _extract_citations(self, latex_content: str) -> List[Citation]:
        """Extract all citation references."""
        citations: List[Citation] = []

        # Pattern for various cite commands
        cite_pattern = re.compile(
            r"\\(cite[pt]?|citep|citet|citeauthor|citeyear|"
            r"textcite|parencite|autocite)\{([^}]+)\}"
        )

        for match in cite_pattern.finditer(latex_content):
            command = match.group(1)
            keys = match.group(2)

            # Handle multiple keys in single cite
            for key in keys.split(","):
                key = key.strip()
                if key:
                    citations.append(
                        Citation(
                            key=key,
                            position=match.start(),
                            command=command,
                        )
                    )

        return citations

    # ========================================================================
    # Figure Extraction
    # ========================================================================

    def _extract_figures(self, latex_content: str) -> List[Figure]:
        """Extract all figure environments."""
        figures: List[Figure] = []

        figure_pattern = re.compile(
            r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.DOTALL
        )

        for match in figure_pattern.finditer(latex_content):
            content = match.group(1)
            figure = self._parse_figure_content(content, match.start())
            figures.append(figure)

        return figures

    def _parse_figure_content(self, content: str, position: int) -> Figure:
        """Parse figure environment content.

        Rule #1: Early returns for missing elements
        """
        # Extract caption
        caption_match = re.search(r"\\caption\{([^}]+)\}", content)
        caption = (
            self._clean_latex_text(caption_match.group(1)) if caption_match else ""
        )

        # Extract label
        label_match = re.search(r"\\label\{([^}]+)\}", content)
        label = label_match.group(1) if label_match else None

        # Extract filename from includegraphics
        filename_match = re.search(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content
        )
        filename = filename_match.group(1) if filename_match else None

        return Figure(
            caption=caption,
            label=label,
            filename=filename,
            position=position,
        )

    # ========================================================================
    # Bibliography Extraction
    # ========================================================================

    def _extract_bibliography(self, latex_content: str) -> List[BibEntry]:
        """Extract bibliography entries from thebibliography environment."""
        entries: List[BibEntry] = []

        # thebibliography environment
        bib_pattern = re.compile(
            r"\\begin\{thebibliography\}.*?" r"(.*?)" r"\\end\{thebibliography\}",
            re.DOTALL,
        )

        match = bib_pattern.search(latex_content)
        if not match:
            return entries

        bib_content = match.group(1)

        # Extract bibitem entries
        item_pattern = re.compile(
            r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|$)", re.DOTALL
        )

        for item_match in item_pattern.finditer(bib_content):
            key = item_match.group(1)
            raw = item_match.group(2).strip()

            entries.append(
                BibEntry(
                    key=key,
                    raw=self._clean_latex_text(raw),
                )
            )

        return entries

    # ========================================================================
    # Structure Extraction
    # ========================================================================

    def _extract_structure(self, latex_content: str) -> List[Section]:
        """Extract document structure (sections).

        Args:
            latex_content: LaTeX source code

        Returns:
            List of structure elements
        """
        structure: List[Section] = []

        section_pattern = re.compile(
            r"\\(section|subsection|subsubsection|chapter|part)\*?\{([^}]+)\}"
        )

        for match in section_pattern.finditer(latex_content):
            level = match.group(1)
            title = self._clean_latex_text(match.group(2))

            structure.append(
                Section(
                    level=level,
                    title=title,
                    position=match.start(),
                )
            )

        return structure

    # ========================================================================
    # Environment and Command Removal
    # ========================================================================

    def _remove_latex_environments(self, text: str) -> str:
        """Remove LaTeX environments.

        Args:
            text: LaTeX text

        Returns:
            Text with environments removed
        """
        # Remove equation environments with placeholder
        text = re.sub(
            r"\\begin\{(?:equation|align|gather|multline)\*?\}"
            r".*?"
            r"\\end\{(?:equation|align|gather|multline)\*?\}",
            "[EQUATION]",
            text,
            flags=re.DOTALL,
        )

        # Remove figure environments with placeholder
        text = re.sub(
            r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",
            "[FIGURE]",
            text,
            flags=re.DOTALL,
        )

        # Remove table environments with placeholder
        text = re.sub(
            r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "[TABLE]", text, flags=re.DOTALL
        )

        # Remove display math
        text = re.sub(r"\\\[.*?\\\]", "[EQUATION]", text, flags=re.DOTALL)
        text = re.sub(r"\$\$.*?\$\$", "[EQUATION]", text, flags=re.DOTALL)

        # Remove other environments but keep content
        text = re.sub(r"\\begin\{[^}]+\}", "", text)
        text = re.sub(r"\\end\{[^}]+\}", "", text)

        return text

    def _remove_latex_commands(self, text: str) -> str:
        """Remove LaTeX commands.

        Args:
            text: LaTeX text

        Returns:
            Text with commands removed
        """
        # Keep text from formatting commands
        text = re.sub(
            r"\\(?:textbf|textit|emph|underline|texttt|textrm|textsf)\{([^}]+)\}",
            r"\1",
            text,
        )

        # Keep section titles with formatting
        text = re.sub(
            r"\\(?:section|subsection|subsubsection|chapter|part)\*?\{([^}]+)\}",
            r"\n\n\1\n\n",
            text,
        )

        # Remove citation commands
        text = re.sub(r"\\cite[pt]?\{[^}]+\}", "[CITATION]", text)

        # Remove other commands with arguments
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)

        # Remove standalone commands
        text = re.sub(r"\\[a-zA-Z]+", "", text)

        # Remove comments
        text = re.sub(r"%.*?$", "", text, flags=re.MULTILINE)

        return text

    def _clean_latex_text(self, text: str) -> str:
        """Clean LaTeX text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove special characters
        text = text.replace("~", " ")

        # Remove curly braces
        text = text.replace("{", "").replace("}", "")

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\n\n+", "\n\n", text)

        return text.strip()


# ============================================================================
# Module-level Functions
# ============================================================================


def extract_text(file_path: Path) -> str:
    """Extract text from LaTeX file.

    Args:
        file_path: Path to .tex file

    Returns:
        Extracted text content
    """
    processor = LaTeXProcessor()
    result = processor.process(file_path)
    return cast(str, result.get("text", ""))


def extract_with_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract text and metadata from LaTeX file.

    Args:
        file_path: Path to .tex file

    Returns:
        Dictionary with text and metadata
    """
    processor = LaTeXProcessor()
    return processor.process(file_path)


def process_to_document(file_path: Path) -> LaTeXDocument:
    """Process LaTeX file to structured document.

    Args:
        file_path: Path to .tex file

    Returns:
        LaTeXDocument with all extracted content
    """
    processor = LaTeXProcessor()
    return processor.process_to_document(file_path)
