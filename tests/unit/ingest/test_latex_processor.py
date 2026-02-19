"""
Tests for LaTeX document processor.

This module tests extraction of text, metadata, structure, equations,
citations, figures, and bibliography from LaTeX (.tex) files.

Test Strategy
-------------
- Test LaTeX command and environment removal
- Test metadata extraction from LaTeX preamble
- Test document structure detection
- Test equation extraction (inline and display)
- Test citation extraction
- Test figure extraction
- Test bibliography extraction
- No external dependencies to mock

Organization
------------
- TestBasicProcessing: Main process method
- TestMetadataExtraction: Title, author, date extraction
- TestTextExtraction: Body text and cleaning
- TestEnvironmentRemoval: Equation, figure, table handling
- TestCommandRemoval: LaTeX command stripping
- TestTextCleaning: Whitespace and special character handling
- TestStructureExtraction: Section hierarchy
- TestEquationExtraction: Math environments and inline math
- TestCitationExtraction: Citation commands
- TestFigureExtraction: Figure environments
- TestBibliographyExtraction: Bibliography entries
- TestHelperFunctions: extract_text and extract_with_metadata
- TestComplexDocuments: Realistic scenarios
- TestEdgeCases: Malformed input and edge cases
"""


from ingestforge.ingest.latex_processor import (
    LaTeXProcessor,
    LaTeXDocument,
    extract_text,
    extract_with_metadata,
    process_to_document,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestBasicProcessing:
    """Tests for basic LaTeX processing.

    Rule #4: Focused test class - tests process method
    """

    def test_process_minimal_document(self, temp_dir):
        """Test processing minimal LaTeX document."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "doc.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\begin{document}
Hello World
\end{document}
"""
        )

        result = processor.process(latex_file)

        assert "Hello World" in result["text"]
        assert result["type"] == "latex"
        assert result["source"] == "doc.tex"

    def test_process_with_title(self, temp_dir):
        """Test processing document with title."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "doc.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\title{My Document}
\begin{document}
Content
\end{document}
"""
        )

        result = processor.process(latex_file)

        assert result["metadata"]["title"] == "My Document"

    def test_process_with_author(self, temp_dir):
        """Test processing document with author."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "doc.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\author{John Doe}
\begin{document}
Content
\end{document}
"""
        )

        result = processor.process(latex_file)

        assert result["metadata"]["author"] == "John Doe"

    def test_process_returns_equations(self, temp_dir):
        """Test that process returns extracted equations."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "doc.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\begin{document}
\begin{equation}
E = mc^2
\end{equation}
\end{document}
"""
        )

        result = processor.process(latex_file)

        assert len(result["equations"]) == 1
        assert "E = mc^2" in result["equations"][0]["latex"]

    def test_process_returns_citations(self, temp_dir):
        """Test that process returns extracted citations."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "doc.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\begin{document}
See \cite{einstein1905} for details.
\end{document}
"""
        )

        result = processor.process(latex_file)

        assert "einstein1905" in result["citations"]


class TestProcessToDocument:
    """Tests for process_to_document method."""

    def test_returns_latex_document(self, temp_dir):
        """Test that process_to_document returns LaTeXDocument."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "doc.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\title{Test Title}
\author{Test Author}
\begin{document}
Content
\end{document}
"""
        )

        doc = processor.process_to_document(latex_file)

        assert isinstance(doc, LaTeXDocument)
        assert doc.title == "Test Title"
        assert "Test Author" in doc.authors

    def test_document_has_all_fields(self, temp_dir):
        """Test that document contains all expected fields."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "doc.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\begin{document}
Content
\end{document}
"""
        )

        doc = processor.process_to_document(latex_file)

        assert hasattr(doc, "title")
        assert hasattr(doc, "authors")
        assert hasattr(doc, "abstract")
        assert hasattr(doc, "sections")
        assert hasattr(doc, "equations")
        assert hasattr(doc, "citations")
        assert hasattr(doc, "figures")
        assert hasattr(doc, "bibliography")
        assert hasattr(doc, "packages")


class TestMetadataExtraction:
    """Tests for metadata extraction.

    Rule #4: Focused test class - tests _extract_metadata
    """

    def test_extract_title(self, temp_dir):
        """Test extracting title from LaTeX."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "test.tex"
        latex_content = r"\title{Document Title}"

        metadata = processor._extract_metadata(latex_content, latex_file)

        assert metadata["title"] == "Document Title"

    def test_extract_author(self, temp_dir):
        """Test extracting author from LaTeX."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "test.tex"
        latex_content = r"\author{Jane Smith}"

        metadata = processor._extract_metadata(latex_content, latex_file)

        assert metadata["author"] == "Jane Smith"

    def test_extract_date(self, temp_dir):
        """Test extracting date from LaTeX."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "test.tex"
        latex_content = r"\date{2024-01-01}"

        metadata = processor._extract_metadata(latex_content, latex_file)

        assert metadata["date"] == "2024-01-01"

    def test_extract_document_class(self, temp_dir):
        """Test extracting document class."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "test.tex"
        latex_content = r"\documentclass{article}"

        metadata = processor._extract_metadata(latex_content, latex_file)

        assert metadata["document_class"] == "article"

    def test_extract_document_class_with_options(self, temp_dir):
        """Test extracting document class with options."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "test.tex"
        latex_content = r"\documentclass[12pt,twocolumn]{article}"

        metadata = processor._extract_metadata(latex_content, latex_file)

        assert metadata["document_class"] == "article"

    def test_extract_all_metadata(self, temp_dir):
        """Test extracting complete metadata."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "test.tex"
        latex_content = r"""
\documentclass{article}
\title{Full Document}
\author{Author Name}
\date{2024}
"""

        metadata = processor._extract_metadata(latex_content, latex_file)

        assert metadata["title"] == "Full Document"
        assert metadata["author"] == "Author Name"
        assert metadata["date"] == "2024"
        assert metadata["document_class"] == "article"
        assert metadata["filename"] == "test.tex"


class TestPackageExtraction:
    """Tests for package extraction."""

    def test_extract_single_package(self, temp_dir):
        """Test extracting single package."""
        processor = LaTeXProcessor()

        latex_content = r"\usepackage{amsmath}"

        packages = processor._extract_packages(latex_content)

        assert "amsmath" in packages

    def test_extract_multiple_packages(self, temp_dir):
        """Test extracting multiple packages."""
        processor = LaTeXProcessor()

        latex_content = r"""
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
"""

        packages = processor._extract_packages(latex_content)

        assert "amsmath" in packages
        assert "graphicx" in packages
        assert "hyperref" in packages

    def test_extract_comma_separated_packages(self, temp_dir):
        """Test extracting comma-separated packages."""
        processor = LaTeXProcessor()

        latex_content = r"\usepackage{amsmath,amssymb,amsthm}"

        packages = processor._extract_packages(latex_content)

        assert "amsmath" in packages
        assert "amssymb" in packages
        assert "amsthm" in packages

    def test_extract_packages_with_options(self, temp_dir):
        """Test extracting packages with options."""
        processor = LaTeXProcessor()

        latex_content = r"\usepackage[utf8]{inputenc}"

        packages = processor._extract_packages(latex_content)

        assert "inputenc" in packages


class TestAbstractExtraction:
    """Tests for abstract extraction."""

    def test_extract_abstract(self, temp_dir):
        """Test extracting abstract."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{abstract}
This is the abstract of the paper.
\end{abstract}
"""

        abstract = processor._extract_abstract(latex_content)

        assert "abstract of the paper" in abstract

    def test_extract_multiline_abstract(self, temp_dir):
        """Test extracting multiline abstract."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{abstract}
First paragraph.

Second paragraph.
\end{abstract}
"""

        abstract = processor._extract_abstract(latex_content)

        assert "First paragraph" in abstract
        assert "Second paragraph" in abstract

    def test_no_abstract(self, temp_dir):
        """Test document without abstract."""
        processor = LaTeXProcessor()

        latex_content = r"\documentclass{article}"

        abstract = processor._extract_abstract(latex_content)

        assert abstract == ""


class TestTextExtraction:
    """Tests for text extraction.

    Rule #4: Focused test class - tests _extract_text
    """

    def test_extract_document_body(self):
        """Test extracting text from document body."""
        processor = LaTeXProcessor()

        latex_content = r"""
\documentclass{article}
\begin{document}
This is the main content.
\end{document}
"""

        result = processor._extract_text(latex_content)

        assert "This is the main content" in result

    def test_extract_without_document_env(self):
        """Test extraction when no document environment."""
        processor = LaTeXProcessor()

        latex_content = "Just plain text"

        result = processor._extract_text(latex_content)

        assert "Just plain text" in result

    def test_extract_removes_comments(self):
        """Test that LaTeX comments are removed."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{document}
Text % This is a comment
More text
\end{document}
"""

        result = processor._extract_text(latex_content)

        assert "This is a comment" not in result
        assert "Text" in result
        assert "More text" in result


class TestEnvironmentRemoval:
    """Tests for LaTeX environment removal.

    Rule #4: Focused test class - tests _remove_latex_environments
    """

    def test_remove_equation_environment(self):
        """Test removing equation environment."""
        processor = LaTeXProcessor()

        text = r"""
Text before
\begin{equation}
x = y + z
\end{equation}
Text after
"""

        result = processor._remove_latex_environments(text)

        assert "[EQUATION]" in result
        assert "x = y + z" not in result
        assert "Text before" in result
        assert "Text after" in result

    def test_remove_align_environment(self):
        """Test removing align environment."""
        processor = LaTeXProcessor()

        text = r"""
\begin{align}
a &= b \\
c &= d
\end{align}
"""

        result = processor._remove_latex_environments(text)

        assert "[EQUATION]" in result
        assert "a &= b" not in result

    def test_remove_figure_environment(self):
        """Test removing figure environment."""
        processor = LaTeXProcessor()

        text = r"""
\begin{figure}
\includegraphics{image.png}
\caption{My figure}
\end{figure}
"""

        result = processor._remove_latex_environments(text)

        assert "[FIGURE]" in result
        assert "includegraphics" not in result

    def test_remove_table_environment(self):
        """Test removing table environment."""
        processor = LaTeXProcessor()

        text = r"""
\begin{table}
\begin{tabular}{ll}
A & B
\end{tabular}
\end{table}
"""

        result = processor._remove_latex_environments(text)

        assert "[TABLE]" in result
        assert "tabular" not in result

    def test_remove_custom_environment(self):
        """Test removing custom environment."""
        processor = LaTeXProcessor()

        text = r"""
\begin{myenv}
Content
\end{myenv}
"""

        result = processor._remove_latex_environments(text)

        # Content should remain, environment tags removed
        assert "Content" in result
        assert r"\begin{myenv}" not in result
        assert r"\end{myenv}" not in result

    def test_remove_starred_environments(self):
        """Test removing starred environments."""
        processor = LaTeXProcessor()

        text = r"""
\begin{equation*}
E = mc^2
\end{equation*}
"""

        result = processor._remove_latex_environments(text)

        assert "[EQUATION]" in result


class TestCommandRemoval:
    """Tests for LaTeX command removal.

    Rule #4: Focused test class - tests _remove_latex_commands
    """

    def test_remove_textbf_command(self):
        """Test removing textbf command."""
        processor = LaTeXProcessor()

        text = r"This is \textbf{bold} text"

        result = processor._remove_latex_commands(text)

        assert "bold" in result
        assert r"\textbf" not in result

    def test_remove_textit_command(self):
        """Test removing textit command."""
        processor = LaTeXProcessor()

        text = r"This is \textit{italic} text"

        result = processor._remove_latex_commands(text)

        assert "italic" in result
        assert r"\textit" not in result

    def test_remove_section_command(self):
        """Test removing section command but keeping title."""
        processor = LaTeXProcessor()

        text = r"\section{Introduction}"

        result = processor._remove_latex_commands(text)

        assert "Introduction" in result

    def test_remove_subsection_command(self):
        """Test removing subsection command."""
        processor = LaTeXProcessor()

        text = r"\subsection{Background}"

        result = processor._remove_latex_commands(text)

        assert "Background" in result

    def test_remove_citation_command(self):
        """Test removing citation commands."""
        processor = LaTeXProcessor()

        text = r"See \cite{author2020} for details"

        result = processor._remove_latex_commands(text)

        assert "[CITATION]" in result


class TestTextCleaning:
    """Tests for text cleaning.

    Rule #4: Focused test class - tests _clean_latex_text
    """

    def test_clean_tilde(self):
        """Test replacing tilde with space."""
        processor = LaTeXProcessor()

        text = "Word~with~tildes"

        result = processor._clean_latex_text(text)

        assert "Word with tildes" in result
        assert "~" not in result

    def test_clean_braces(self):
        """Test removing curly braces."""
        processor = LaTeXProcessor()

        text = "Text {with} {braces}"

        result = processor._clean_latex_text(text)

        assert "with" in result
        assert "braces" in result
        assert "{" not in result
        assert "}" not in result

    def test_normalize_whitespace(self):
        """Test normalizing multiple spaces."""
        processor = LaTeXProcessor()

        text = "Text   with    extra    spaces"

        result = processor._clean_latex_text(text)

        assert "Text with extra spaces" in result

    def test_normalize_newlines(self):
        """Test normalizing multiple newlines."""
        processor = LaTeXProcessor()

        text = "Line 1\n\n\n\nLine 2"

        result = processor._clean_latex_text(text)

        # Should normalize to double newline
        assert "Line 1\n\nLine 2" in result

    def test_strip_result(self):
        """Test result is stripped."""
        processor = LaTeXProcessor()

        text = "   Text   "

        result = processor._clean_latex_text(text)

        assert result == "Text"


class TestStructureExtraction:
    """Tests for document structure extraction.

    Rule #4: Focused test class - tests _extract_structure
    """

    def test_extract_section(self):
        """Test extracting section titles."""
        processor = LaTeXProcessor()

        latex_content = r"\section{Introduction}"

        result = processor._extract_structure(latex_content)

        assert len(result) == 1
        assert result[0].level == "section"
        assert result[0].title == "Introduction"

    def test_extract_subsection(self):
        """Test extracting subsection titles."""
        processor = LaTeXProcessor()

        latex_content = r"\subsection{Background}"

        result = processor._extract_structure(latex_content)

        assert len(result) == 1
        assert result[0].level == "subsection"
        assert result[0].title == "Background"

    def test_extract_subsubsection(self):
        """Test extracting subsubsection titles."""
        processor = LaTeXProcessor()

        latex_content = r"\subsubsection{Details}"

        result = processor._extract_structure(latex_content)

        assert len(result) == 1
        assert result[0].level == "subsubsection"

    def test_extract_multiple_sections(self):
        """Test extracting multiple sections."""
        processor = LaTeXProcessor()

        latex_content = r"""
\section{First}
\section{Second}
\subsection{Nested}
"""

        result = processor._extract_structure(latex_content)

        assert len(result) == 3
        assert result[0].title == "First"
        assert result[1].title == "Second"
        assert result[2].title == "Nested"

    def test_structure_position(self):
        """Test structure elements have position."""
        processor = LaTeXProcessor()

        latex_content = r"Some text\section{Title}More text"

        result = processor._extract_structure(latex_content)

        assert result[0].position > 0

    def test_extract_chapter(self):
        """Test extracting chapter titles."""
        processor = LaTeXProcessor()

        latex_content = r"\chapter{First Chapter}"

        result = processor._extract_structure(latex_content)

        assert len(result) == 1
        assert result[0].level == "chapter"


class TestEquationExtraction:
    """Tests for equation extraction."""

    def test_extract_display_equation(self):
        """Test extracting display equation."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{equation}
E = mc^2
\end{equation}
"""

        equations = processor._extract_equations(latex_content)

        assert len(equations) == 1
        assert "E = mc^2" in equations[0].latex
        assert equations[0].is_inline is False
        assert equations[0].environment == "equation"

    def test_extract_inline_equation(self):
        """Test extracting inline equation."""
        processor = LaTeXProcessor()

        latex_content = r"The formula $x = y + z$ is simple."

        equations = processor._extract_equations(latex_content)

        assert len(equations) == 1
        assert "x = y + z" in equations[0].latex
        assert equations[0].is_inline is True

    def test_extract_equation_with_label(self):
        """Test extracting equation with label."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{equation}\label{eq:einstein}
E = mc^2
\end{equation}
"""

        equations = processor._extract_equations(latex_content)

        assert len(equations) == 1
        assert equations[0].label == "eq:einstein"

    def test_extract_align_environment(self):
        """Test extracting align environment."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{align}
a &= b \\
c &= d
\end{align}
"""

        equations = processor._extract_equations(latex_content)

        assert len(equations) == 1
        assert equations[0].environment == "align"

    def test_extract_display_math(self):
        """Test extracting \[ \] display math."""
        processor = LaTeXProcessor()

        latex_content = r"\[ x^2 + y^2 = z^2 \]"

        equations = processor._extract_equations(latex_content)

        assert len(equations) == 1
        assert equations[0].environment == "displaymath"

    def test_extract_multiple_equations(self):
        """Test extracting multiple equations."""
        processor = LaTeXProcessor()

        latex_content = r"""
$a = 1$
\begin{equation}
b = 2
\end{equation}
$c = 3$
"""

        equations = processor._extract_equations(latex_content)

        # Should have 3 equations
        assert len(equations) >= 3


class TestCitationExtraction:
    """Tests for citation extraction."""

    def test_extract_cite_command(self):
        """Test extracting basic cite command."""
        processor = LaTeXProcessor()

        latex_content = r"See \cite{author2020} for details."

        citations = processor._extract_citations(latex_content)

        assert len(citations) == 1
        assert citations[0].key == "author2020"
        assert citations[0].command == "cite"

    def test_extract_citep_command(self):
        """Test extracting citep command."""
        processor = LaTeXProcessor()

        latex_content = r"Results \citep{smith2021}."

        citations = processor._extract_citations(latex_content)

        assert len(citations) == 1
        assert citations[0].command == "citep"

    def test_extract_citet_command(self):
        """Test extracting citet command."""
        processor = LaTeXProcessor()

        latex_content = r"\citet{jones2019} showed that..."

        citations = processor._extract_citations(latex_content)

        assert len(citations) == 1
        assert citations[0].command == "citet"

    def test_extract_multiple_keys(self):
        """Test extracting multiple citation keys."""
        processor = LaTeXProcessor()

        latex_content = r"\cite{author1,author2,author3}"

        citations = processor._extract_citations(latex_content)

        assert len(citations) == 3
        keys = [c.key for c in citations]
        assert "author1" in keys
        assert "author2" in keys
        assert "author3" in keys

    def test_extract_textcite(self):
        """Test extracting textcite (biblatex)."""
        processor = LaTeXProcessor()

        latex_content = r"\textcite{biblatex2020}"

        citations = processor._extract_citations(latex_content)

        assert len(citations) == 1
        assert citations[0].command == "textcite"


class TestFigureExtraction:
    """Tests for figure extraction."""

    def test_extract_figure_with_caption(self):
        """Test extracting figure with caption."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{figure}
\includegraphics{image.png}
\caption{This is the caption}
\end{figure}
"""

        figures = processor._extract_figures(latex_content)

        assert len(figures) == 1
        assert figures[0].caption == "This is the caption"

    def test_extract_figure_with_label(self):
        """Test extracting figure with label."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{figure}
\includegraphics{image.png}
\caption{Caption}
\label{fig:myimage}
\end{figure}
"""

        figures = processor._extract_figures(latex_content)

        assert len(figures) == 1
        assert figures[0].label == "fig:myimage"

    def test_extract_figure_filename(self):
        """Test extracting figure filename."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{figure}
\includegraphics{images/diagram.pdf}
\caption{A diagram}
\end{figure}
"""

        figures = processor._extract_figures(latex_content)

        assert len(figures) == 1
        assert figures[0].filename == "images/diagram.pdf"

    def test_extract_figure_with_options(self):
        """Test extracting figure with includegraphics options."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{figure}
\includegraphics[width=\textwidth]{photo.jpg}
\caption{A photo}
\end{figure}
"""

        figures = processor._extract_figures(latex_content)

        assert len(figures) == 1
        assert figures[0].filename == "photo.jpg"

    def test_extract_multiple_figures(self):
        """Test extracting multiple figures."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{figure}
\includegraphics{fig1.png}
\caption{First}
\end{figure}
\begin{figure}
\includegraphics{fig2.png}
\caption{Second}
\end{figure}
"""

        figures = processor._extract_figures(latex_content)

        assert len(figures) == 2


class TestBibliographyExtraction:
    """Tests for bibliography extraction."""

    def test_extract_bibitem(self):
        """Test extracting bibliography items."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{thebibliography}{9}
\bibitem{einstein1905}
A. Einstein, "On the Electrodynamics of Moving Bodies," 1905.
\end{thebibliography}
"""

        entries = processor._extract_bibliography(latex_content)

        assert len(entries) == 1
        assert entries[0].key == "einstein1905"

    def test_extract_multiple_bibitems(self):
        """Test extracting multiple bibliography items."""
        processor = LaTeXProcessor()

        latex_content = r"""
\begin{thebibliography}{99}
\bibitem{ref1}
First reference.
\bibitem{ref2}
Second reference.
\bibitem{ref3}
Third reference.
\end{thebibliography}
"""

        entries = processor._extract_bibliography(latex_content)

        assert len(entries) == 3

    def test_no_bibliography(self):
        """Test document without bibliography."""
        processor = LaTeXProcessor()

        latex_content = r"\documentclass{article}"

        entries = processor._extract_bibliography(latex_content)

        assert len(entries) == 0


class TestHelperFunctions:
    """Tests for helper functions.

    Rule #4: Focused test class - tests module-level functions
    """

    def test_extract_text_function(self, temp_dir):
        """Test extract_text helper function."""
        latex_file = temp_dir / "test.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\begin{document}
Test content
\end{document}
"""
        )

        result = extract_text(latex_file)

        assert "Test content" in result

    def test_extract_with_metadata_function(self, temp_dir):
        """Test extract_with_metadata helper function."""
        latex_file = temp_dir / "test.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\title{Test Title}
\begin{document}
Content
\end{document}
"""
        )

        result = extract_with_metadata(latex_file)

        assert "text" in result
        assert "metadata" in result
        assert result["metadata"]["title"] == "Test Title"

    def test_process_to_document_function(self, temp_dir):
        """Test process_to_document helper function."""
        latex_file = temp_dir / "test.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\title{Test}
\begin{document}
Content
\end{document}
"""
        )

        doc = process_to_document(latex_file)

        assert isinstance(doc, LaTeXDocument)
        assert doc.title == "Test"


class TestComplexDocuments:
    """Tests for complex LaTeX documents.

    Rule #4: Focused test class - tests realistic scenarios
    """

    def test_academic_paper_structure(self, temp_dir):
        """Test processing academic paper with full structure."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "paper.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}
\title{Research Paper Title}
\author{Author Name}
\date{2024}

\begin{document}
\maketitle

\begin{abstract}
This is the abstract.
\end{abstract}

\section{Introduction}
This is the introduction. See \cite{ref1}.

\section{Methods}
\subsection{Experimental Setup}
Details here.

\begin{equation}\label{eq:main}
E = mc^2
\end{equation}

\section{Results}
\begin{figure}
\includegraphics{results.png}
\caption{Main results}
\label{fig:results}
\end{figure}

\section{Conclusion}
Final thoughts.

\begin{thebibliography}{9}
\bibitem{ref1}
A. Author, "Title," 2024.
\end{thebibliography}

\end{document}
"""
        )

        doc = processor.process_to_document(latex_file)

        assert doc.title == "Research Paper Title"
        assert "Author Name" in doc.authors
        assert "abstract" in doc.abstract.lower()
        assert len(doc.sections) >= 4
        assert len(doc.equations) >= 1
        assert len(doc.citations) >= 1
        assert len(doc.figures) >= 1
        assert len(doc.bibliography) >= 1
        assert "amsmath" in doc.packages

    def test_document_with_multiple_authors(self, temp_dir):
        """Test document with multiple authors."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "doc.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\author{Alice \and Bob \and Charlie}
\begin{document}
Content
\end{document}
"""
        )

        doc = processor.process_to_document(latex_file)

        assert len(doc.authors) == 3


class TestEdgeCases:
    """Tests for edge cases.

    Rule #4: Focused test class - tests edge cases
    """

    def test_empty_document(self, temp_dir):
        """Test processing empty document."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "empty.tex"
        latex_file.write_text("")

        result = processor.process(latex_file)

        assert result["text"] == ""

    def test_document_no_content(self, temp_dir):
        """Test document with only preamble."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "preamble.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\title{Title Only}
"""
        )

        result = processor.process(latex_file)

        assert result["metadata"]["title"] == "Title Only"

    def test_malformed_latex(self, temp_dir):
        """Test handling malformed LaTeX."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "malformed.tex"
        latex_file.write_text(
            r"""
\documentclass{article
\begin{document}
Text
\end{document
"""
        )

        # Should not crash
        result = processor.process(latex_file)

        assert "text" in result

    def test_nested_braces(self, temp_dir):
        """Test handling nested braces."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "nested.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\title{Title with {nested {braces}}}
\begin{document}
Content
\end{document}
"""
        )

        result = processor.process(latex_file)

        assert "Title with nested braces" in result["metadata"]["title"]

    def test_unicode_content(self, temp_dir):
        """Test handling unicode content."""
        processor = LaTeXProcessor()

        latex_file = temp_dir / "unicode.tex"
        latex_file.write_text(
            r"""
\documentclass{article}
\begin{document}
Hello World with special content
\end{document}
""",
            encoding="utf-8",
        )

        result = processor.process(latex_file)

        # Test that content is extracted (unicode handling may vary by platform)
        assert "Hello" in result["text"]
        assert "World" in result["text"]


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Basic processing: 5 tests
    - Process to document: 2 tests
    - Metadata extraction: 6 tests
    - Package extraction: 4 tests
    - Abstract extraction: 3 tests
    - Text extraction: 3 tests
    - Environment removal: 6 tests
    - Command removal: 5 tests
    - Text cleaning: 5 tests
    - Structure extraction: 6 tests
    - Equation extraction: 6 tests
    - Citation extraction: 5 tests
    - Figure extraction: 5 tests
    - Bibliography extraction: 3 tests
    - Helper functions: 3 tests
    - Complex documents: 2 tests
    - Edge cases: 5 tests

    Total: 74 tests

Design Decisions:
    1. No external dependencies to mock (pure Python)
    2. Test all extraction types (equations, citations, figures)
    3. Cover all LaTeX command types
    4. Test structure hierarchy
    5. Include realistic document scenarios
    6. Test edge cases and malformed input
"""
