"""Jupyter notebook processor.

Processes Jupyter notebook (.ipynb) files, extracting code, markdown,
outputs, imports, and dependencies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CellOutput:
    """Represents output from a code cell."""

    output_type: str  # stream, execute_result, display_data, error
    text: str = ""
    mime_type: str = "text/plain"
    is_error: bool = False


@dataclass
class NotebookCell:
    """Represents a cell in a Jupyter notebook."""

    cell_type: str  # code, markdown, raw
    source: str
    outputs: List[CellOutput] = field(default_factory=list)
    execution_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    index: int = 0


@dataclass
class KernelInfo:
    """Kernel specification information."""

    name: str = ""
    display_name: str = ""
    language: str = "unknown"


@dataclass
class JupyterNotebook:
    """Represents a processed Jupyter notebook."""

    title: str = ""
    kernel: KernelInfo = field(default_factory=KernelInfo)
    cells: List[NotebookCell] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    nbformat: int = 4
    nbformat_minor: int = 0
    total_code_lines: int = 0
    total_markdown_lines: int = 0
    total_code_cells: int = 0
    total_markdown_cells: int = 0


# ============================================================================
# Processor Class
# ============================================================================


class JupyterProcessor:
    """Process Jupyter notebook files.

    Extracts code, markdown, outputs, imports, and statistics
    from Jupyter notebook files.
    """

    # Python import patterns
    IMPORT_PATTERN = re.compile(
        r"^(?:from\s+([\w.]+)\s+)?import\s+(.+?)(?:\s+as\s+\w+)?$", re.MULTILINE
    )
    FROM_IMPORT_PATTERN = re.compile(r"^from\s+([\w.]+)\s+import\s+", re.MULTILINE)

    def process(self, file_path: Path) -> Dict[str, Any]:
        """Process Jupyter notebook file.

        Args:
            file_path: Path to .ipynb file

        Returns:
            Dictionary with extracted content and metadata
        """
        notebook_data = self._read_notebook(file_path)
        notebook = self._build_notebook(notebook_data, file_path)

        return {
            "text": self._build_text_content(notebook.cells),
            "metadata": self._build_metadata(notebook, file_path),
            "cells": [self._cell_to_dict(c) for c in notebook.cells],
            "imports": notebook.imports,
            "type": "jupyter",
            "source": str(file_path.name),
        }

    def process_to_notebook(self, file_path: Path) -> JupyterNotebook:
        """Process Jupyter notebook to structured object.

        Args:
            file_path: Path to .ipynb file

        Returns:
            JupyterNotebook with all extracted content
        """
        notebook_data = self._read_notebook(file_path)
        return self._build_notebook(notebook_data, file_path)

    def _read_notebook(self, file_path: Path) -> Dict[str, Any]:
        """Read and parse notebook JSON."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in notebook {file_path}: {e}")
            raise
        except OSError as e:
            logger.error(f"Failed to read notebook {file_path}: {e}")
            raise

    def _build_notebook(
        self, notebook_data: Dict[str, Any], file_path: Path
    ) -> JupyterNotebook:
        """Build structured notebook from raw data.

        Rule #4: Orchestrator function - delegates to extractors
        """
        notebook = JupyterNotebook()

        # Extract format version
        notebook.nbformat = notebook_data.get("nbformat", 4)
        notebook.nbformat_minor = notebook_data.get("nbformat_minor", 0)

        # Extract kernel info
        nb_metadata = notebook_data.get("metadata", {})
        notebook.kernel = self._extract_kernel_info(nb_metadata)

        # Extract title
        notebook.title = nb_metadata.get("title", file_path.stem)

        # Extract and process cells
        notebook.cells = self._extract_cells(notebook_data)

        # Calculate statistics
        self._calculate_statistics(notebook)

        # Extract imports from code cells
        notebook.imports = self._extract_imports(notebook.cells)

        return notebook

    def _build_metadata(
        self, notebook: JupyterNotebook, file_path: Path
    ) -> Dict[str, Any]:
        """Build metadata dictionary from notebook."""
        return {
            "filename": file_path.name,
            "title": notebook.title,
            "nbformat": notebook.nbformat,
            "language": notebook.kernel.language,
            "kernel": notebook.kernel.name,
            "total_cells": len(notebook.cells),
            "code_cells": notebook.total_code_cells,
            "markdown_cells": notebook.total_markdown_cells,
            "total_code_lines": notebook.total_code_lines,
            "total_markdown_lines": notebook.total_markdown_lines,
            "imports": notebook.imports,
        }

    def _cell_to_dict(self, cell: NotebookCell) -> Dict[str, Any]:
        """Convert NotebookCell to dictionary."""
        return {
            "index": cell.index,
            "type": cell.cell_type,
            "source": cell.source,
            "outputs": [o.text for o in cell.outputs if o.text],
            "execution_count": cell.execution_count,
        }

    # ========================================================================
    # Kernel Info Extraction
    # ========================================================================

    def _extract_kernel_info(self, metadata: Dict[str, Any]) -> KernelInfo:
        """Extract kernel information from metadata."""
        kernelspec = metadata.get("kernelspec", {})
        language_info = metadata.get("language_info", {})

        name = kernelspec.get("name", "")
        display_name = kernelspec.get("display_name", name)

        # Get language from kernelspec first, then language_info
        language = kernelspec.get("language", "")
        if not language:
            language = language_info.get("name", "unknown")

        return KernelInfo(
            name=name,
            display_name=display_name,
            language=language,
        )

    def _extract_metadata(
        self, notebook: Dict[str, Any], file_path: Path
    ) -> Dict[str, Any]:
        """Extract notebook metadata (legacy compatibility).

        Args:
            notebook: Parsed notebook dictionary
            file_path: Path to notebook file

        Returns:
            Metadata dictionary
        """
        nb_metadata = notebook.get("metadata", {})

        metadata: Dict[str, Any] = {
            "filename": file_path.name,
            "nbformat": notebook.get("nbformat"),
            "language": self._get_kernel_language(nb_metadata),
            "kernel": nb_metadata.get("kernelspec", {}).get("name"),
        }

        title = nb_metadata.get("title")
        if title:
            metadata["title"] = title

        return metadata

    def _get_kernel_language(self, metadata: Dict[str, Any]) -> str:
        """Get kernel language from metadata.

        Args:
            metadata: Notebook metadata

        Returns:
            Language name
        """
        kernelspec = metadata.get("kernelspec", {})
        language = kernelspec.get("language", "")

        if not language:
            lang_info = metadata.get("language_info", {})
            language = lang_info.get("name", "unknown")

        return language

    # ========================================================================
    # Cell Extraction
    # ========================================================================

    def _extract_cells(self, notebook: Dict[str, Any]) -> List[NotebookCell]:
        """Extract cells from notebook.

        Args:
            notebook: Parsed notebook dictionary

        Returns:
            List of NotebookCell objects
        """
        cells = notebook.get("cells", [])
        extracted_cells: List[NotebookCell] = []

        for idx, cell in enumerate(cells):
            notebook_cell = self._extract_single_cell(cell, idx)
            extracted_cells.append(notebook_cell)

        return extracted_cells

    def _extract_single_cell(self, cell: Dict[str, Any], idx: int) -> NotebookCell:
        """Extract a single cell.

        Rule #1: Early return not needed - straightforward extraction
        """
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", [])

        # Convert source to string
        source_text = "".join(source) if isinstance(source, list) else str(source)

        # Extract outputs for code cells
        outputs: List[CellOutput] = []
        if cell_type == "code":
            outputs = self._extract_outputs(cell.get("outputs", []))

        return NotebookCell(
            cell_type=cell_type,
            source=source_text,
            outputs=outputs,
            execution_count=cell.get("execution_count"),
            metadata=cell.get("metadata", {}),
            index=idx,
        )

    # ========================================================================
    # Output Extraction
    # ========================================================================

    def _extract_outputs(self, outputs: List[Any]) -> List[CellOutput]:
        """Extract outputs from code cell.

        Args:
            outputs: List of output dictionaries

        Returns:
            List of CellOutput objects
        """
        extracted: List[CellOutput] = []

        for output in outputs:
            cell_output = self._extract_single_output(output)
            if cell_output:
                extracted.append(cell_output)

        return extracted

    def _extract_single_output(self, output: Dict[str, Any]) -> Optional[CellOutput]:
        """Extract single output to CellOutput object.

        Args:
            output: Output dictionary

        Returns:
            CellOutput or None
        """
        output_type = output.get("output_type", "")

        if output_type == "stream":
            return self._extract_stream_output(output)
        if output_type in ("execute_result", "display_data"):
            return self._extract_result_output(output)
        if output_type == "error":
            return self._extract_error_output(output)

        return None

    def _extract_stream_output(self, output: Dict[str, Any]) -> Optional[CellOutput]:
        """Extract stream output."""
        text = output.get("text", [])
        text_str = (
            "".join(text) if isinstance(text, list) else str(text) if text else None
        )

        if not text_str:
            return None

        return CellOutput(
            output_type="stream",
            text=text_str,
            mime_type="text/plain",
        )

    def _extract_result_output(self, output: Dict[str, Any]) -> Optional[CellOutput]:
        """Extract execution result or display data."""
        data = output.get("data", {})

        # Try different MIME types in order of preference
        for mime_type in ["text/plain", "text/html", "text/markdown"]:
            if mime_type not in data:
                continue

            text = data[mime_type]
            text_str = "".join(text) if isinstance(text, list) else str(text)

            return CellOutput(
                output_type=output.get("output_type", "execute_result"),
                text=text_str,
                mime_type=mime_type,
            )

        return None

    def _extract_error_output(self, output: Dict[str, Any]) -> Optional[CellOutput]:
        """Extract error traceback."""
        traceback = output.get("traceback", [])
        if not traceback:
            return None

        # Clean ANSI escape codes from traceback
        text = "\n".join(traceback)
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)

        return CellOutput(
            output_type="error",
            text=text,
            is_error=True,
        )

    # ========================================================================
    # Statistics Calculation
    # ========================================================================

    def _calculate_statistics(self, notebook: JupyterNotebook) -> None:
        """Calculate notebook statistics.

        Rule #1: Simple loop with no nesting
        """
        for cell in notebook.cells:
            lines = cell.source.count("\n") + 1 if cell.source else 0

            if cell.cell_type == "code":
                notebook.total_code_cells += 1
                notebook.total_code_lines += lines
            elif cell.cell_type == "markdown":
                notebook.total_markdown_cells += 1
                notebook.total_markdown_lines += lines

    # ========================================================================
    # Import Extraction
    # ========================================================================

    def _extract_imports(self, cells: List[NotebookCell]) -> List[str]:
        """Extract import statements from code cells.

        Returns unique module names that are imported.
        """
        imports: set[str] = set()

        for cell in cells:
            if cell.cell_type != "code":
                continue

            cell_imports = self._extract_imports_from_code(cell.source)
            imports.update(cell_imports)

        return sorted(imports)

    def _extract_imports_from_code(self, code: str) -> List[str]:
        """Extract imports from a code string.

        Rule #1: Max 3 nesting levels via early continue.
        Rule #4: Function under 60 lines
        """
        imports: List[str] = []

        for line in code.split("\n"):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Match 'from X import ...'
            from_match = re.match(r"^from\s+([\w.]+)\s+import", line)
            if from_match:
                module = from_match.group(1).split(".")[0]
                imports.append(module)
                continue

            # Match 'import X, Y, Z' or 'import X as alias'
            import_match = re.match(r"^import\s+(.+)$", line)
            if not import_match:
                continue
            imports.extend(self._parse_import_statement(import_match.group(1)))

        return imports

    def _parse_import_statement(self, import_part: str) -> List[str]:
        """Parse import statement into module names.

        Rule #1: Extracted to reduce nesting in _extract_imports_from_code.

        Args:
            import_part: The part after 'import' keyword

        Returns:
            List of module names
        """
        modules = []
        for item in import_part.split(","):
            # Handle 'module as alias'
            module = item.split(" as ")[0].strip()
            # Get top-level module
            module = module.split(".")[0]
            if module:
                modules.append(module)
        return modules

    # ========================================================================
    # Text Building
    # ========================================================================

    def _build_text_content(self, cells: List[NotebookCell]) -> str:
        """Build combined text content from cells.

        Args:
            cells: List of NotebookCell objects

        Returns:
            Combined text content
        """
        parts: List[str] = []

        for cell in cells:
            cell_text = self._format_cell(cell)
            if cell_text:
                parts.append(cell_text)

        return "\n\n".join(parts)

    def _format_cell(self, cell: NotebookCell) -> str:
        """Format a single cell for text output.

        Args:
            cell: NotebookCell object

        Returns:
            Formatted cell text
        """
        cell_type = cell.cell_type
        source = cell.source
        idx = cell.index

        if cell_type == "markdown":
            return f"## Markdown Cell {idx}\n\n{source}"

        if cell_type == "code":
            return self._format_code_cell(cell)

        if cell_type == "raw":
            return f"## Raw Cell {idx}\n\n{source}"

        return ""

    def _format_code_cell(self, cell: NotebookCell) -> str:
        """Format code cell with output."""
        idx = cell.index
        source = cell.source

        code_text = f"## Code Cell {idx}\n\n```\n{source}\n```"

        output_texts = [o.text for o in cell.outputs if o.text]
        if output_texts:
            outputs_text = "\n\n".join(output_texts)
            return f"{code_text}\n\nOutput:\n```\n{outputs_text}\n```"

        return code_text


# ============================================================================
# Module-level Functions
# ============================================================================


def extract_text(file_path: Path) -> str:
    """Extract text from Jupyter notebook.

    Args:
        file_path: Path to .ipynb file

    Returns:
        Extracted text content
    """
    processor = JupyterProcessor()
    result = processor.process(file_path)
    return result.get("text", "")


def extract_with_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract text and metadata from Jupyter notebook.

    Args:
        file_path: Path to .ipynb file

    Returns:
        Dictionary with text and metadata
    """
    processor = JupyterProcessor()
    return processor.process(file_path)


def process_to_notebook(file_path: Path) -> JupyterNotebook:
    """Process Jupyter notebook to structured object.

    Args:
        file_path: Path to .ipynb file

    Returns:
        JupyterNotebook with all extracted content
    """
    processor = JupyterProcessor()
    return processor.process_to_notebook(file_path)
