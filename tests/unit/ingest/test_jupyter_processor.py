"""
Tests for Jupyter notebook processor.

This module tests extraction of code, markdown, outputs, imports,
and statistics from Jupyter notebooks.

Test Strategy
-------------
- Test JSON notebook parsing
- Test code cell extraction
- Test markdown cell extraction
- Test output extraction (stream, result, error)
- Test metadata extraction
- Test import extraction
- Test statistics calculation
- No external dependencies (uses standard json)

Organization
------------
- TestBasicProcessing: Main process method
- TestProcessToNotebook: Structured document output
- TestMetadataExtraction: Notebook metadata and kernel info
- TestKernelInfoExtraction: Kernel specification parsing
- TestCellExtraction: Cell parsing and organization
- TestCodeCellOutput: Output extraction from code cells
- TestCellFormatting: Text formatting for different cell types
- TestStatisticsCalculation: Line and cell counts
- TestImportExtraction: Python import parsing
- TestHelperFunctions: extract_text and extract_with_metadata
- TestComplexNotebooks: Realistic scenarios
- TestEdgeCases: Malformed notebooks and edge cases
"""

import json

from ingestforge.ingest.jupyter_processor import (
    JupyterProcessor,
    JupyterNotebook,
    NotebookCell,
    CellOutput,
    KernelInfo,
    extract_text,
    extract_with_metadata,
    process_to_notebook,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestBasicProcessing:
    """Tests for basic notebook processing.

    Rule #4: Focused test class - tests process method
    """

    def test_process_minimal_notebook(self, temp_dir):
        """Test processing minimal notebook."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        result = processor.process(nb_file)

        assert result["type"] == "jupyter"
        assert result["source"] == "test.ipynb"
        assert result["cells"] == []

    def test_process_with_code_cell(self, temp_dir):
        """Test processing notebook with code cell."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('Hello')"],
                    "outputs": [],
                }
            ],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        result = processor.process(nb_file)

        assert len(result["cells"]) == 1
        assert result["cells"][0]["type"] == "code"
        assert "print('Hello')" in result["cells"][0]["source"]

    def test_process_with_markdown_cell(self, temp_dir):
        """Test processing notebook with markdown cell."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Title\n", "Content"],
                }
            ],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        result = processor.process(nb_file)

        assert len(result["cells"]) == 1
        assert result["cells"][0]["type"] == "markdown"
        assert "# Title" in result["cells"][0]["source"]

    def test_process_returns_imports(self, temp_dir):
        """Test that process returns extracted imports."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["import pandas as pd\nimport numpy as np"],
                    "outputs": [],
                }
            ],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        result = processor.process(nb_file)

        assert "pandas" in result["imports"]
        assert "numpy" in result["imports"]


class TestProcessToNotebook:
    """Tests for process_to_notebook method."""

    def test_returns_jupyter_notebook(self, temp_dir):
        """Test that process_to_notebook returns JupyterNotebook."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [],
            "metadata": {"title": "My Notebook"},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        doc = processor.process_to_notebook(nb_file)

        assert isinstance(doc, JupyterNotebook)
        assert doc.title == "My Notebook"

    def test_document_has_all_fields(self, temp_dir):
        """Test that document contains all expected fields."""
        processor = JupyterProcessor()

        notebook = {"cells": [], "metadata": {}, "nbformat": 4}

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        doc = processor.process_to_notebook(nb_file)

        assert hasattr(doc, "title")
        assert hasattr(doc, "kernel")
        assert hasattr(doc, "cells")
        assert hasattr(doc, "imports")
        assert hasattr(doc, "total_code_lines")
        assert hasattr(doc, "total_markdown_lines")

    def test_document_cells_are_notebookcell(self, temp_dir):
        """Test that cells are NotebookCell objects."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [{"cell_type": "code", "source": ["x = 1"], "outputs": []}],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        doc = processor.process_to_notebook(nb_file)

        assert len(doc.cells) == 1
        assert isinstance(doc.cells[0], NotebookCell)


class TestMetadataExtraction:
    """Tests for metadata extraction.

    Rule #4: Focused test class - tests _extract_metadata
    """

    def test_extract_basic_metadata(self, temp_dir):
        """Test extracting basic notebook metadata."""
        processor = JupyterProcessor()

        nb_file = temp_dir / "test.ipynb"
        notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "name": "python3",
                    "language": "python",
                }
            },
            "nbformat": 4,
        }

        metadata = processor._extract_metadata(notebook, nb_file)

        assert metadata["filename"] == "test.ipynb"
        assert metadata["nbformat"] == 4
        assert metadata["kernel"] == "python3"
        assert metadata["language"] == "python"

    def test_extract_metadata_with_title(self, temp_dir):
        """Test extracting title from metadata."""
        processor = JupyterProcessor()

        nb_file = temp_dir / "test.ipynb"
        notebook = {
            "cells": [],
            "metadata": {
                "title": "My Notebook",
            },
            "nbformat": 4,
        }

        metadata = processor._extract_metadata(notebook, nb_file)

        assert metadata["title"] == "My Notebook"

    def test_metadata_includes_statistics(self, temp_dir):
        """Test that metadata includes statistics."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {"cell_type": "code", "source": ["x = 1\ny = 2"], "outputs": []},
                {"cell_type": "markdown", "source": ["# Title"]},
            ],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        result = processor.process(nb_file)

        assert "total_cells" in result["metadata"]
        assert "code_cells" in result["metadata"]
        assert "markdown_cells" in result["metadata"]


class TestKernelInfoExtraction:
    """Tests for kernel info extraction."""

    def test_extract_kernel_info(self, temp_dir):
        """Test extracting kernel info."""
        processor = JupyterProcessor()

        metadata = {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            }
        }

        kernel = processor._extract_kernel_info(metadata)

        assert isinstance(kernel, KernelInfo)
        assert kernel.name == "python3"
        assert kernel.display_name == "Python 3"
        assert kernel.language == "python"

    def test_get_kernel_language_from_kernelspec(self, temp_dir):
        """Test getting language from kernelspec."""
        processor = JupyterProcessor()

        metadata = {
            "kernelspec": {
                "language": "python",
            }
        }

        language = processor._get_kernel_language(metadata)

        assert language == "python"

    def test_get_kernel_language_from_language_info(self, temp_dir):
        """Test getting language from language_info fallback."""
        processor = JupyterProcessor()

        metadata = {
            "kernelspec": {},
            "language_info": {
                "name": "julia",
            },
        }

        language = processor._get_kernel_language(metadata)

        assert language == "julia"

    def test_get_kernel_language_unknown(self, temp_dir):
        """Test language defaults to unknown."""
        processor = JupyterProcessor()

        metadata = {}

        language = processor._get_kernel_language(metadata)

        assert language == "unknown"

    def test_kernel_info_defaults(self):
        """Test KernelInfo default values."""
        kernel = KernelInfo()

        assert kernel.name == ""
        assert kernel.display_name == ""
        assert kernel.language == "unknown"


class TestCellExtraction:
    """Tests for cell extraction.

    Rule #4: Focused test class - tests _extract_cells
    """

    def test_extract_single_cell(self):
        """Test extracting single cell."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["x = 1"],
                    "outputs": [],
                }
            ]
        }

        cells = processor._extract_cells(notebook)

        assert len(cells) == 1
        assert cells[0].index == 0
        assert cells[0].cell_type == "code"
        assert cells[0].source == "x = 1"

    def test_extract_multiple_cells(self):
        """Test extracting multiple cells."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title"]},
                {"cell_type": "code", "source": ["x = 1"], "outputs": []},
                {"cell_type": "markdown", "source": ["Text"]},
            ]
        }

        cells = processor._extract_cells(notebook)

        assert len(cells) == 3
        assert cells[0].cell_type == "markdown"
        assert cells[1].cell_type == "code"
        assert cells[2].cell_type == "markdown"

    def test_extract_source_as_list(self):
        """Test extracting source when it's a list."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["Line 1\n", "Line 2"],
                }
            ]
        }

        cells = processor._extract_cells(notebook)

        assert cells[0].source == "Line 1\nLine 2"

    def test_extract_source_as_string(self):
        """Test extracting source when it's a string."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "print('hello')",
                    "outputs": [],
                }
            ]
        }

        cells = processor._extract_cells(notebook)

        assert cells[0].source == "print('hello')"

    def test_extract_cell_outputs(self):
        """Test extracting outputs from code cells."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["x"],
                    "outputs": [
                        {
                            "output_type": "execute_result",
                            "data": {"text/plain": "42"},
                        }
                    ],
                }
            ]
        }

        cells = processor._extract_cells(notebook)

        assert len(cells[0].outputs) == 1
        assert cells[0].outputs[0].text == "42"

    def test_extract_cell_execution_count(self):
        """Test extracting execution count."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["x = 1"],
                    "outputs": [],
                    "execution_count": 5,
                }
            ]
        }

        cells = processor._extract_cells(notebook)

        assert cells[0].execution_count == 5


class TestCodeCellOutput:
    """Tests for code cell output extraction.

    Rule #4: Focused test class - tests output processing
    """

    def test_extract_stream_output(self):
        """Test extracting stream output."""
        processor = JupyterProcessor()

        output = {
            "output_type": "stream",
            "text": ["Hello\n", "World"],
        }

        result = processor._extract_single_output(output)

        assert result.text == "Hello\nWorld"
        assert result.output_type == "stream"

    def test_extract_stream_output_string(self):
        """Test extracting stream output as string."""
        processor = JupyterProcessor()

        output = {
            "output_type": "stream",
            "text": "Output text",
        }

        result = processor._extract_single_output(output)

        assert result.text == "Output text"

    def test_extract_execute_result(self):
        """Test extracting execution result."""
        processor = JupyterProcessor()

        output = {
            "output_type": "execute_result",
            "data": {
                "text/plain": "42",
            },
        }

        result = processor._extract_single_output(output)

        assert result.text == "42"

    def test_extract_execute_result_list(self):
        """Test extracting execution result as list."""
        processor = JupyterProcessor()

        output = {
            "output_type": "execute_result",
            "data": {
                "text/plain": ["Line 1\n", "Line 2"],
            },
        }

        result = processor._extract_single_output(output)

        assert result.text == "Line 1\nLine 2"

    def test_extract_display_data(self):
        """Test extracting display_data output."""
        processor = JupyterProcessor()

        output = {
            "output_type": "display_data",
            "data": {
                "text/plain": "<matplotlib figure>",
            },
        }

        result = processor._extract_single_output(output)

        assert result.text == "<matplotlib figure>"

    def test_extract_error_output(self):
        """Test extracting error traceback."""
        processor = JupyterProcessor()

        output = {
            "output_type": "error",
            "traceback": [
                "Traceback (most recent call last):",
                '  File "<stdin>", line 1',
                "NameError: name 'x' is not defined",
            ],
        }

        result = processor._extract_single_output(output)

        assert "Traceback" in result.text
        assert "NameError" in result.text
        assert result.is_error is True

    def test_extract_output_no_text_plain(self):
        """Test extraction when text/plain not available."""
        processor = JupyterProcessor()

        output = {
            "output_type": "execute_result",
            "data": {
                "text/html": "<div>HTML</div>",
            },
        }

        result = processor._extract_single_output(output)

        assert result.text == "<div>HTML</div>"
        assert result.mime_type == "text/html"

    def test_extract_unknown_output_type(self):
        """Test extraction of unknown output type."""
        processor = JupyterProcessor()

        output = {
            "output_type": "unknown",
        }

        result = processor._extract_single_output(output)

        assert result is None

    def test_cell_output_dataclass(self):
        """Test CellOutput dataclass."""
        output = CellOutput(
            output_type="stream",
            text="Hello",
            mime_type="text/plain",
            is_error=False,
        )

        assert output.output_type == "stream"
        assert output.text == "Hello"
        assert output.is_error is False


class TestCellFormatting:
    """Tests for cell formatting.

    Rule #4: Focused test class - tests _format_cell
    """

    def test_format_markdown_cell(self):
        """Test formatting markdown cell."""
        processor = JupyterProcessor()

        cell = NotebookCell(
            cell_type="markdown",
            source="# Title\n\nContent",
            index=0,
        )

        result = processor._format_cell(cell)

        assert "## Markdown Cell 0" in result
        assert "# Title" in result
        assert "Content" in result

    def test_format_code_cell_no_output(self):
        """Test formatting code cell without output."""
        processor = JupyterProcessor()

        cell = NotebookCell(
            cell_type="code",
            source="x = 1",
            outputs=[],
            index=1,
        )

        result = processor._format_cell(cell)

        assert "## Code Cell 1" in result
        assert "```\nx = 1\n```" in result
        assert "Output:" not in result

    def test_format_code_cell_with_output(self):
        """Test formatting code cell with output."""
        processor = JupyterProcessor()

        cell = NotebookCell(
            cell_type="code",
            source="print('hello')",
            outputs=[CellOutput(output_type="stream", text="hello")],
            index=2,
        )

        result = processor._format_cell(cell)

        assert "## Code Cell 2" in result
        assert "print('hello')" in result
        assert "Output:" in result
        assert "hello" in result

    def test_format_raw_cell(self):
        """Test formatting raw cell."""
        processor = JupyterProcessor()

        cell = NotebookCell(
            cell_type="raw",
            source="Raw content",
            index=4,
        )

        result = processor._format_cell(cell)

        assert "## Raw Cell 4" in result
        assert "Raw content" in result


class TestStatisticsCalculation:
    """Tests for statistics calculation."""

    def test_calculate_code_lines(self, temp_dir):
        """Test calculating total code lines."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {"cell_type": "code", "source": ["x = 1\ny = 2\nz = 3"], "outputs": []},
                {"cell_type": "code", "source": ["a = 4\nb = 5"], "outputs": []},
            ],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        doc = processor.process_to_notebook(nb_file)

        assert doc.total_code_lines == 5
        assert doc.total_code_cells == 2

    def test_calculate_markdown_lines(self, temp_dir):
        """Test calculating total markdown lines."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n\nParagraph"]},
                {"cell_type": "markdown", "source": ["More text"]},
            ],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        doc = processor.process_to_notebook(nb_file)

        assert doc.total_markdown_cells == 2
        assert doc.total_markdown_lines > 0

    def test_statistics_empty_notebook(self, temp_dir):
        """Test statistics for empty notebook."""
        processor = JupyterProcessor()

        notebook = {"cells": [], "metadata": {}, "nbformat": 4}

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        doc = processor.process_to_notebook(nb_file)

        assert doc.total_code_cells == 0
        assert doc.total_markdown_cells == 0
        assert doc.total_code_lines == 0
        assert doc.total_markdown_lines == 0


class TestImportExtraction:
    """Tests for import extraction."""

    def test_extract_basic_import(self, temp_dir):
        """Test extracting basic import."""
        processor = JupyterProcessor()

        cells = [NotebookCell(cell_type="code", source="import pandas", index=0)]

        imports = processor._extract_imports(cells)

        assert "pandas" in imports

    def test_extract_from_import(self, temp_dir):
        """Test extracting from...import."""
        processor = JupyterProcessor()

        cells = [
            NotebookCell(
                cell_type="code",
                source="from sklearn.model_selection import train_test_split",
                index=0,
            )
        ]

        imports = processor._extract_imports(cells)

        assert "sklearn" in imports

    def test_extract_import_as(self, temp_dir):
        """Test extracting import with alias."""
        processor = JupyterProcessor()

        cells = [NotebookCell(cell_type="code", source="import numpy as np", index=0)]

        imports = processor._extract_imports(cells)

        assert "numpy" in imports

    def test_extract_multiple_imports_one_line(self, temp_dir):
        """Test extracting multiple imports from one line."""
        processor = JupyterProcessor()

        cells = [NotebookCell(cell_type="code", source="import os, sys, json", index=0)]

        imports = processor._extract_imports(cells)

        assert "os" in imports
        assert "sys" in imports
        assert "json" in imports

    def test_extract_imports_from_multiple_cells(self, temp_dir):
        """Test extracting imports from multiple cells."""
        processor = JupyterProcessor()

        cells = [
            NotebookCell(cell_type="code", source="import pandas", index=0),
            NotebookCell(cell_type="code", source="import numpy", index=1),
        ]

        imports = processor._extract_imports(cells)

        assert "pandas" in imports
        assert "numpy" in imports

    def test_ignore_markdown_cells(self, temp_dir):
        """Test that markdown cells are ignored."""
        processor = JupyterProcessor()

        cells = [
            NotebookCell(cell_type="markdown", source="import fake", index=0),
            NotebookCell(cell_type="code", source="import real", index=1),
        ]

        imports = processor._extract_imports(cells)

        assert "fake" not in imports
        assert "real" in imports

    def test_deduplicate_imports(self, temp_dir):
        """Test that duplicate imports are removed."""
        processor = JupyterProcessor()

        cells = [
            NotebookCell(cell_type="code", source="import pandas", index=0),
            NotebookCell(cell_type="code", source="import pandas as pd", index=1),
        ]

        imports = processor._extract_imports(cells)

        assert imports.count("pandas") == 1

    def test_extract_submodule_imports(self, temp_dir):
        """Test extracting submodule imports gets top-level."""
        processor = JupyterProcessor()

        cells = [
            NotebookCell(
                cell_type="code", source="import matplotlib.pyplot as plt", index=0
            )
        ]

        imports = processor._extract_imports(cells)

        assert "matplotlib" in imports


class TestBuildTextContent:
    """Tests for building combined text content.

    Rule #4: Focused test class - tests _build_text_content
    """

    def test_build_text_from_cells(self):
        """Test building combined text from cells."""
        processor = JupyterProcessor()

        cells = [
            NotebookCell(cell_type="markdown", source="# Title", index=0),
            NotebookCell(cell_type="code", source="x = 1", outputs=[], index=1),
        ]

        result = processor._build_text_content(cells)

        assert "# Title" in result
        assert "x = 1" in result
        assert "\n\n" in result

    def test_build_text_single_cell(self):
        """Test building text from single cell."""
        processor = JupyterProcessor()

        cells = [
            NotebookCell(cell_type="markdown", source="Content", index=0),
        ]

        result = processor._build_text_content(cells)

        assert "Content" in result

    def test_build_text_empty_cells(self):
        """Test building text from empty cell list."""
        processor = JupyterProcessor()

        cells = []

        result = processor._build_text_content(cells)

        assert result == ""


class TestHelperFunctions:
    """Tests for helper functions.

    Rule #4: Focused test class - tests module-level functions
    """

    def test_extract_text_function(self, temp_dir):
        """Test extract_text helper function."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["Test content"]},
            ],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        result = extract_text(nb_file)

        assert "Test content" in result

    def test_extract_with_metadata_function(self, temp_dir):
        """Test extract_with_metadata helper function."""
        notebook = {
            "cells": [
                {"cell_type": "code", "source": ["x = 1"], "outputs": []},
            ],
            "metadata": {"title": "Test Notebook"},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        result = extract_with_metadata(nb_file)

        assert "text" in result
        assert "metadata" in result
        assert result["metadata"]["title"] == "Test Notebook"

    def test_process_to_notebook_function(self, temp_dir):
        """Test process_to_notebook helper function."""
        notebook = {
            "cells": [],
            "metadata": {"title": "Test"},
            "nbformat": 4,
        }

        nb_file = temp_dir / "test.ipynb"
        nb_file.write_text(json.dumps(notebook))

        doc = process_to_notebook(nb_file)

        assert isinstance(doc, JupyterNotebook)
        assert doc.title == "Test"


class TestComplexNotebooks:
    """Tests for complex notebooks.

    Rule #4: Focused test class - tests realistic scenarios
    """

    def test_data_analysis_notebook(self, temp_dir):
        """Test processing data analysis notebook."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Data Analysis\n", "Analysis of dataset"],
                },
                {"cell_type": "code", "source": ["import pandas as pd"], "outputs": []},
                {
                    "cell_type": "code",
                    "source": ["df = pd.read_csv('data.csv')"],
                    "outputs": [],
                },
                {
                    "cell_type": "code",
                    "source": ["df.head()"],
                    "outputs": [
                        {
                            "output_type": "execute_result",
                            "data": {"text/plain": "   A  B\n0  1  2"},
                        }
                    ],
                },
                {
                    "cell_type": "markdown",
                    "source": ["## Results\n", "The analysis shows..."],
                },
            ],
            "metadata": {"kernelspec": {"name": "python3", "language": "python"}},
            "nbformat": 4,
        }

        nb_file = temp_dir / "analysis.ipynb"
        nb_file.write_text(json.dumps(notebook))

        result = processor.process(nb_file)

        assert "Data Analysis" in result["text"]
        assert "import pandas" in result["text"]
        assert "Results" in result["text"]
        assert len(result["cells"]) == 5
        assert "pandas" in result["imports"]

    def test_notebook_with_errors(self, temp_dir):
        """Test processing notebook with error outputs."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["undefined_var"],
                    "outputs": [
                        {
                            "output_type": "error",
                            "traceback": [
                                "NameError: name 'undefined_var' is not defined"
                            ],
                        }
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "errors.ipynb"
        nb_file.write_text(json.dumps(notebook))

        result = processor.process(nb_file)

        assert "NameError" in result["text"]

    def test_machine_learning_notebook(self, temp_dir):
        """Test processing ML notebook."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# ML Model"]},
                {
                    "cell_type": "code",
                    "source": [
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "from sklearn.model_selection import train_test_split\n",
                        "from sklearn.linear_model import LogisticRegression",
                    ],
                    "outputs": [],
                },
                {
                    "cell_type": "code",
                    "source": [
                        "model = LogisticRegression()\nmodel.fit(X_train, y_train)"
                    ],
                    "outputs": [],
                },
            ],
            "metadata": {"kernelspec": {"name": "python3", "language": "python"}},
            "nbformat": 4,
        }

        nb_file = temp_dir / "ml.ipynb"
        nb_file.write_text(json.dumps(notebook))

        doc = processor.process_to_notebook(nb_file)

        assert "numpy" in doc.imports
        assert "pandas" in doc.imports
        assert "sklearn" in doc.imports


class TestEdgeCases:
    """Tests for edge cases.

    Rule #4: Focused test class - tests edge cases
    """

    def test_empty_notebook(self, temp_dir):
        """Test processing empty notebook."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "empty.ipynb"
        nb_file.write_text(json.dumps(notebook))

        result = processor.process(nb_file)

        assert result["cells"] == []
        assert result["text"] == ""

    def test_cell_missing_outputs(self):
        """Test cell without outputs key."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["x = 1"],
                    # No outputs key
                }
            ]
        }

        cells = processor._extract_cells(notebook)

        assert cells[0].outputs == []

    def test_cell_empty_source(self):
        """Test cell with empty source."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [],
                    "outputs": [],
                }
            ]
        }

        cells = processor._extract_cells(notebook)

        assert cells[0].source == ""

    def test_notebook_no_metadata(self, temp_dir):
        """Test notebook without metadata."""
        processor = JupyterProcessor()

        nb_file = temp_dir / "test.ipynb"
        notebook = {
            "cells": [],
            "nbformat": 4,
        }

        metadata = processor._extract_metadata(notebook, nb_file)

        assert metadata["filename"] == "test.ipynb"

    def test_output_with_ansi_codes(self):
        """Test that ANSI codes are cleaned from error output."""
        processor = JupyterProcessor()

        output = {
            "output_type": "error",
            "traceback": [
                "\x1b[31mError:\x1b[0m Something went wrong",
            ],
        }

        result = processor._extract_single_output(output)

        assert "\x1b[" not in result.text
        assert "Error:" in result.text

    def test_notebook_with_raw_cells(self, temp_dir):
        """Test notebook with raw cells."""
        processor = JupyterProcessor()

        notebook = {
            "cells": [
                {"cell_type": "raw", "source": ["Raw text content"]},
            ],
            "metadata": {},
            "nbformat": 4,
        }

        nb_file = temp_dir / "raw.ipynb"
        nb_file.write_text(json.dumps(notebook))

        result = processor.process(nb_file)

        assert "Raw text content" in result["text"]


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Basic processing: 4 tests
    - Process to notebook: 3 tests
    - Metadata extraction: 3 tests
    - Kernel info extraction: 5 tests
    - Cell extraction: 6 tests
    - Code cell output: 9 tests
    - Cell formatting: 4 tests
    - Statistics calculation: 3 tests
    - Import extraction: 8 tests
    - Text building: 3 tests
    - Helper functions: 3 tests
    - Complex notebooks: 3 tests
    - Edge cases: 6 tests

    Total: 60 tests

Design Decisions:
    1. No external dependencies (uses standard json)
    2. Test all cell types (code, markdown, raw)
    3. Cover all output types (stream, execute_result, display_data, error)
    4. Test source as both list and string
    5. Include realistic notebook scenarios
    6. Test edge cases and missing fields
    7. Test import extraction from various formats
    8. Test statistics calculation
"""
