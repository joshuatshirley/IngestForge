"""
Sample document generators for testing.

This module provides functions to create minimal valid documents
in various formats for testing extraction and processing pipelines.

All generators create real files in the provided directory.
Use with temp_dir fixture for automatic cleanup.

Usage Example
-------------
    def test_pdf_extraction(temp_dir):
        pdf_path = create_sample_pdf(temp_dir, "Sample PDF content")
        # Test PDF extraction logic
        ...
"""

from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# Text-Based Document Generators
# ============================================================================


def create_sample_txt(
    output_dir: Path, content: Optional[str] = None, filename: str = "sample.txt"
) -> Path:
    """Create a sample text file.

    Args:
        output_dir: Directory to create file in
        content: File content (uses default if None)
        filename: Output filename

    Returns:
        Path to created file
    """
    if content is None:
        content = """This is a sample text file.
It contains multiple lines of text.
This helps test text extraction and processing.

It even has multiple paragraphs.
"""

    file_path = output_dir / filename
    file_path.write_text(content, encoding="utf-8")
    return file_path


def create_sample_markdown(
    output_dir: Path, content: Optional[str] = None, filename: str = "sample.md"
) -> Path:
    """Create a sample markdown file with structure.

    Args:
        output_dir: Directory to create file in
        content: File content (uses default if None)
        filename: Output filename

    Returns:
        Path to created file
    """
    if content is None:
        content = """# Main Document Title

This is the introduction paragraph with some **bold** and *italic* text.

## Section 1: Overview

This section provides an overview of the topic. It contains multiple
sentences to create realistic content for testing.

### Subsection 1.1

Nested content goes here.

## Section 2: Details

More detailed content in this section.

- Item 1
- Item 2
- Item 3

### Subsection 2.1: Code Example

```python
def hello():
    print("Hello, world!")
```

## Section 3: Conclusion

Concluding remarks go here.
"""

    file_path = output_dir / filename
    file_path.write_text(content, encoding="utf-8")
    return file_path


def create_sample_html(
    output_dir: Path, content: Optional[str] = None, filename: str = "sample.html"
) -> Path:
    """Create a sample HTML file.

    Args:
        output_dir: Directory to create file in
        content: File content (uses default if None)
        filename: Output filename

    Returns:
        Path to created file
    """
    if content is None:
        content = """<!DOCTYPE html>
<html>
<head>
    <title>Sample HTML Document</title>
    <meta charset="UTF-8">
</head>
<body>
    <h1>Main Title</h1>
    <p>This is a sample HTML document for testing.</p>

    <h2>Section 1</h2>
    <p>Content in section 1. It has multiple sentences.</p>

    <h3>Subsection 1.1</h3>
    <p>Nested content here.</p>

    <h2>Section 2</h2>
    <p>More content in section 2.</p>

    <ul>
        <li>List item 1</li>
        <li>List item 2</li>
        <li>List item 3</li>
    </ul>

    <table>
        <tr>
            <th>Header 1</th>
            <th>Header 2</th>
        </tr>
        <tr>
            <td>Data 1</td>
            <td>Data 2</td>
        </tr>
    </table>
</body>
</html>
"""

    file_path = output_dir / filename
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ============================================================================
# Structured Data Document Generators
# ============================================================================


def create_sample_json(
    output_dir: Path, data: Optional[Dict] = None, filename: str = "sample.json"
) -> Path:
    """Create a sample JSON file.

    Args:
        output_dir: Directory to create file in
        data: JSON data (uses default if None)
        filename: Output filename

    Returns:
        Path to created file
    """
    import json

    if data is None:
        data = {
            "title": "Sample Document",
            "author": "Test Author",
            "date": "2024-01-01",
            "content": "This is the main content of the document.",
            "sections": [
                {
                    "title": "Section 1",
                    "content": "Content for section 1.",
                },
                {
                    "title": "Section 2",
                    "content": "Content for section 2.",
                },
            ],
            "metadata": {
                "keywords": ["test", "sample", "document"],
                "category": "testing",
            },
        }

    file_path = output_dir / filename
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return file_path


def create_sample_jsonl(
    output_dir: Path,
    records: Optional[List[Dict]] = None,
    filename: str = "sample.jsonl",
) -> Path:
    """Create a sample JSONL file with multiple records.

    Args:
        output_dir: Directory to create file in
        records: List of record dicts (uses default if None)
        filename: Output filename

    Returns:
        Path to created file
    """
    import json

    if records is None:
        records = [
            {"chunk_id": "chunk_1", "content": "First chunk content"},
            {"chunk_id": "chunk_2", "content": "Second chunk content"},
            {"chunk_id": "chunk_3", "content": "Third chunk content"},
        ]

    file_path = output_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return file_path


def create_sample_csv(
    output_dir: Path,
    rows: Optional[List[List[str]]] = None,
    filename: str = "sample.csv",
) -> Path:
    """Create a sample CSV file.

    Args:
        output_dir: Directory to create file in
        rows: List of rows (uses default if None)
        filename: Output filename

    Returns:
        Path to created file
    """
    import csv

    if rows is None:
        rows = [
            ["Name", "Age", "City"],
            ["Alice", "30", "New York"],
            ["Bob", "25", "San Francisco"],
            ["Charlie", "35", "Los Angeles"],
        ]

    file_path = output_dir / filename
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return file_path


# ============================================================================
# Binary Document Generators (Minimal/Mock)
# ============================================================================


def create_mock_pdf(
    output_dir: Path, content: Optional[str] = None, filename: str = "sample.pdf"
) -> Path:
    """Create a mock PDF file for testing.

    NOTE: This creates a minimal PDF structure that may not be valid
    for all PDF parsers. Use for testing file detection and basic
    structure parsing only.

    For real PDF testing, use integration tests with actual PDF files.

    Args:
        output_dir: Directory to create file in
        content: Text content (uses default if None)
        filename: Output filename

    Returns:
        Path to created file
    """
    if content is None:
        content = "Sample PDF content for testing"

    # Create minimal PDF structure
    pdf_content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
({content}) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000261 00000 n
0000000330 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
424
%%EOF
"""

    file_path = output_dir / filename
    file_path.write_bytes(pdf_content.encode("latin-1"))
    return file_path


def create_mock_docx(
    output_dir: Path, content: Optional[str] = None, filename: str = "sample.docx"
) -> Path:
    """Create a mock DOCX file for testing.

    NOTE: This creates a minimal DOCX structure (ZIP with XML).
    May not be valid for all DOCX parsers.

    For real DOCX testing, use integration tests with actual DOCX files.

    Args:
        output_dir: Directory to create file in
        content: Text content (uses default if None)
        filename: Output filename

    Returns:
        Path to created file
    """
    import zipfile

    if content is None:
        content = "Sample DOCX content for testing"

    # Create minimal DOCX structure (simplified)
    file_path = output_dir / filename

    with zipfile.ZipFile(file_path, "w") as docx:
        # Add minimal content types
        content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
        docx.writestr("[Content_Types].xml", content_types)

        # Add minimal document.xml
        document_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:r>
        <w:t>{content}</w:t>
      </w:r>
    </w:p>
  </w:body>
</w:document>
"""
        docx.writestr("word/document.xml", document_xml)

    return file_path


def create_mock_pptx(
    output_dir: Path,
    slide_content: Optional[List[str]] = None,
    filename: str = "sample.pptx",
) -> Path:
    """Create a mock PPTX file for testing.

    NOTE: This creates a minimal PPTX structure (ZIP with XML).
    May not be valid for all PPTX parsers.

    For real PPTX testing, use integration tests with actual PPTX files.

    Args:
        output_dir: Directory to create file in
        slide_content: List of slide text content (uses default if None)
        filename: Output filename

    Returns:
        Path to created file
    """
    import zipfile

    if slide_content is None:
        slide_content = ["Slide 1 content", "Slide 2 content"]

    file_path = output_dir / filename

    with zipfile.ZipFile(file_path, "w") as pptx:
        # Add minimal content types
        content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
</Types>
"""
        pptx.writestr("[Content_Types].xml", content_types)

        # Add minimal slide
        for idx, content in enumerate(slide_content, 1):
            slide_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:sp>
        <p:txBody>
          <a:p>
            <a:r>
              <a:t>{content}</a:t>
            </a:r>
          </a:p>
        </p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>
"""
            pptx.writestr(f"ppt/slides/slide{idx}.xml", slide_xml)

    return file_path


# ============================================================================
# Multi-Format Document Generator
# ============================================================================


def create_sample_documents(output_dir: Path) -> Dict[str, Path]:
    """Create a full set of sample documents in various formats.

    Args:
        output_dir: Directory to create files in

    Returns:
        Dict mapping format name to file path

    Example:
        def test_all_formats(temp_dir):
            docs = create_sample_documents(temp_dir)
            assert docs["txt"].exists()
            assert docs["markdown"].exists()
    """
    return {
        "txt": create_sample_txt(output_dir),
        "markdown": create_sample_markdown(output_dir),
        "html": create_sample_html(output_dir),
        "json": create_sample_json(output_dir),
        "jsonl": create_sample_jsonl(output_dir),
        "csv": create_sample_csv(output_dir),
        "pdf": create_mock_pdf(output_dir),
        "docx": create_mock_docx(output_dir),
        "pptx": create_mock_pptx(output_dir),
    }
