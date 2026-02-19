"""
Base Interface for Document Processors.

This module defines the IProcessor interface - the contract that all document
processing implementations must follow. Processors extract text and metadata
from various file formats.

Architecture Context
--------------------
Processing is Stage 2 in the pipeline (Split → **Extract** → Chunk → Enrich → Index).
Processors receive file paths and produce ExtractedContent:

    ┌───────────────┐       ┌───────────────────┐       ┌───────────────────┐
    │  document.pdf │ ────→ │   IProcessor      │ ────→ │ ExtractedContent  │
    │  document.html│       │  (PDF/HTML/etc)   │       │   + text          │
    │  document.docx│       └───────────────────┘       │   + metadata      │
    └───────────────┘                                   │   + sections      │
                                                        └───────────────────┘

Available Implementations
-------------------------
- PDFProcessor: Extract text from PDF files (pymupdf)
- HTMLProcessor: Extract text from web pages (trafilatura)
- DocxProcessor: Extract text from Word documents (python-docx)
- EpubProcessor: Extract text from e-books (ebooklib)
- OCRProcessor: Extract text from images (pytesseract)
- MarkdownProcessor: Parse markdown files

File Format Detection
---------------------
ProcessorFactory maintains a registry of processors and selects the
appropriate one based on file extension:

    factory = ProcessorFactory()
    factory.register(PDFProcessor())
    factory.register(HTMLProcessor())

    processor = factory.get_processor(Path("doc.pdf"))  # Returns PDFProcessor
    processor = factory.get_processor(Path("page.html"))  # Returns HTMLProcessor

ExtractedContent Structure
--------------------------
Processors return an ExtractedContent dataclass:

    @dataclass
    class ExtractedContent:
        text: str              # Main text content
        metadata: dict         # Title, author, date, etc.
        sections: List[str]    # Section headers (optional)
        images: List[str]      # Extracted image paths (optional)
        tables: List[dict]     # Extracted tables (optional)

Interface Contract
------------------
Implementors must provide:

    can_process(file_path)         - Check if processor handles this file type
    process(file_path)             - Extract content from file
    get_supported_extensions()     - List of supported extensions

The base class provides:

    validate_file(file_path)  - Check file exists and is readable
    get_processor_name()      - Return class name for logging
    get_metadata()            - Return processor info dict

Implementing a Custom Processor
-------------------------------
    class XMLProcessor(IProcessor):
        def can_process(self, file_path: Any):
            return file_path.suffix.lower() == ".xml"

        def process(self, file_path: Any):
            tree = ElementTree.parse(file_path)
            text = extract_text_from_xml(tree)
            return ExtractedContent(
                text=text,
                metadata={"format": "xml"},
            )

        def get_supported_extensions(self) -> None:
            return [".xml"]
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class ExtractedContent:
    """Content extracted from a document.

    Attributes:
        text: Extracted text content
        metadata: Document metadata (title, author, etc.)
        sections: List of section titles/hierarchy
        images: List of extracted image paths (optional)
        tables: List of extracted tables (optional)
    """

    text: str
    metadata: Dict[str, Any]
    sections: Optional[List[str]] = None
    images: Optional[List[str]] = None
    tables: Optional[List[Dict[str, Any]]] = None


class IProcessor(ABC):
    """Interface for document processors.

    All document processors (PDF, HTML, DOCX, etc.) should implement this
    interface to ensure consistent behavior across the ingestion pipeline.

    Examples:
        >>> class MyProcessor(IProcessor):
        ...     def can_process(self, file_path: Any):
        ...         return file_path.suffix.lower() == ".xyz"
        ...
        ...     def process(self, file_path: Any):
        ...         text = file_path.read_text()
        ...         return ExtractedContent(text=text, metadata={})
        ...
        ...     def get_supported_extensions(self):
        ...         return [".xyz"]
        ...
        >>> processor = MyProcessor()
        >>> if processor.can_process(Path("doc.xyz")):
        ...     content = processor.process(Path("doc.xyz"))
    """

    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if this processor can handle the file

        Examples:
            >>> processor = PDFProcessor()
            >>> processor.can_process(Path("document.pdf"))
            True
            >>> processor.can_process(Path("document.docx"))
            False
        """
        pass

    @abstractmethod
    def process(self, file_path: Path) -> ExtractedContent:
        """Process a document and extract content.

        Args:
            file_path: Path to the file to process

        Returns:
            ExtractedContent with text, metadata, and optional sections/images

        Raises:
            ProcessingError: If processing fails

        Examples:
            >>> processor = PDFProcessor()
            >>> content = processor.process(Path("document.pdf"))
            >>> print(content.text)
            >>> print(content.metadata)
        """
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions.

        Returns:
            List of extensions (e.g., [".pdf", ".epub"])

        Examples:
            >>> processor = PDFProcessor()
            >>> processor.get_supported_extensions()
            ['.pdf']
        """
        pass

    def get_processor_name(self) -> str:
        """Get the name of this processor.

        Returns:
            Processor name

        Examples:
            >>> processor = PDFProcessor()
            >>> processor.get_processor_name()
            'PDFProcessor'
        """
        return self.__class__.__name__

    def validate_file(self, file_path: Path) -> bool:
        """Validate that file exists and is readable.

        Args:
            file_path: Path to validate

        Returns:
            True if file is valid

        Examples:
            >>> processor = PDFProcessor()
            >>> processor.validate_file(Path("document.pdf"))
            True
        """
        return file_path.exists() and file_path.is_file()

    def get_metadata(self) -> Dict[str, Any]:
        """Get processor metadata.

        Returns:
            Dictionary with processor metadata

        Examples:
            >>> processor = PDFProcessor()
            >>> metadata = processor.get_metadata()
            >>> print(f"Supports: {metadata['extensions']}")
        """
        return {
            "name": self.get_processor_name(),
            "extensions": self.get_supported_extensions(),
            "version": getattr(self, "__version__", "unknown"),
        }

    def __repr__(self) -> str:
        """String representation of processor."""
        exts = ", ".join(self.get_supported_extensions())
        return f"{self.get_processor_name()}({exts})"


class ProcessingError(Exception):
    """Raised when document processing fails."""

    pass


class ProcessorFactory:
    """Factory for managing and selecting document processors.

    This class maintains a registry of processors and selects the appropriate
    one based on file extension.

    Examples:
        >>> factory = ProcessorFactory()
        >>> factory.register(PDFProcessor())
        >>> factory.register(HTMLProcessor())
        >>>
        >>> processor = factory.get_processor(Path("document.pdf"))
        >>> content = processor.process(Path("document.pdf"))
    """

    def __init__(self) -> None:
        """Initialize processor factory."""
        self.processors: List[IProcessor] = []

    def register(self, processor: IProcessor) -> None:
        """Register a processor.

        Args:
            processor: Processor to register

        Examples:
            >>> factory = ProcessorFactory()
            >>> factory.register(PDFProcessor())
        """
        self.processors.append(processor)

    def get_processor(self, file_path: Path) -> Optional[IProcessor]:
        """Get appropriate processor for a file.

        Args:
            file_path: Path to the file

        Returns:
            Processor that can handle the file, or None

        Examples:
            >>> factory = ProcessorFactory()
            >>> factory.register(PDFProcessor())
            >>> processor = factory.get_processor(Path("doc.pdf"))
        """
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        return None

    def get_supported_extensions(self) -> List[str]:
        """Get all supported extensions from registered processors.

        Returns:
            List of supported extensions

        Examples:
            >>> factory = ProcessorFactory()
            >>> factory.get_supported_extensions()
            ['.pdf', '.html', '.docx', '.txt', '.md']
        """
        extensions = set()
        for processor in self.processors:
            extensions.update(processor.get_supported_extensions())
        return sorted(extensions)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of registered processors.

        Returns:
            Dictionary with processor information
        """
        return {
            "processor_count": len(self.processors),
            "supported_extensions": self.get_supported_extensions(),
            "processors": [p.get_metadata() for p in self.processors],
        }
