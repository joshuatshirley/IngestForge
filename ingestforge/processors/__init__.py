"""
IngestForge Processors Package.

Contains domain-specific processors that implement IFProcessor interface.
Document type processors extracted from PipelineSplittersMixin.
"""

from ingestforge.processors.security.redaction import IFRedactionProcessor
from ingestforge.processors.synthesis.synthesizer import (
    IFSynthesisProcessor,
    IFSynthesisArtifact,
)

# Document type processors
from ingestforge.processors.pdf_processor import StandardPDFProcessor
from ingestforge.processors.html_processor import HTMLExtractor
from ingestforge.processors.code_processor import CodeProcessor
from ingestforge.processors.markdown_processor import MarkdownProcessor

__all__ = [
    # Security processors
    "IFRedactionProcessor",
    # Synthesis processors
    "IFSynthesisProcessor",
    "IFSynthesisArtifact",
    # Document type processors ()
    "StandardPDFProcessor",
    "HTMLExtractor",
    "CodeProcessor",
    "MarkdownProcessor",
]
