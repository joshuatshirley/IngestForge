"""
Shared Architectural Patterns and Interfaces.

This module defines the abstract contracts (interfaces) that feature modules
implement. By defining interfaces in the Shared layer, we enable:

1. **Dependency inversion**: High-level modules depend on abstractions, not concretions.
2. **Plugin architecture**: New implementations can be added without changing existing code.
3. **Testability**: Interfaces can be mocked in tests.
4. **Consistency**: All implementations follow the same contract.

Interface Catalog
-----------------
**IEnricher** (enricher.py)
    Contract for chunk enrichment. Implementations add metadata to chunks:
    - EntityExtractor: Extract named entities
    - QuestionGenerator: Generate hypothetical questions
    - EmbeddingGenerator: Generate vector embeddings

    Example implementation:
        class MyEnricher(IEnricher):
            def enrich_chunk(self, chunk: Any) -> None: ...
            def is_available(self) -> bool: ...

**IChunkingStrategy** (chunking.py)
    Contract for text chunking. Implementations split text into chunks:
    - SemanticChunker: Split on semantic boundaries
    - LegalChunker: Split legal documents by section
    - CodeChunker: Split code by function/class

    Example implementation:
        class MyChunker(IChunkingStrategy):
            def chunk(self, text, document_id) -> None: ...
            def get_strategy_name(self) -> None: ...

**IProcessor** (processor.py)
    Contract for document processing. Implementations extract content:
    - PDFProcessor: Extract text from PDFs
    - HTMLProcessor: Extract text from web pages
    - OCRProcessor: Extract text from images

    Example implementation:
        class MyProcessor(IProcessor):
            def can_process(self, file_path: Any) -> None: ...
            def process(self, file_path: Any) -> None: ...
            def get_supported_extensions(self) -> None: ...

Composition Helpers
-------------------
**EnrichmentPipeline**
    Chains multiple IEnricher implementations:
        pipeline = EnrichmentPipeline([
            EntityExtractor(),
            EmbeddingGenerator(config),
        ])
        enriched = pipeline.enrich(chunks)

**ProcessorFactory**
    Selects appropriate IProcessor for a file:
        factory = ProcessorFactory()
        factory.register(PDFProcessor())
        processor = factory.get_processor(Path("doc.pdf"))

**ChunkValidator**
    Validates chunks meet quality requirements:
        validator = ChunkValidator(min_size=50, max_size=1000)
        valid_chunks = validator.filter_valid(all_chunks)

Design Pattern: Strategy
------------------------
All interfaces follow the Strategy pattern. The caller doesn't know or care
which concrete implementation is used:

    def process_document(chunker: IChunkingStrategy, text: str):
        # Works with SemanticChunker, LegalChunker, CodeChunker, etc.
        return chunker.chunk(text, "doc_id")

This allows runtime selection of strategies via configuration.
"""

# IEnricher moved to enricher_adapter.py (deprecated)
from ingestforge.core.pipeline.enricher_adapter import (
    IEnricher,
    EnrichmentPipeline,
    EnrichmentError,
)
from ingestforge.shared.patterns.chunking import (
    IChunkingStrategy,
    ChunkValidator,
    ChunkingError,
)
from ingestforge.shared.patterns.processor import (
    IProcessor,
    ProcessorFactory,
    ProcessingError,
    ExtractedContent,
)

__all__ = [
    "IEnricher",
    "EnrichmentPipeline",
    "EnrichmentError",
    "IChunkingStrategy",
    "ChunkValidator",
    "ChunkingError",
    "IProcessor",
    "ProcessorFactory",
    "ProcessingError",
    "ExtractedContent",
]
