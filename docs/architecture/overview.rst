Architecture Overview
====================

IngestForge follows a modular pipeline architecture with clear separation of concerns.

System Architecture
-------------------

.. code-block:: text

    ┌─────────────┐
    │     CLI     │  Entry point for user commands
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   Pipeline  │  Orchestrates the ingestion and query flow
    └──────┬──────┘
           │
    ┌──────▼──────────────────────────────────────────┐
    │                                                  │
    │  ┌────────┐  ┌───────────┐  ┌─────────┐       │
    │  │ Ingest │→ │ Enrichment│→ │ Storage │       │
    │  └────────┘  └───────────┘  └─────────┘       │
    │                                                  │
    │  ┌────────┐  ┌───────────┐  ┌─────────┐       │
    │  │ Query  │← │ Retrieval │← │ Storage │       │
    │  └────────┘  └───────────┘  └─────────┘       │
    │                                                  │
    └──────────────────────────────────────────────────┘

Module Responsibilities
-----------------------

Core
~~~~
- Configuration management
- Pipeline orchestration
- State tracking
- Retry logic
- Logging

Ingest
~~~~~~
- Document format detection
- Text extraction (PDF, HTML, DOCX, etc.)
- OCR processing
- Citation metadata extraction
- Content refinement

Enrichment
~~~~~~~~~~
- Embedding generation
- Metadata extraction
- Summary generation
- Question generation
- Entity extraction

Storage
~~~~~~~
- JSONL file storage
- ChromaDB vector storage
- Storage factory pattern
- Compression utilities

Retrieval
~~~~~~~~~
- BM25 keyword search
- Semantic vector search
- Hybrid search combining both
- Reranking with cross-encoders
- Authority boosting

Query
~~~~~
- Query pipeline
- Query caching
- Advanced query parsing (AND/OR/NOT)
- Query history tracking
- Concept map generation

LLM
~~~
- LLM provider abstraction
- Support for Claude, OpenAI, Ollama, llama.cpp
- Factory pattern for provider selection

Data Flow
---------

Ingestion Pipeline
~~~~~~~~~~~~~~~~~~

1. **Input**: User provides documents in ``.ingest/pending/``
2. **Detection**: Format detector identifies document type
3. **Extraction**: Appropriate processor extracts text and metadata
4. **Chunking**: Text is split into semantic chunks
5. **Enrichment**: Chunks receive embeddings, summaries, questions
6. **Storage**: Chunks are persisted to JSONL and vector store

Query Pipeline
~~~~~~~~~~~~~~

1. **Input**: User submits natural language query
2. **Parsing**: Advanced query parser extracts operators and filters
3. **Retrieval**: Hybrid search finds relevant chunks
4. **Reranking**: Cross-encoder reranks results by relevance
5. **Filtering**: Field filters and exclusions applied
6. **Output**: Results returned with citations and scores

Extension Points
----------------

The architecture is designed for extensibility:

- **New document formats**: Implement ``Processor`` interface
- **New storage backends**: Implement ``BaseStorage`` interface
- **New LLM providers**: Implement ``BaseLLM`` interface
- **New retrieval methods**: Extend ``BaseRetriever``
- **Custom enrichment**: Add enricher classes with ``process()`` method
