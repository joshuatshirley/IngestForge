# Integration Tests for IngestForge Pipeline

## Overview

This directory contains comprehensive integration tests for the complete IngestForge document processing pipeline. These tests verify end-to-end functionality from raw documents to retrieval-ready storage.

## Test Files

### 1. `test_ingest_pipeline.py` (27 tests)

Tests document ingestion from various formats through type detection, processing, and metadata extraction.

**Coverage:**
- **Type Detection** (8 tests): PDF, HTML, Markdown, LaTeX, JSON, Jupyter notebooks, magic bytes, content detection
- **Text Extraction** (5 tests): HTML, Markdown, JSON, structure preservation, formatting handling
- **Metadata Extraction** (3 tests): HTML metadata, file metadata, document type metadata
- **Error Handling** (4 tests): Missing files, empty files, corrupt files, binary files
- **Format-Specific** (5 tests): Lists, code blocks, sections, notebook cells
- **Encoding** (2 tests): UTF-8 content, mixed encoding

**Key Features:**
- Real file processing (no mocks for core functionality)
- Minimal valid documents for each format
- Error recovery and validation
- Character encoding handling

### 2. `test_chunking_pipeline.py` (22 tests)

Tests chunking strategies including semantic, code, and legal document chunking with quality optimization.

**Coverage:**
- **Semantic Chunking** (6 tests): Long documents, size limits, overlap, content preservation, sentence boundaries, metadata
- **Code Chunking** (5 tests): Python code, class preservation, function preservation, docstrings, type metadata
- **Legal Chunking** (4 tests): Legal documents, article structure, section numbering, section titles
- **Deduplication** (2 tests): Exact duplicates, near duplicates
- **Size Optimization** (2 tests): Merge small chunks, split large chunks
- **Edge Cases** (3 tests): Short text, empty text, long sentences

**Key Features:**
- Strategy-specific chunking tests
- Semantic boundary detection
- Size constraint validation
- Duplicate detection and removal
- Edge case robustness

### 3. `test_enrichment_pipeline.py` (27 tests)

Tests enrichment components including entity extraction, topic detection, and LLM-powered generation.

**Coverage:**
- **Entity Extraction** (5 tests): Basic extraction, organizations, persons, locations, normalization
- **Topic Detection** (3 tests): Multi-chunk topics, ML topic identification, consistency
- **Question Generation** (3 tests): Basic generation, relevance, multiple questions
- **Summary Generation** (3 tests): Basic summaries, conciseness, batch processing
- **Embedding Generation** (4 tests): Basic embeddings, dimensions, normalization, batch processing
- **Temporal Extraction** (3 tests): Date extraction, date ranges, normalization
- **Sentiment Analysis** (3 tests): Positive, negative, neutral sentiment
- **Integration** (3 tests): Full pipeline, metadata preservation, error handling

**Key Features:**
- Mocked LLM and embedding models
- Component-level testing
- Integration testing between components
- Metadata preservation verification
- Graceful error handling

### 4. `test_storage_pipeline.py` (27 tests)

Tests storage backends including JSONL, ChromaDB, with focus on data persistence and retrieval.

**Coverage:**
- **JSONL Storage** (7 tests): Add chunks, retrieve by ID, get all, delete, clear, document IDs
- **JSONL Search** (5 tests): Keyword search, scores, top-k, semantic search, metadata filters
- **Bulk Operations** (3 tests): Bulk add, bulk delete, retrieval performance
- **Data Persistence** (3 tests): Reload storage, retrieve persisted chunks, embedding persistence
- **Storage Factory** (3 tests): JSONL backend, ChromaDB backend, config respect
- **Cross-Backend** (2 tests): Serialization format, JSONL export
- **Error Handling** (4 tests): Duplicate IDs, missing chunks, invalid queries, corruption

**Key Features:**
- JSONL backend primary testing
- ChromaDB conditional testing
- Data durability verification
- Performance benchmarking
- Cross-backend compatibility

### 5. `test_retrieval_pipeline.py` (28 tests)

Tests retrieval strategies including BM25, semantic search, hybrid fusion, and reranking.

**Coverage:**
- **BM25 Retrieval** (5 tests): Basic search, relevance, scoring, top-k, no matches
- **Semantic Retrieval** (4 tests): Basic search, concept similarity, scores, ordering
- **Hybrid Retrieval** (4 tests): Strategy combination, fusion weights, RRF, comparison
- **Query Processing** (4 tests): Simple queries, quoted phrases, expansion, technical terms
- **Reranking** (3 tests): Relevance reranking, order changes, preserve top results
- **Parent Retrieval** (2 tests): Parent context, sibling chunks
- **Cross-Corpus** (1 test): Multiple library search
- **Performance** (3 tests): BM25 speed, semantic speed, hybrid speed
- **Accuracy** (2 tests): Exact match, related concepts

**Key Features:**
- Strategy-specific testing
- Hybrid fusion testing (weighted, RRF)
- Performance benchmarking
- Accuracy validation
- Mock embeddings for consistency

## Test Statistics

| Test File | Tests | Lines | Coverage |
|-----------|-------|-------|----------|
| `test_ingest_pipeline.py` | 27 | 580 | Ingestion flow |
| `test_chunking_pipeline.py` | 22 | 620 | Chunking strategies |
| `test_enrichment_pipeline.py` | 27 | 680 | Enrichment components |
| `test_storage_pipeline.py` | 27 | 650 | Storage backends |
| `test_retrieval_pipeline.py` | 28 | 690 | Retrieval strategies |
| **Total** | **131** | **3,220** | **Full pipeline** |

## Running Tests

### Run All Integration Tests
```bash
pytest tests/integration/ -v
```

### Run Specific Test File
```bash
pytest tests/integration/test_ingest_pipeline.py -v
pytest tests/integration/test_chunking_pipeline.py -v
pytest tests/integration/test_enrichment_pipeline.py -v
pytest tests/integration/test_storage_pipeline.py -v
pytest tests/integration/test_retrieval_pipeline.py -v
```

### Run Specific Test Class
```bash
pytest tests/integration/test_retrieval_pipeline.py::TestHybridRetrieval -v
```

### Run with Coverage
```bash
pytest tests/integration/ --cov=ingestforge --cov-report=html
```

### Run Performance Tests Only
```bash
pytest tests/integration/test_storage_pipeline.py::TestBulkOperations -v
pytest tests/integration/test_retrieval_pipeline.py::TestRetrievalPerformance -v
```

## Design Principles

### 1. Real Integration Testing
- Use actual components (minimal mocking)
- Test real file I/O where appropriate
- Verify end-to-end data flow

### 2. Isolation and Cleanup
- Each test uses temporary directories
- Automatic cleanup via fixtures
- No cross-test contamination

### 3. Focused Test Classes
- One class per component/feature (Rule #4)
- Clear test organization
- Easy to locate specific tests

### 4. Comprehensive Coverage
- Happy path testing
- Error case testing
- Edge case testing
- Performance testing

### 5. Mock Strategy
- Mock external dependencies (LLMs, APIs)
- Use real local components
- Deterministic test data

## Test Data Strategy

### Fixtures
- **temp_dir**: Temporary directory for file operations
- **sample_chunks**: Pre-generated chunk collections
- **mock_llm_client**: Mocked LLM for generation tasks
- **mock_embedding_model**: Mocked embedding model with deterministic outputs
- **populated_storage**: Storage pre-loaded with test corpus

### Sample Documents
- **Minimal Valid Documents**: Create smallest valid files per format
- **Deterministic Data**: Use seeded random generators for embeddings
- **Realistic Content**: Use domain-relevant content (ML, tech, legal)

## Known Limitations

### Current Test Adaptations Needed

Some tests require adjustment to match actual API signatures:

1. **TextExtractor Return Type**: Some tests expect `.text` attribute, but actual implementation may return strings directly
2. **Format Support**: Not all formats have processors implemented yet (LaTeX, Jupyter)
3. **ChromaDB Availability**: ChromaDB tests skip if library not installed
4. **LLM Integration**: Real LLM tests require API keys (use mocks by default)

### Future Enhancements

1. **Add More Formats**: EPUB, DOCX, PPTX processors
2. **PostgreSQL Tests**: Add PostgreSQL storage backend tests
3. **Multi-Modal**: Add image and audio processing tests
4. **Streaming**: Add tests for streaming pipeline
5. **Distributed**: Add tests for distributed processing

## Maintenance

### Adding New Tests

1. Follow existing test structure
2. Use appropriate fixtures
3. Add to relevant test class
4. Update this README

### Test Naming Convention

```python
def test_<component>_<behavior>_<condition>:
    """Test that <component> <expected behavior> when <condition>."""
```

### Assertions

- **Positive**: Assert expected behavior
- **Negative**: Test error handling
- **Boundary**: Test edge cases

## Success Criteria

- [ ] All 131+ tests passing
- [ ] Pipeline stages tested end-to-end
- [ ] Error cases handled gracefully
- [ ] Performance within acceptable limits
- [ ] Cross-backend compatibility verified

## End-to-End Workflow Tests (NEW)

### Additional Workflow Test Files

Five comprehensive E2E workflow test files have been added to validate complete user scenarios:

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_academic_workflow.py` | 19 | Academic paper research workflow |
| `test_code_workflow.py` | 19 | Code repository analysis workflow |
| `test_study_workflow.py` | 17 | Study material generation workflow |
| `test_knowledge_workflow.py` | 17 | Knowledge base building workflow |
| `test_query_workflow.py` | 17 | Search and retrieval workflow |
| **Workflow Tests Total** | **89** | **Complete user workflows** |

**Combined Total: 220 integration tests**

### Workflow Test Documentation

- `WORKFLOW_TESTS_SUMMARY.md` - Detailed summary of workflow tests
- `API_FIX_GUIDE.md` - Guide for fixing API compatibility issues

### Running Workflow Tests

```bash
# All workflow tests
pytest tests/integration/test_*_workflow.py -v

# Specific workflow
pytest tests/integration/test_academic_workflow.py -v

# Without external APIs
pytest tests/integration/ -v -m "integration and not requires_api"
```

**Note:** Workflow tests may require import/API fixes. See `API_FIX_GUIDE.md` for details.

## Contact

For questions about integration tests, see:
- Main Pipeline: `ingestforge/core/pipeline/`
- Test Utilities: `tests/conftest.py`
- Workflow Tests: `WORKFLOW_TESTS_SUMMARY.md`
- Issue Tracker: GitHub Issues
