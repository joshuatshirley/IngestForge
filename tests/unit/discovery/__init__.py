"""Unit tests for discovery module.

This package contains tests for academic paper discovery clients:
- test_arxiv_client: Tests for ArxivSearcher class
- test_arxiv_search: Tests for legacy arxiv search (academic_search module)
- test_semantic_scholar: Tests for SemanticScholarClient
- test_crossref: Tests for CrossRefClient

Total test count: 79+ tests covering:
- Search functionality for all APIs
- Rate limiting behavior
- Paper dataclass methods
- BibTeX export
- Error handling
- ID parsing and validation
"""
