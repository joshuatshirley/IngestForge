"""Tests for knowledge base tools.

Tests agent tools for search, ingestion, and chunk retrieval."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from ingestforge.agent.knowledge_tools import (
    search_knowledge_base,
    ingest_document,
    get_chunk_details,
    register_knowledge_tools,
    _format_search_results,
    _format_ingest_result,
    _format_chunk_details,
    MAX_SEARCH_RESULTS,
)
from ingestforge.agent.react_engine import ToolResult
from ingestforge.agent.tool_registry import ToolCategory, create_registry

# Fixtures


@pytest.fixture
def mock_storage():
    """Create mock ChunkRepository."""
    storage = Mock()
    return storage


@pytest.fixture
def mock_pipeline():
    """Create mock Pipeline."""
    pipeline = Mock()
    return pipeline


@pytest.fixture
def mock_chunk():
    """Create mock ChunkRecord."""
    chunk = Mock()
    chunk.chunk_id = "chunk_123"
    chunk.content = "This is test content"
    chunk.document_id = "doc_456"
    chunk.source_file = "test.pdf"
    chunk.section_title = "Introduction"
    chunk.chunk_type = "paragraph"
    chunk.word_count = 50
    chunk.quality_score = 0.85
    chunk.entities = ["Entity1", "Entity2"]
    chunk.concepts = ["Concept1"]
    return chunk


@pytest.fixture
def mock_search_result():
    """Create mock SearchResult."""
    result = Mock()
    result.chunk_id = "chunk_123"
    result.content = "Test content for search result"
    result.score = 0.95
    result.source_file = "test.pdf"
    return result


@pytest.fixture
def mock_pipeline_result():
    """Create mock PipelineResult."""
    result = Mock()
    result.success = True
    result.document_id = "doc_789"
    result.chunks_created = 10
    result.chunks_indexed = 10
    result.processing_time_sec = 2.5
    result.error_message = None
    return result


# search_knowledge_base tests


class TestSearchKnowledgeBase:
    """Tests for search_knowledge_base tool."""

    def test_search_success(self, mock_storage, mock_search_result):
        """Test successful search."""
        mock_storage.search.return_value = [mock_search_result]

        output = search_knowledge_base(mock_storage, "test query", top_k=5)

        assert output.status == ToolResult.SUCCESS
        assert output.data is not None
        assert "Found 1 results" in output.data
        mock_storage.search.assert_called_once_with("test query", top_k=5)

    def test_search_empty_query(self, mock_storage):
        """Test search with empty query."""
        output = search_knowledge_base(mock_storage, "", top_k=5)

        assert output.status == ToolResult.ERROR
        assert "cannot be empty" in output.error_message

    def test_search_whitespace_query(self, mock_storage):
        """Test search with whitespace-only query."""
        output = search_knowledge_base(mock_storage, "   ", top_k=5)

        assert output.status == ToolResult.ERROR
        assert "cannot be empty" in output.error_message

    def test_search_no_results(self, mock_storage):
        """Test search returning no results."""
        mock_storage.search.return_value = []

        output = search_knowledge_base(mock_storage, "query", top_k=5)

        assert output.status == ToolResult.SUCCESS
        assert "No results found" in output.data

    def test_search_top_k_bounds(self, mock_storage, mock_search_result):
        """Test top_k parameter bounds."""
        mock_storage.search.return_value = [mock_search_result]

        # Test minimum
        output = search_knowledge_base(mock_storage, "query", top_k=0)
        mock_storage.search.assert_called_with("query", top_k=1)

        # Test maximum
        output = search_knowledge_base(mock_storage, "query", top_k=100)
        mock_storage.search.assert_called_with("query", top_k=MAX_SEARCH_RESULTS)

    def test_search_exception(self, mock_storage):
        """Test search with exception."""
        mock_storage.search.side_effect = Exception("Search failed")

        output = search_knowledge_base(mock_storage, "query", top_k=5)

        assert output.status == ToolResult.ERROR
        assert "Search error" in output.error_message


# ingest_document tests


class TestIngestDocument:
    """Tests for ingest_document tool."""

    def test_ingest_success(self, mock_pipeline, mock_pipeline_result, tmp_path):
        """Test successful document ingestion."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        mock_pipeline.process_file.return_value = mock_pipeline_result

        output = ingest_document(mock_pipeline, str(test_file))

        assert output.status == ToolResult.SUCCESS
        assert "ingested successfully" in output.data
        assert "doc_789" in output.data
        mock_pipeline.process_file.assert_called_once()

    def test_ingest_empty_path(self, mock_pipeline):
        """Test ingest with empty path."""
        output = ingest_document(mock_pipeline, "")

        assert output.status == ToolResult.ERROR
        assert "cannot be empty" in output.error_message

    def test_ingest_whitespace_path(self, mock_pipeline):
        """Test ingest with whitespace path."""
        output = ingest_document(mock_pipeline, "   ")

        assert output.status == ToolResult.ERROR
        assert "cannot be empty" in output.error_message

    def test_ingest_nonexistent_file(self, mock_pipeline):
        """Test ingest with nonexistent file."""
        output = ingest_document(mock_pipeline, "/nonexistent/file.pdf")

        assert output.status == ToolResult.ERROR
        assert "File not found" in output.error_message

    def test_ingest_directory(self, mock_pipeline, tmp_path):
        """Test ingest with directory instead of file."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        output = ingest_document(mock_pipeline, str(test_dir))

        assert output.status == ToolResult.ERROR
        assert "not a file" in output.error_message

    def test_ingest_pipeline_failure(self, mock_pipeline, tmp_path):
        """Test ingest when pipeline processing fails."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        failed_result = Mock()
        failed_result.success = False
        failed_result.error_message = "Processing error"
        mock_pipeline.process_file.return_value = failed_result

        output = ingest_document(mock_pipeline, str(test_file))

        assert output.status == ToolResult.ERROR
        assert "Processing error" in output.error_message

    def test_ingest_exception(self, mock_pipeline, tmp_path):
        """Test ingest with exception during processing."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        mock_pipeline.process_file.side_effect = Exception("Pipeline error")

        output = ingest_document(mock_pipeline, str(test_file))

        assert output.status == ToolResult.ERROR
        assert "Ingest error" in output.error_message


# get_chunk_details tests


class TestGetChunkDetails:
    """Tests for get_chunk_details tool."""

    def test_get_chunk_success(self, mock_storage, mock_chunk):
        """Test successful chunk retrieval."""
        mock_storage.get_chunk.return_value = mock_chunk

        output = get_chunk_details(mock_storage, "chunk_123")

        assert output.status == ToolResult.SUCCESS
        assert "chunk_123" in output.data
        assert "This is test content" in output.data
        mock_storage.get_chunk.assert_called_once_with("chunk_123")

    def test_get_chunk_empty_id(self, mock_storage):
        """Test get chunk with empty ID."""
        output = get_chunk_details(mock_storage, "")

        assert output.status == ToolResult.ERROR
        assert "cannot be empty" in output.error_message

    def test_get_chunk_whitespace_id(self, mock_storage):
        """Test get chunk with whitespace ID."""
        output = get_chunk_details(mock_storage, "   ")

        assert output.status == ToolResult.ERROR
        assert "cannot be empty" in output.error_message

    def test_get_chunk_not_found(self, mock_storage):
        """Test get chunk when chunk doesn't exist."""
        mock_storage.get_chunk.return_value = None

        output = get_chunk_details(mock_storage, "nonexistent")

        assert output.status == ToolResult.ERROR
        assert "not found" in output.error_message

    def test_get_chunk_exception(self, mock_storage):
        """Test get chunk with exception."""
        mock_storage.get_chunk.side_effect = Exception("Retrieval error")

        output = get_chunk_details(mock_storage, "chunk_123")

        assert output.status == ToolResult.ERROR
        assert "Retrieval error" in output.error_message


# Formatting function tests


class TestFormatSearchResults:
    """Tests for _format_search_results helper."""

    def test_format_single_result(self, mock_search_result):
        """Test formatting single search result."""
        results = [mock_search_result]

        formatted = _format_search_results(results)

        assert "Found 1 results" in formatted
        assert "chunk_123" in formatted
        assert "0.950" in formatted
        assert "test.pdf" in formatted

    def test_format_multiple_results(self):
        """Test formatting multiple search results."""
        results = []
        for i in range(3):
            result = Mock()
            result.chunk_id = f"chunk_{i}"
            result.content = f"Content {i}"
            result.score = 0.9 - (i * 0.1)
            result.source_file = f"file_{i}.pdf"
            results.append(result)

        formatted = _format_search_results(results)

        assert "Found 3 results" in formatted
        assert "chunk_0" in formatted
        assert "chunk_2" in formatted

    def test_format_long_content(self):
        """Test formatting truncates long content."""
        result = Mock()
        result.chunk_id = "chunk_123"
        result.content = "x" * 500  # Long content
        result.score = 0.9
        result.source_file = "test.pdf"

        formatted = _format_search_results([result])

        assert "..." in formatted
        assert len(result.content) > 200  # Original is long
        assert formatted.count("x") <= 200  # Formatted is truncated


class TestFormatIngestResult:
    """Tests for _format_ingest_result helper."""

    def test_format_result(self, mock_pipeline_result):
        """Test formatting pipeline result."""
        formatted = _format_ingest_result(mock_pipeline_result)

        assert "doc_789" in formatted
        assert "10" in formatted  # chunks created
        assert "2.50s" in formatted  # processing time


class TestFormatChunkDetails:
    """Tests for _format_chunk_details helper."""

    def test_format_chunk(self, mock_chunk):
        """Test formatting chunk details."""
        formatted = _format_chunk_details(mock_chunk)

        assert "chunk_123" in formatted
        assert "doc_456" in formatted
        assert "test.pdf" in formatted
        assert "Introduction" in formatted
        assert "0.85" in formatted  # quality score
        assert "This is test content" in formatted
        assert "Entity1, Entity2" in formatted
        assert "Concept1" in formatted

    def test_format_chunk_without_metadata(self):
        """Test formatting chunk without entities/concepts."""
        chunk = Mock()
        chunk.chunk_id = "chunk_123"
        chunk.content = "Content"
        chunk.document_id = "doc_456"
        chunk.source_file = "test.pdf"
        chunk.section_title = "Intro"
        chunk.chunk_type = "paragraph"
        chunk.word_count = 10
        chunk.quality_score = 0.5
        chunk.entities = []
        chunk.concepts = []

        formatted = _format_chunk_details(chunk)

        assert "chunk_123" in formatted
        assert "Content" in formatted
        # Should not have entity/concept lines
        assert formatted.count("Entities:") == 0
        assert formatted.count("Concepts:") == 0

    def test_format_long_content(self):
        """Test formatting truncates very long content."""
        chunk = Mock()
        chunk.chunk_id = "chunk_123"
        chunk.content = "x" * 15000  # Very long content
        chunk.document_id = "doc_456"
        chunk.source_file = "test.pdf"
        chunk.section_title = "Intro"
        chunk.chunk_type = "paragraph"
        chunk.word_count = 5000
        chunk.quality_score = 0.8
        chunk.entities = []
        chunk.concepts = []

        formatted = _format_chunk_details(chunk)

        assert "..." in formatted
        # Content should be truncated
        from ingestforge.agent.knowledge_tools import MAX_CHUNK_CONTENT_LENGTH

        assert formatted.count("x") <= MAX_CHUNK_CONTENT_LENGTH


# register_knowledge_tools tests


class TestRegisterKnowledgeTools:
    """Tests for register_knowledge_tools function."""

    def test_register_all_tools(self, mock_storage, mock_pipeline):
        """Test registering all knowledge tools."""
        registry = create_registry()

        count = register_knowledge_tools(registry, mock_storage, mock_pipeline)

        assert count == 3
        assert "search_knowledge_base" in registry.tool_names
        assert "ingest_document" in registry.tool_names
        assert "get_chunk_details" in registry.tool_names

    def test_registered_tools_work(self, mock_storage, mock_pipeline):
        """Test that registered tools are executable."""
        registry = create_registry()
        mock_storage.search.return_value = []

        register_knowledge_tools(registry, mock_storage, mock_pipeline)

        # Test search tool
        search_tool = registry.get("search_knowledge_base")
        assert search_tool is not None
        output = search_tool.execute(query="test")
        assert output.status == ToolResult.SUCCESS

    def test_tool_categories(self, mock_storage, mock_pipeline):
        """Test tools have correct categories."""
        registry = create_registry()

        register_knowledge_tools(registry, mock_storage, mock_pipeline)

        search_tools = registry.list_by_category(ToolCategory.SEARCH)
        assert len(search_tools) == 1
        assert search_tools[0].name == "search_knowledge_base"

        retrieve_tools = registry.list_by_category(ToolCategory.RETRIEVE)
        assert len(retrieve_tools) == 1
        assert retrieve_tools[0].name == "get_chunk_details"

    def test_tool_parameters(self, mock_storage, mock_pipeline):
        """Test tools have proper parameters defined."""
        registry = create_registry()

        register_knowledge_tools(registry, mock_storage, mock_pipeline)

        search_tool = registry.get("search_knowledge_base")
        params = search_tool.metadata.parameters

        assert len(params) == 2
        assert params[0].name == "query"
        assert params[0].required is True
        assert params[1].name == "top_k"
        assert params[1].required is False
        assert params[1].default == 5


# Integration tests


class TestKnowledgeToolsIntegration:
    """Integration tests for knowledge tools."""

    def test_search_and_retrieve_flow(
        self, mock_storage, mock_chunk, mock_search_result
    ):
        """Test searching then retrieving chunk details."""
        # Setup mocks
        mock_storage.search.return_value = [mock_search_result]
        mock_storage.get_chunk.return_value = mock_chunk

        # Search
        search_output = search_knowledge_base(mock_storage, "test query")
        assert search_output.status == ToolResult.SUCCESS
        assert "chunk_123" in search_output.data

        # Get details
        details_output = get_chunk_details(mock_storage, "chunk_123")
        assert details_output.status == ToolResult.SUCCESS
        assert "This is test content" in details_output.data

    def test_ingest_then_search(
        self,
        mock_storage,
        mock_pipeline,
        mock_pipeline_result,
        mock_search_result,
        tmp_path,
    ):
        """Test ingesting document then searching for it."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        # Ingest
        mock_pipeline.process_file.return_value = mock_pipeline_result
        ingest_output = ingest_document(mock_pipeline, str(test_file))
        assert ingest_output.status == ToolResult.SUCCESS

        # Search
        mock_storage.search.return_value = [mock_search_result]
        search_output = search_knowledge_base(mock_storage, "test query")
        assert search_output.status == ToolResult.SUCCESS
