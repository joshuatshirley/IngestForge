"""Comprehensive GWT unit tests for HybridRetriever metadata filtering.

NL Search Redesign.
JPL Compliance: Rule #2, #4, #9.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from ingestforge.retrieval.hybrid import HybridRetriever


@pytest.fixture
def mock_storage():
    return MagicMock()


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.retrieval.hybrid.bm25_weight = 0.4
    config.retrieval.hybrid.semantic_weight = 0.6
    return config


@pytest.fixture
def retriever(mock_config, mock_storage):
    return HybridRetriever(mock_config, mock_storage)


# =============================================================================
# HYBRID RETRIEVER FILTERING TESTS
# =============================================================================


def test_search_passes_metadata_filter(retriever, mock_storage):
    """GIVEN a search query with metadata filters
    WHEN retriever.search is called
    THEN it passes the filter down to underlying retrievers.
    """
    filters = {"doc_type": "PDF", "source": "ArXiv"}

    # Mock the underlying search execution to avoid full pipeline run
    with patch.object(retriever, "_execute_searches") as mock_exec:
        mock_exec.return_value = ([], [])

        retriever.search(query="test query", metadata_filter=filters)

        # Verify filters were passed to _execute_searches
        args, kwargs = mock_exec.call_args
        assert kwargs["metadata_filter"] == filters


def test_execute_searches_respects_parallel_filters(retriever):
    """GIVEN parallel search enabled
    WHEN _execute_searches is called with metadata_filter
    THEN it passes filters to _search_parallel.
    """
    filters = {"year": 2024}

    with patch.object(retriever, "_search_parallel") as mock_parallel:
        mock_parallel.return_value = ([], [])

        retriever._execute_searches(
            query="test",
            candidate_k=10,
            use_parallel=True,
            library_filter=None,
            metadata_filter=filters,
        )

        assert mock_parallel.call_args[1]["metadata_filter"] == filters


def test_search_parallel_tasks_receive_filters(retriever):
    """GIVEN parallel search execution
    WHEN _search_parallel is called
    THEN it submits search tasks with metadata_filters.
    """
    filters = {"author": "Einstein"}

    # Mock the underlying retrievers
    retriever._bm25 = MagicMock()
    retriever._semantic = MagicMock()

    with patch("concurrent.futures.ThreadPoolExecutor.submit") as mock_submit:
        retriever._search_parallel(query="relativity", top_k=5, metadata_filter=filters)

        # Verify both BM25 and Semantic search tasks were submitted with filters
        # Submit is called twice (BM25, Semantic)
        assert mock_submit.call_count == 2
        for call in mock_submit.call_args_list:
            assert call.args[4] == filters  # metadata_filter is 5th arg
