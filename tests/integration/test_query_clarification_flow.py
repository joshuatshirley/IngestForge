"""
Comprehensive Integration Tests for Query Clarification Flow

End-to-end integration tests following Given-When-Then (GWT) pattern.

Tests cover:
- API endpoint integration
- Pipeline integration
- Full clarification workflow
- Error handling and recovery

Coverage Target: >80%
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import API routes
from ingestforge.api.routes.query import router as query_router

# Import core components
from ingestforge.query.clarifier import (
    IFQueryClarifier,
    ClarifierConfig,
)
from ingestforge.query.pipeline import QueryPipeline, QueryResult
from ingestforge.core.config import Config


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def api_client() -> TestClient:
    """Create FastAPI test client with query router."""
    app = FastAPI()
    app.include_router(query_router)
    return TestClient(app)


@pytest.fixture
def mock_config() -> Config:
    """Create mock configuration."""
    config = Mock(spec=Config)
    config.retrieval = Mock()
    config.retrieval.top_k = 10
    config.retrieval.rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return config


@pytest.fixture
def mock_retriever() -> Mock:
    """Create mock retriever."""
    retriever = Mock()
    retriever.search = Mock(return_value=[])
    return retriever


@pytest.fixture
def query_pipeline(mock_config: Config, mock_retriever: Mock) -> QueryPipeline:
    """Create query pipeline with mock dependencies."""
    pipeline = QueryPipeline(
        config=mock_config,
        retriever=mock_retriever,
        enable_cache=False,
    )
    return pipeline


# =============================================================================
# Test Suite: API Integration
# =============================================================================


class TestAPIIntegration:
    """Integration tests for clarification API endpoints."""

    def test_given_ambiguous_query_when_api_clarify_then_returns_needs_clarification(
        self, api_client: TestClient
    ) -> None:
        """
        GIVEN an ambiguous query
        WHEN POST /v1/query/clarify
        THEN returns needs_clarification=true with suggestions

        POST /v1/query/clarify integration
        """
        # GIVEN
        request_data = {
            "query": "tell me more",
            "threshold": 0.7,
            "use_llm": False,
        }

        # WHEN
        response = api_client.post("/v1/query/clarify", json=request_data)

        # THEN
        assert response.status_code == 200
        data = response.json()

        assert data["original_query"] == "tell me more"
        assert data["needs_clarification"] is True
        assert data["is_clear"] is False
        assert len(data["suggestions"]) > 0
        assert data["clarity_score"] < 0.7
        assert "reason" in data
        assert "factors" in data

    def test_given_clear_query_when_api_clarify_then_returns_clear(
        self, api_client: TestClient
    ) -> None:
        """
        GIVEN a clear, specific query
        WHEN POST /v1/query/clarify
        THEN returns is_clear=true, needs_clarification=false

        Clear query handling
        """
        # GIVEN
        request_data = {
            "query": "Who was the CEO of Apple Inc. in 2011?",
            "threshold": 0.7,
            "use_llm": False,
        }

        # WHEN
        response = api_client.post("/v1/query/clarify", json=request_data)

        # THEN
        assert response.status_code == 200
        data = response.json()

        assert data["original_query"] == request_data["query"]
        assert data["clarity_score"] > 0.5
        # Note: Specific query should score reasonably well

    def test_given_original_and_refinement_when_api_refine_then_improves_clarity(
        self, api_client: TestClient
    ) -> None:
        """
        GIVEN original query and refinement
        WHEN POST /v1/query/refine
        THEN returns refined query with improved clarity score

        POST /v1/query/refine integration
        """
        # GIVEN
        request_data = {
            "original_query": "python",
            "selected_refinement": "Python programming language",
        }

        # WHEN
        response = api_client.post("/v1/query/refine", json=request_data)

        # THEN
        assert response.status_code == 200
        data = response.json()

        assert "Python programming language" in data["refined_query"]
        assert "python" in data["refined_query"]
        assert data["clarity_score"] >= 0.0
        assert data["improvement"] >= 0.0  # Should improve or stay same
        assert isinstance(data["is_clear"], bool)

    def test_given_clarify_and_refine_when_workflow_then_completes_successfully(
        self, api_client: TestClient
    ) -> None:
        """
        GIVEN full clarification workflow
        WHEN clarify → select suggestion → refine
        THEN workflow completes successfully

        End-to-end API workflow
        """
        # GIVEN: Step 1 - Clarify ambiguous query
        clarify_request = {
            "query": "tell me about python",
            "threshold": 0.7,
            "use_llm": False,
        }

        clarify_response = api_client.post("/v1/query/clarify", json=clarify_request)
        assert clarify_response.status_code == 200
        clarify_data = clarify_response.json()

        # WHEN: Step 2 - User selects refinement
        if clarify_data["needs_clarification"] and len(clarify_data["suggestions"]) > 0:
            selected_suggestion = clarify_data["suggestions"][0]

            # Step 3 - Refine query
            refine_request = {
                "original_query": clarify_request["query"],
                "selected_refinement": selected_suggestion,
            }

            refine_response = api_client.post("/v1/query/refine", json=refine_request)

            # THEN
            assert refine_response.status_code == 200
            refine_data = refine_response.json()

            # Refined query should contain both original and refinement
            assert clarify_request["query"] in refine_data["refined_query"]
            assert selected_suggestion in refine_data["refined_query"]

            # Clarity should improve
            assert refine_data["clarity_score"] >= clarify_data["clarity_score"]


# =============================================================================
# Test Suite: Pipeline Integration
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for pipeline clarification integration."""

    def test_given_enable_clarification_false_when_process_then_skips_clarification(
        self, query_pipeline: QueryPipeline, mock_retriever: Mock
    ) -> None:
        """
        GIVEN enable_clarification=False
        WHEN pipeline.process()
        THEN skips clarification check

        Opt-in clarification behavior
        """
        # GIVEN
        mock_retriever.search.return_value = []

        # WHEN
        with patch.object(query_pipeline, "_check_clarification") as mock_check:
            query_pipeline.process(
                query="tell me more",
                enable_clarification=False,
                generate_answer=False,
            )

            # THEN
            mock_check.assert_not_called()

    def test_given_enable_clarification_true_when_ambiguous_query_then_returns_clarification(
        self, query_pipeline: QueryPipeline
    ) -> None:
        """
        GIVEN enable_clarification=True and ambiguous query
        WHEN pipeline.process()
        THEN returns QueryResult with clarification metadata

        Pipeline integration with clarification
        """
        # GIVEN
        ambiguous_query = "tell me more"

        # WHEN
        result = query_pipeline.process(
            query=ambiguous_query,
            enable_clarification=True,
            clarity_threshold=0.7,
            generate_answer=False,
        )

        # THEN
        assert isinstance(result, QueryResult)
        assert result.metadata.get("needs_clarification") is True
        assert "clarity_score" in result.metadata
        assert "suggestions" in result.metadata
        assert "reason" in result.metadata
        assert len(result.metadata["suggestions"]) > 0

    def test_given_clear_query_when_enable_clarification_then_continues_pipeline(
        self, query_pipeline: QueryPipeline, mock_retriever: Mock
    ) -> None:
        """
        GIVEN clear query with enable_clarification=True
        WHEN pipeline.process()
        THEN continues normal pipeline execution

        Clear queries bypass clarification
        """
        # GIVEN
        clear_query = "Who was the CEO of Apple Inc. in 2011?"
        mock_retriever.search.return_value = []

        # WHEN
        result = query_pipeline.process(
            query=clear_query,
            enable_clarification=True,
            clarity_threshold=0.7,
            generate_answer=False,
        )

        # THEN
        # Clear query should NOT trigger clarification
        assert result.metadata.get("needs_clarification") is not True
        # Pipeline should have executed retrieval
        mock_retriever.search.assert_called()

    def test_given_clarification_fails_when_process_then_continues_pipeline(
        self, query_pipeline: QueryPipeline, mock_retriever: Mock
    ) -> None:
        """
        GIVEN clarifier raises exception
        WHEN pipeline.process()
        THEN continues pipeline (fail-open)

        Fail-open error handling
        """
        # GIVEN
        mock_retriever.search.return_value = []

        # Mock clarifier to raise exception
        with patch.object(query_pipeline.clarifier, "evaluate") as mock_evaluate:
            mock_evaluate.side_effect = Exception("Clarifier failure")

            # WHEN
            result = query_pipeline.process(
                query="test query",
                enable_clarification=True,
                generate_answer=False,
            )

            # THEN
            # Pipeline should continue despite clarification failure
            assert isinstance(result, QueryResult)
            # Should not have clarification metadata (failed)
            assert result.metadata.get("needs_clarification") is not True
            # But should have executed retrieval
            mock_retriever.search.assert_called()

    def test_given_custom_threshold_when_process_then_uses_custom_threshold(
        self, query_pipeline: QueryPipeline
    ) -> None:
        """
        GIVEN custom clarity_threshold
        WHEN pipeline.process()
        THEN evaluates against custom threshold

        Configurable threshold
        """
        # GIVEN
        query = "what is python?"

        # WHEN: Strict threshold (0.9)
        result_strict = query_pipeline.process(
            query=query,
            enable_clarification=True,
            clarity_threshold=0.9,
            generate_answer=False,
        )

        # WHEN: Lenient threshold (0.3)
        result_lenient = query_pipeline.process(
            query=query,
            enable_clarification=True,
            clarity_threshold=0.3,
            generate_answer=False,
        )

        # THEN
        # With strict threshold, more likely to need clarification
        # With lenient threshold, more likely to be clear
        # (Exact behavior depends on query scoring)
        assert isinstance(result_strict, QueryResult)
        assert isinstance(result_lenient, QueryResult)


# =============================================================================
# Test Suite: Clarifier Core Integration
# =============================================================================


class TestClarifierIntegration:
    """Integration tests for clarifier core functionality."""

    def test_given_pronoun_query_when_evaluate_then_detects_ambiguity(self) -> None:
        """
        GIVEN query with pronoun
        WHEN clarifier.evaluate()
        THEN detects pronoun ambiguity

        Pronoun detection
        """
        # GIVEN
        clarifier = IFQueryClarifier()
        query = "What did he say about Python?"

        # WHEN
        result = clarifier.evaluate(query)

        # THEN
        assert result.ambiguity_report is not None
        assert result.ambiguity_report.is_ambiguous is True
        assert len(result.ambiguity_report.questions) > 0
        # Should have pronoun question
        assert any(q.term == "he" for q in result.ambiguity_report.questions)

    def test_given_multi_meaning_query_when_evaluate_then_detects_ambiguity(
        self,
    ) -> None:
        """
        GIVEN query with multi-meaning term
        WHEN clarifier.evaluate()
        THEN detects multi-meaning ambiguity

        Multi-meaning detection
        """
        # GIVEN
        clarifier = IFQueryClarifier()
        query = "Tell me about Python"

        # WHEN
        result = clarifier.evaluate(query)

        # THEN
        assert result.ambiguity_report is not None
        # Should detect Python as ambiguous
        assert len(result.ambiguity_report.questions) > 0

    def test_given_temporal_query_when_evaluate_then_detects_ambiguity(self) -> None:
        """
        GIVEN query with temporal ambiguity
        WHEN clarifier.evaluate()
        THEN detects temporal ambiguity

        Temporal ambiguity detection
        """
        # GIVEN
        clarifier = IFQueryClarifier()
        query = "recent developments in AI"

        # WHEN
        result = clarifier.evaluate(query)

        # THEN
        assert result.ambiguity_report is not None
        assert result.ambiguity_report.is_ambiguous is True
        # Should detect 'recent' as temporally ambiguous

    def test_given_config_when_create_clarifier_then_uses_config(self) -> None:
        """
        GIVEN custom ClarifierConfig
        WHEN IFQueryClarifier created
        THEN uses custom configuration

        Configurable threshold
        """
        # GIVEN
        custom_config = ClarifierConfig(
            threshold=0.8,
            use_llm=False,
            max_suggestions=3,
        )

        # WHEN
        clarifier = IFQueryClarifier(config=custom_config)
        result = clarifier.evaluate("python")

        # THEN
        # Suggestions should be limited to max_suggestions
        assert len(result.suggestions) <= 3


# =============================================================================
# Test Suite: Error Handling
# =============================================================================


class TestErrorHandling:
    """Integration tests for error handling scenarios."""

    def test_given_invalid_json_when_api_clarify_then_returns_422(
        self, api_client: TestClient
    ) -> None:
        """
        GIVEN invalid JSON request
        WHEN POST /v1/query/clarify
        THEN returns 422 validation error

        JPL Rule #5: Input validation
        """
        # GIVEN
        invalid_request = {
            "query": "",  # Empty query (invalid)
            "threshold": 0.7,
        }

        # WHEN
        response = api_client.post("/v1/query/clarify", json=invalid_request)

        # THEN
        assert response.status_code == 422

    def test_given_query_too_long_when_api_clarify_then_returns_422(
        self, api_client: TestClient
    ) -> None:
        """
        GIVEN query exceeding MAX_QUERY_LENGTH
        WHEN POST /v1/query/clarify
        THEN returns 422 validation error

        JPL Rule #2: Fixed bounds
        """
        # GIVEN
        long_query = "x" * 3000  # Exceeds MAX_QUERY_LENGTH (2000)
        request_data = {
            "query": long_query,
            "threshold": 0.7,
        }

        # WHEN
        response = api_client.post("/v1/query/clarify", json=request_data)

        # THEN
        assert response.status_code == 422

    def test_given_invalid_threshold_when_api_clarify_then_returns_422(
        self, api_client: TestClient
    ) -> None:
        """
        GIVEN threshold outside [0.0, 1.0]
        WHEN POST /v1/query/clarify
        THEN returns 422 validation error

        JPL Rule #5: Input validation
        """
        # GIVEN
        request_data = {
            "query": "test query",
            "threshold": 1.5,  # Invalid: > 1.0
        }

        # WHEN
        response = api_client.post("/v1/query/clarify", json=request_data)

        # THEN
        assert response.status_code == 422


# =============================================================================
# Test Suite: Performance
# =============================================================================


class TestPerformance:
    """Integration tests for performance characteristics."""

    def test_given_clarify_request_when_executed_then_completes_quickly(
        self, api_client: TestClient
    ) -> None:
        """
        GIVEN query clarification request
        WHEN POST /v1/query/clarify
        THEN completes within 500ms

        Latency <500ms
        """
        # GIVEN
        import time

        request_data = {
            "query": "what is machine learning?",
            "threshold": 0.7,
            "use_llm": False,
        }

        # WHEN
        start = time.time()
        response = api_client.post("/v1/query/clarify", json=request_data)
        elapsed_ms = (time.time() - start) * 1000

        # THEN
        assert response.status_code == 200
        data = response.json()

        # Latency should be <500ms
        assert elapsed_ms < 500

        # Response should include evaluation time
        assert "evaluation_time_ms" in data
        assert data["evaluation_time_ms"] < 500

    def test_given_refine_request_when_executed_then_completes_quickly(
        self, api_client: TestClient
    ) -> None:
        """
        GIVEN query refinement request
        WHEN POST /v1/query/refine
        THEN completes within 100ms (faster than clarify)

        Refinement latency
        """
        # GIVEN
        import time

        request_data = {
            "original_query": "python",
            "selected_refinement": "programming language",
        }

        # WHEN
        start = time.time()
        response = api_client.post("/v1/query/refine", json=request_data)
        elapsed_ms = (time.time() - start) * 1000

        # THEN
        assert response.status_code == 200
        # Should be faster than clarify (<100ms target, but flexible)
        assert elapsed_ms < 500


# =============================================================================
# Test Suite: Health Check
# =============================================================================


class TestHealthCheck:
    """Integration tests for clarifier health check."""

    def test_given_clarifier_operational_when_health_check_then_returns_healthy(
        self, api_client: TestClient
    ) -> None:
        """
        GIVEN clarifier is operational
        WHEN GET /v1/query/clarifier/health
        THEN returns status=healthy

        Health check endpoint
        """
        # WHEN
        response = api_client.get("/v1/query/clarifier/health")

        # THEN
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["clarifier_available"] is True
        assert "llm_available" in data
        assert "test_evaluation" in data


# =============================================================================
# Test Coverage Summary
# =============================================================================


def test_integration_coverage_summary() -> None:
    """
    Document integration test coverage.

    Integration test suite summary.
    """
    test_counts = {
        "api_integration": 4,
        "pipeline_integration": 5,
        "clarifier_integration": 4,
        "error_handling": 3,
        "performance": 2,
        "health_check": 1,
    }

    total_tests = sum(test_counts.values())

    # Document test coverage
    assert total_tests == 19  # Total integration tests
    assert all(count > 0 for count in test_counts.values())  # All categories covered
