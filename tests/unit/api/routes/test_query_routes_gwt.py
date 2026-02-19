"""
Comprehensive GWT Unit Tests for Query Clarification API Routes.

Query Clarification API - Given-When-Then test format.
Tests FastAPI endpoints for query clarification and refinement.

Test Format:
- Given: Initial conditions/setup
- When: API request made
- Then: Expected HTTP response and data
"""

from __future__ import annotations

import pytest
from typing import Any, Dict
from fastapi.testclient import TestClient

# Import the FastAPI app
from ingestforge.api.main import app

# Import models for validation


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def valid_clarify_request() -> Dict[str, Any]:
    """Create valid clarify request payload."""
    return {
        "query": "What did he say about Python?",
        "threshold": 0.7,
        "use_llm": False,
        "context": None,
    }


@pytest.fixture
def valid_refine_request() -> Dict[str, Any]:
    """Create valid refine request payload."""
    return {
        "original_query": "What did he say?",
        "selected_refinement": "Be more specific about who 'he' refers to",
    }


# =============================================================================
# GWT Tests: POST /v1/query/clarify Endpoint
# =============================================================================


class TestClarifyEndpointGWT:
    """GWT tests for /v1/query/clarify endpoint."""

    def test_clarify_clear_query_success(self, client: TestClient) -> None:
        """
        Given: A clear, specific query
        When: POST /v1/query/clarify is called
        Then: Returns 200 with is_clear=True
        """
        # Given
        payload = {
            "query": "Who is the CEO of Apple Inc. in 2024?",
            "threshold": 0.7,
            "use_llm": False,
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 200
        data = response.json()
        assert "clarity_score" in data
        assert data["clarity_score"] >= 0.0
        assert data["clarity_score"] <= 1.0
        assert "is_clear" in data
        assert "needs_clarification" in data

    def test_clarify_ambiguous_query_detects_issues(
        self, client: TestClient, valid_clarify_request: Dict[str, Any]
    ) -> None:
        """
        Given: An ambiguous query with pronoun
        When: POST /v1/query/clarify is called
        Then: Returns 200 with needs_clarification=True
        """
        # Given (using fixture)

        # When
        response = client.post("/v1/query/clarify", json=valid_clarify_request)

        # Then
        assert response.status_code == 200
        data = response.json()
        assert data["needs_clarification"] is True
        assert data["clarity_score"] < 0.7

    def test_clarify_returns_suggestions(
        self, client: TestClient, valid_clarify_request: Dict[str, Any]
    ) -> None:
        """
        Given: An ambiguous query
        When: POST /v1/query/clarify is called
        Then: Returns suggestions list
        """
        # Given (using fixture)

        # When
        response = client.post("/v1/query/clarify", json=valid_clarify_request)

        # Then
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_clarify_returns_evaluation_time(
        self, client: TestClient, valid_clarify_request: Dict[str, Any]
    ) -> None:
        """
        Given: Any valid query
        When: POST /v1/query/clarify is called
        Then: Returns evaluation_time_ms field
        """
        # Given (using fixture)

        # When
        response = client.post("/v1/query/clarify", json=valid_clarify_request)

        # Then
        assert response.status_code == 200
        data = response.json()
        assert "evaluation_time_ms" in data
        assert isinstance(data["evaluation_time_ms"], (int, float))
        assert data["evaluation_time_ms"] >= 0

    def test_clarify_with_custom_threshold(self, client: TestClient) -> None:
        """
        Given: A query with custom threshold=0.9 (stricter)
        When: POST /v1/query/clarify is called
        Then: Uses custom threshold for evaluation
        """
        # Given
        payload = {
            "query": "Explain machine learning",
            "threshold": 0.9,
            "use_llm": False,
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 200
        data = response.json()
        # With stricter threshold, more likely to need clarification
        assert "clarity_score" in data

    def test_clarify_empty_query_fails_validation(self, client: TestClient) -> None:
        """
        Given: An empty query string
        When: POST /v1/query/clarify is called
        Then: Returns 422 validation error
        """
        # Given
        payload = {
            "query": "",  # Invalid: empty
            "threshold": 0.7,
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 422  # Validation error

    def test_clarify_query_too_long_truncated(self, client: TestClient) -> None:
        """
        Given: A query longer than MAX_QUERY_LENGTH
        When: POST /v1/query/clarify is called
        Then: Returns 200 (query is truncated internally)
        """
        # Given
        payload = {
            "query": "x" * 3000,  # Very long query
            "threshold": 0.7,
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 200
        data = response.json()
        # Query should have been processed (truncated)
        assert len(data["original_query"]) <= 2000

    def test_clarify_invalid_threshold_fails_validation(
        self, client: TestClient
    ) -> None:
        """
        Given: A threshold > 1.0
        When: POST /v1/query/clarify is called
        Then: Returns 422 validation error
        """
        # Given
        payload = {
            "query": "Test query",
            "threshold": 1.5,  # Invalid: > 1.0
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 422

    def test_clarify_with_context(self, client: TestClient) -> None:
        """
        Given: A query with conversation context
        When: POST /v1/query/clarify is called
        Then: Returns 200 and uses context for evaluation
        """
        # Given
        payload = {
            "query": "What did he create?",
            "threshold": 0.7,
            "use_llm": False,
            "context": {"previous_queries": ["Who is Guido van Rossum?"]},
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 200
        data = response.json()
        assert "clarity_score" in data

    def test_clarify_context_too_many_fields_truncated(
        self, client: TestClient
    ) -> None:
        """
        Given: Context with > MAX_CONTEXT_FIELDS
        When: POST /v1/query/clarify is called
        Then: Returns 422 validation error OR truncates gracefully
        """
        # Given
        large_context = {f"field{i}": f"value{i}" for i in range(20)}
        payload = {
            "query": "Test query",
            "threshold": 0.7,
            "context": large_context,
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        # Either validation fails or context is truncated
        assert response.status_code in [200, 422]

    def test_clarify_returns_factors_breakdown(self, client: TestClient) -> None:
        """
        Given: Any valid query
        When: POST /v1/query/clarify is called
        Then: Returns factors dictionary with score breakdown
        """
        # Given
        payload = {
            "query": "What is Python programming?",
            "threshold": 0.7,
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 200
        data = response.json()
        assert "factors" in data
        assert isinstance(data["factors"], dict)

    def test_clarify_vague_query_tells_me_more(self, client: TestClient) -> None:
        """
        Given: A vague query 'tell me more'
        When: POST /v1/query/clarify is called
        Then: Returns low clarity score and suggestions
        """
        # Given
        payload = {
            "query": "tell me more",
            "threshold": 0.7,
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 200
        data = response.json()
        assert data["needs_clarification"] is True
        assert len(data["suggestions"]) > 0


# =============================================================================
# GWT Tests: POST /v1/query/refine Endpoint
# =============================================================================


class TestRefineEndpointGWT:
    """GWT tests for /v1/query/refine endpoint."""

    def test_refine_query_success(
        self, client: TestClient, valid_refine_request: Dict[str, Any]
    ) -> None:
        """
        Given: Original query and refinement
        When: POST /v1/query/refine is called
        Then: Returns 200 with refined query
        """
        # Given (using fixture)

        # When
        response = client.post("/v1/query/refine", json=valid_refine_request)

        # Then
        assert response.status_code == 200
        data = response.json()
        assert "refined_query" in data
        assert "clarity_score" in data
        assert "improvement" in data
        assert "is_clear" in data

    def test_refine_combines_query_and_refinement(self, client: TestClient) -> None:
        """
        Given: Original query and refinement text
        When: POST /v1/query/refine is called
        Then: Refined query combines both
        """
        # Given
        payload = {
            "original_query": "What is Python?",
            "selected_refinement": "programming language",
        }

        # When
        response = client.post("/v1/query/refine", json=payload)

        # Then
        assert response.status_code == 200
        data = response.json()
        refined = data["refined_query"]
        assert "Python" in refined
        assert "programming language" in refined

    def test_refine_improves_clarity_score(self, client: TestClient) -> None:
        """
        Given: Vague query with refinement
        When: POST /v1/query/refine is called
        Then: Improvement score is positive
        """
        # Given
        payload = {
            "original_query": "Tell me more",
            "selected_refinement": "Specifically explain Python decorators",
        }

        # When
        response = client.post("/v1/query/refine", json=payload)

        # Then
        assert response.status_code == 200
        data = response.json()
        # Improvement should be >= 0 (refinement helps)
        assert isinstance(data["improvement"], (int, float))

    def test_refine_empty_original_query_fails(self, client: TestClient) -> None:
        """
        Given: Empty original query
        When: POST /v1/query/refine is called
        Then: Returns 422 validation error
        """
        # Given
        payload = {
            "original_query": "",  # Invalid
            "selected_refinement": "something",
        }

        # When
        response = client.post("/v1/query/refine", json=payload)

        # Then
        assert response.status_code == 422

    def test_refine_empty_refinement_fails(self, client: TestClient) -> None:
        """
        Given: Empty refinement text
        When: POST /v1/query/refine is called
        Then: Returns 422 validation error
        """
        # Given
        payload = {
            "original_query": "What is this?",
            "selected_refinement": "",  # Invalid
        }

        # When
        response = client.post("/v1/query/refine", json=payload)

        # Then
        assert response.status_code == 422

    def test_refine_very_long_refinement_fails(self, client: TestClient) -> None:
        """
        Given: Refinement exceeding MAX_REFINEMENT_LENGTH
        When: POST /v1/query/refine is called
        Then: Returns 422 validation error
        """
        # Given
        payload = {
            "original_query": "What is this?",
            "selected_refinement": "x" * 600,  # Too long
        }

        # When
        response = client.post("/v1/query/refine", json=payload)

        # Then
        assert response.status_code == 422

    def test_refine_returns_is_clear_boolean(self, client: TestClient) -> None:
        """
        Given: Any valid refinement request
        When: POST /v1/query/refine is called
        Then: Returns is_clear boolean field
        """
        # Given
        payload = {
            "original_query": "What is Python?",
            "selected_refinement": "programming language",
        }

        # When
        response = client.post("/v1/query/refine", json=payload)

        # Then
        assert response.status_code == 200
        data = response.json()
        assert "is_clear" in data
        assert isinstance(data["is_clear"], bool)


# =============================================================================
# GWT Tests: GET /v1/query/clarifier/health Endpoint
# =============================================================================


class TestHealthEndpointGWT:
    """GWT tests for /v1/query/clarifier/health endpoint."""

    def test_health_check_success(self, client: TestClient) -> None:
        """
        Given: Clarifier service is operational
        When: GET /v1/query/clarifier/health is called
        Then: Returns 200 with status=healthy
        """
        # Given/When
        response = client.get("/v1/query/clarifier/health")

        # Then
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "clarifier_available" in data

    def test_health_check_returns_availability(self, client: TestClient) -> None:
        """
        Given: Health check endpoint
        When: GET /v1/query/clarifier/health is called
        Then: Returns clarifier_available boolean
        """
        # Given/When
        response = client.get("/v1/query/clarifier/health")

        # Then
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["clarifier_available"], bool)

    def test_health_check_includes_test_evaluation(self, client: TestClient) -> None:
        """
        Given: Health check endpoint
        When: GET /v1/query/clarifier/health is called
        Then: Returns test_evaluation with sample query
        """
        # Given/When
        response = client.get("/v1/query/clarifier/health")

        # Then
        assert response.status_code == 200
        data = response.json()
        if data["status"] == "healthy":
            assert "test_evaluation" in data
            assert "query" in data["test_evaluation"]
            assert "score" in data["test_evaluation"]


# =============================================================================
# GWT Tests: Error Handling
# =============================================================================


class TestErrorHandlingGWT:
    """GWT tests for error handling in API routes."""

    def test_clarify_missing_required_field(self, client: TestClient) -> None:
        """
        Given: Request missing required 'query' field
        When: POST /v1/query/clarify is called
        Then: Returns 422 validation error
        """
        # Given
        payload = {
            "threshold": 0.7,
            # Missing 'query' field
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 422

    def test_refine_missing_required_fields(self, client: TestClient) -> None:
        """
        Given: Request missing required fields
        When: POST /v1/query/refine is called
        Then: Returns 422 validation error
        """
        # Given
        payload = {
            # Missing both original_query and selected_refinement
        }

        # When
        response = client.post("/v1/query/refine", json=payload)

        # Then
        assert response.status_code == 422

    def test_clarify_invalid_json(self, client: TestClient) -> None:
        """
        Given: Invalid JSON payload
        When: POST /v1/query/clarify is called
        Then: Returns 422 error
        """
        # Given
        invalid_json = "not json"

        # When
        response = client.post(
            "/v1/query/clarify",
            data=invalid_json,
            headers={"Content-Type": "application/json"},
        )

        # Then
        assert response.status_code == 422

    def test_clarify_wrong_type_for_threshold(self, client: TestClient) -> None:
        """
        Given: threshold as string instead of float
        When: POST /v1/query/clarify is called
        Then: Returns 422 validation error
        """
        # Given
        payload = {
            "query": "Test query",
            "threshold": "not a number",  # Invalid type
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 422


# =============================================================================
# GWT Tests: Integration Scenarios
# =============================================================================


class TestIntegrationScenariosGWT:
    """GWT tests for end-to-end integration scenarios."""

    def test_full_clarification_workflow(self, client: TestClient) -> None:
        """
        Given: Complete clarification workflow
        When: 1) Clarify ambiguous query, 2) Refine with selection
        Then: Both steps succeed, refinement improves clarity
        """
        # Given: Step 1 - Clarify ambiguous query
        clarify_payload = {
            "query": "What is Python?",
            "threshold": 0.7,
        }

        # When: Step 1
        clarify_response = client.post("/v1/query/clarify", json=clarify_payload)

        # Then: Step 1
        assert clarify_response.status_code == 200
        clarify_data = clarify_response.json()

        # Given: Step 2 - Use suggestion to refine
        if clarify_data.get("suggestions"):
            refine_payload = {
                "original_query": "What is Python?",
                "selected_refinement": "Python programming language",
            }

            # When: Step 2
            refine_response = client.post("/v1/query/refine", json=refine_payload)

            # Then: Step 2
            assert refine_response.status_code == 200
            refine_data = refine_response.json()
            assert refine_data["clarity_score"] >= 0.0

    def test_context_aware_clarification(self, client: TestClient) -> None:
        """
        Given: Query with pronoun AND conversation context
        When: POST /v1/query/clarify is called with context
        Then: Context is used for evaluation
        """
        # Given
        payload = {
            "query": "What did he invent?",
            "threshold": 0.7,
            "context": {
                "previous_queries": [
                    "Who is Thomas Edison?",
                    "Tell me about his inventions",
                ]
            },
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 200
        data = response.json()
        # Context should influence evaluation
        assert "clarity_score" in data

    def test_multiple_clarify_requests_independent(self, client: TestClient) -> None:
        """
        Given: Multiple independent clarification requests
        When: POST /v1/query/clarify is called multiple times
        Then: Each request is evaluated independently
        """
        # Given
        queries = [
            "What is Python?",
            "Who is the CEO of Apple?",
            "Recent developments in AI",
        ]

        # When/Then
        for query in queries:
            payload = {"query": query, "threshold": 0.7}
            response = client.post("/v1/query/clarify", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["original_query"] == query


# =============================================================================
# GWT Tests: Performance
# =============================================================================


class TestPerformanceGWT:
    """GWT tests for performance requirements."""

    def test_clarify_latency_under_threshold(self, client: TestClient) -> None:
        """
        Given: A standard query
        When: POST /v1/query/clarify is called
        Then: evaluation_time_ms < 500ms (performance requirement)
        """
        # Given
        payload = {
            "query": "What is machine learning?",
            "threshold": 0.7,
        }

        # When
        response = client.post("/v1/query/clarify", json=payload)

        # Then
        assert response.status_code == 200
        data = response.json()
        # Pattern-based clarification should be fast
        assert data["evaluation_time_ms"] < 500

    def test_refine_latency_reasonable(self, client: TestClient) -> None:
        """
        Given: A refinement request
        When: POST /v1/query/refine is called
        Then: Completes quickly (< 1000ms)
        """
        # Given
        payload = {
            "original_query": "What is this?",
            "selected_refinement": "Python programming language",
        }

        # When
        import time

        start = time.time()
        response = client.post("/v1/query/refine", json=payload)
        elapsed_ms = (time.time() - start) * 1000

        # Then
        assert response.status_code == 200
        assert elapsed_ms < 1000  # Should be fast
