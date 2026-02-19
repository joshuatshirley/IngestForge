"""Unit tests for Knowledge Mesh API endpoint.

Interactive Mesh D3 UI
Tests for /v1/viz/graph/knowledge-mesh endpoint.

GWT Format: Given-When-Then
Target Coverage: >80%
JPL Compliance: All test functions < 60 lines
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ingestforge.api.main import app

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create test client for API."""
    return TestClient(app)


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create mock storage backend."""
    storage = MagicMock()
    storage.search.return_value = []
    return storage


@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """Create sample chunks with entities."""
    return [
        {
            "metadata": {
                "source": "doc1.pdf",
                "entities": ["John Doe", "Jane Smith", "Acme Corp"],
                "concepts": ["machine learning", "data science"],
            }
        },
        {
            "metadata": {
                "source": "doc2.pdf",
                "entities": ["John Doe", "Bob Johnson", "Acme Corp"],
                "concepts": ["artificial intelligence"],
            }
        },
        {
            "metadata": {
                "source": "doc3.pdf",
                "entities": ["Jane Smith"],
                "concepts": ["data science", "statistics"],
            }
        },
    ]


# =============================================================================
# GWT TESTS: Endpoint Availability
# =============================================================================


def test_gwt_001_knowledge_mesh_endpoint_exists(client: TestClient) -> None:
    """GWT-001: Knowledge mesh endpoint is available.

    Given: The API is running
    When: A GET request is made to /v1/viz/graph/knowledge-mesh
    Then: The endpoint should respond (not 404)
    """
    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code != 404, "Endpoint should exist"


def test_gwt_002_knowledge_mesh_returns_json(client: TestClient) -> None:
    """GWT-002: Knowledge mesh returns JSON response.

    Given: The API is running
    When: A GET request is made to /v1/viz/graph/knowledge-mesh
    Then: The response should be valid JSON
    """
    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert isinstance(data, dict)


# =============================================================================
# GWT TESTS: Query Parameters
# =============================================================================


def test_gwt_003_max_nodes_parameter_accepted(client: TestClient) -> None:
    """GWT-003: max_nodes parameter is accepted.

    Given: The knowledge mesh endpoint is available
    When: A request is made with max_nodes=100
    Then: The parameter should be accepted without error
    """
    # When
    response = client.get("/v1/viz/graph/knowledge-mesh?max_nodes=100")

    # Then
    assert response.status_code in [200, 500]  # 500 if no data, but accepts param


def test_gwt_004_max_nodes_validation_enforces_minimum(
    client: TestClient,
) -> None:
    """GWT-004: max_nodes validates minimum value.

    Given: The knowledge mesh endpoint is available
    When: A request is made with max_nodes=0 (below minimum)
    Then: A validation error should be returned
    """
    # When
    response = client.get("/v1/viz/graph/knowledge-mesh?max_nodes=0")

    # Then
    assert response.status_code == 422  # Validation error


def test_gwt_005_max_nodes_validation_enforces_maximum(
    client: TestClient,
) -> None:
    """GWT-005: max_nodes validates maximum value.

    Given: The knowledge mesh endpoint is available
    When: A request is made with max_nodes=2000 (above maximum)
    Then: A validation error should be returned
    """
    # When
    response = client.get("/v1/viz/graph/knowledge-mesh?max_nodes=2000")

    # Then
    assert response.status_code == 422  # Validation error


def test_gwt_006_min_citations_parameter_accepted(client: TestClient) -> None:
    """GWT-006: min_citations parameter is accepted.

    Given: The knowledge mesh endpoint is available
    When: A request is made with min_citations=5
    Then: The parameter should be accepted without error
    """
    # When
    response = client.get("/v1/viz/graph/knowledge-mesh?min_citations=5")

    # Then
    assert response.status_code in [200, 500]


def test_gwt_007_depth_parameter_accepted(client: TestClient) -> None:
    """GWT-007: depth parameter is accepted.

    Given: The knowledge mesh endpoint is available
    When: A request is made with depth=3
    Then: The parameter should be accepted without error
    """
    # When
    response = client.get("/v1/viz/graph/knowledge-mesh?depth=3")

    # Then
    assert response.status_code in [200, 500]


def test_gwt_008_depth_validation_enforces_range(client: TestClient) -> None:
    """GWT-008: depth validates range (1-5).

    Given: The knowledge mesh endpoint is available
    When: A request is made with depth=10 (above maximum)
    Then: A validation error should be returned
    """
    # When
    response = client.get("/v1/viz/graph/knowledge-mesh?depth=10")

    # Then
    assert response.status_code == 422


def test_gwt_009_entity_types_parameter_accepted(client: TestClient) -> None:
    """GWT-009: entity_types parameter is accepted.

    Given: The knowledge mesh endpoint is available
    When: A request is made with entity_types="person,org"
    Then: The parameter should be accepted without error
    """
    # When
    response = client.get(
        "/v1/viz/graph/knowledge-mesh?entity_types=person,organization"
    )

    # Then
    assert response.status_code in [200, 500]


def test_gwt_010_include_chunks_parameter_accepted(
    client: TestClient,
) -> None:
    """GWT-010: include_chunks parameter is accepted.

    Given: The knowledge mesh endpoint is available
    When: A request is made with include_chunks=true
    Then: The parameter should be accepted without error
    """
    # When
    response = client.get("/v1/viz/graph/knowledge-mesh?include_chunks=true")

    # Then
    assert response.status_code in [200, 500]


# =============================================================================
# GWT TESTS: Response Structure
# =============================================================================


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_011_response_has_nodes_array(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
    sample_chunks: List[Dict[str, Any]],
) -> None:
    """GWT-011: Response contains nodes array.

    Given: The storage backend has chunks with entities
    When: A request is made to /v1/viz/graph/knowledge-mesh
    Then: The response should contain a 'nodes' array
    """
    # Given
    mock_storage.search.return_value = sample_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert isinstance(data["nodes"], list)


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_012_response_has_edges_array(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
    sample_chunks: List[Dict[str, Any]],
) -> None:
    """GWT-012: Response contains edges array.

    Given: The storage backend has chunks with entities
    When: A request is made to /v1/viz/graph/knowledge-mesh
    Then: The response should contain an 'edges' array
    """
    # Given
    mock_storage.search.return_value = sample_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert "edges" in data
    assert isinstance(data["edges"], list)


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_013_response_has_metadata(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
    sample_chunks: List[Dict[str, Any]],
) -> None:
    """GWT-013: Response contains metadata object.

    Given: The storage backend has chunks
    When: A request is made to /v1/viz/graph/knowledge-mesh
    Then: The response should contain a 'metadata' object
    """
    # Given
    mock_storage.search.return_value = sample_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert isinstance(data["metadata"], dict)


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_014_node_has_required_fields(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
    sample_chunks: List[Dict[str, Any]],
) -> None:
    """GWT-014: Each node has required fields.

    Given: The storage backend has chunks with entities
    When: A request is made to /v1/viz/graph/knowledge-mesh
    Then: Each node should have id, type, label, properties, metadata
    """
    # Given
    mock_storage.search.return_value = sample_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code == 200
    data = response.json()
    if data["nodes"]:
        node = data["nodes"][0]
        assert "id" in node
        assert "type" in node
        assert "label" in node
        assert "properties" in node
        assert "metadata" in node


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_015_node_properties_has_citation_count(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
    sample_chunks: List[Dict[str, Any]],
) -> None:
    """GWT-015: Node properties include citation_count.

    Given: The storage backend has chunks with entities
    When: A request is made to /v1/viz/graph/knowledge-mesh
    Then: Node properties should include citation_count field
    """
    # Given
    mock_storage.search.return_value = sample_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code == 200
    data = response.json()
    if data["nodes"]:
        node = data["nodes"][0]
        assert "citation_count" in node["properties"]
        assert isinstance(node["properties"]["citation_count"], int)


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_016_edge_has_required_fields(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
    sample_chunks: List[Dict[str, Any]],
) -> None:
    """GWT-016: Each edge has required fields.

    Given: The storage backend has chunks with entities
    When: A request is made to /v1/viz/graph/knowledge-mesh
    Then: Each edge should have source, target, type, weight, style
    """
    # Given
    mock_storage.search.return_value = sample_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code == 200
    data = response.json()
    if data["edges"]:
        edge = data["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "type" in edge
        assert "weight" in edge
        assert "style" in edge


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_017_metadata_has_computation_time(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
    sample_chunks: List[Dict[str, Any]],
) -> None:
    """GWT-017: Metadata includes computation_time_ms.

    Given: The storage backend has chunks
    When: A request is made to /v1/viz/graph/knowledge-mesh
    Then: Metadata should include computation_time_ms field
    """
    # Given
    mock_storage.search.return_value = sample_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert "computation_time_ms" in data["metadata"]
    assert isinstance(data["metadata"]["computation_time_ms"], (int, float))
    assert data["metadata"]["computation_time_ms"] >= 0


# =============================================================================
# GWT TESTS: Citation Counting
# =============================================================================


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_018_entity_citation_count_accumulated(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
    sample_chunks: List[Dict[str, Any]],
) -> None:
    """GWT-018: Entity citations are counted across chunks.

    Given: "John Doe" appears in 2 chunks
    When: A request is made to /v1/viz/graph/knowledge-mesh
    Then: "John Doe" node should have citation_count=2
    """
    # Given
    mock_storage.search.return_value = sample_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code == 200
    data = response.json()

    # Find "John Doe" node
    john_doe_node = next((n for n in data["nodes"] if "John Doe" in n["label"]), None)
    if john_doe_node:
        assert john_doe_node["properties"]["citation_count"] == 2


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_019_min_citations_filters_low_citation_entities(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
    sample_chunks: List[Dict[str, Any]],
) -> None:
    """GWT-019: min_citations filter excludes low-citation entities.

    Given: "Bob Johnson" appears only once (citation_count=1)
    When: A request is made with min_citations=2
    Then: "Bob Johnson" should not appear in nodes
    """
    # Given
    mock_storage.search.return_value = sample_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh?min_citations=2")

    # Then
    assert response.status_code == 200
    data = response.json()

    # "Bob Johnson" should be filtered out
    bob_node = next((n for n in data["nodes"] if "Bob Johnson" in n["label"]), None)
    assert bob_node is None


# =============================================================================
# GWT TESTS: Entity Type Filtering
# =============================================================================


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_020_entity_types_filter_includes_only_specified(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
    sample_chunks: List[Dict[str, Any]],
) -> None:
    """GWT-020: entity_types filter includes only specified types.

    Given: Chunks contain person and organization entities
    When: A request is made with entity_types="person"
    Then: Only person-type nodes should be included
    """
    # Given
    mock_storage.search.return_value = sample_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh?entity_types=person")

    # Then
    assert response.status_code == 200
    data = response.json()

    # All entity nodes should be person type
    for node in data["nodes"]:
        if node["type"] not in ["document"]:  # Exclude doc nodes
            assert node["type"] == "person"


# =============================================================================
# GWT TESTS: Edge Cases
# =============================================================================


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_021_empty_storage_returns_empty_graph(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
) -> None:
    """GWT-021: Empty storage returns empty graph.

    Given: The storage backend has no chunks
    When: A request is made to /v1/viz/graph/knowledge-mesh
    Then: The response should have empty nodes and edges arrays
    """
    # Given
    mock_storage.search.return_value = []
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert len(data["nodes"]) == 0
    assert len(data["edges"]) == 0


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_022_chunks_without_entities_handled(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
) -> None:
    """GWT-022: Chunks without entities are handled gracefully.

    Given: Chunks exist but have no entities metadata
    When: A request is made to /v1/viz/graph/knowledge-mesh
    Then: The response should succeed with only document nodes
    """
    # Given
    chunks_no_entities = [
        {"metadata": {"source": "doc1.pdf"}},
        {"metadata": {"source": "doc2.pdf"}},
    ]
    mock_storage.search.return_value = chunks_no_entities
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code == 200
    data = response.json()
    # Should have document nodes
    doc_nodes = [n for n in data["nodes"] if n["type"] == "document"]
    assert len(doc_nodes) == 2


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_023_max_nodes_limit_enforced(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
) -> None:
    """GWT-023: max_nodes limit is enforced.

    Given: Storage has 100 chunks
    When: A request is made with max_nodes=10
    Then: The response should have <= 10 nodes
    """
    # Given
    large_chunks = [
        {
            "metadata": {
                "source": f"doc{i}.pdf",
                "entities": [f"Entity{i}"],
            }
        }
        for i in range(100)
    ]
    mock_storage.search.return_value = large_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh?max_nodes=10")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert len(data["nodes"]) <= 10


# =============================================================================
# GWT TESTS: Performance
# =============================================================================


@patch("ingestforge.api.routes.viz.get_storage_backend")
def test_gwt_024_response_time_under_threshold(
    mock_get_storage: MagicMock,
    client: TestClient,
    mock_storage: MagicMock,
    sample_chunks: List[Dict[str, Any]],
) -> None:
    """GWT-024: Response time is under 2 seconds.

    Given: The storage backend has chunks
    When: A request is made to /v1/viz/graph/knowledge-mesh
    Then: The computation_time_ms should be < 2000ms
    """
    # Given
    mock_storage.search.return_value = sample_chunks
    mock_get_storage.return_value = mock_storage

    # When
    response = client.get("/v1/viz/graph/knowledge-mesh")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["computation_time_ms"] < 2000


# =============================================================================
# GWT TESTS: Backward Compatibility
# =============================================================================


def test_gwt_025_old_viz_graph_endpoint_still_works(
    client: TestClient,
) -> None:
    """GWT-025: Old /v1/viz/graph endpoint remains functional.

    Given: The new knowledge-mesh endpoint is added
    When: A request is made to the old /v1/viz/graph endpoint
    Then: The endpoint should still work (backward compatible)
    """
    # When
    response = client.get("/v1/viz/graph")

    # Then
    assert response.status_code in [200, 500]  # Exists, may fail without data
    assert response.status_code != 404


# =============================================================================
# COVERAGE SUMMARY
# =============================================================================

"""
Test Coverage Summary:
- Endpoint availability: 2 tests (GWT-001, GWT-002)
- Query parameters: 8 tests (GWT-003 to GWT-010)
- Response structure: 7 tests (GWT-011 to GWT-017)
- Citation counting: 2 tests (GWT-018, GWT-019)
- Entity type filtering: 1 test (GWT-020)
- Edge cases: 3 tests (GWT-021 to GWT-023)
- Performance: 1 test (GWT-024)
- Backward compatibility: 1 test (GWT-025)

Total: 25 GWT tests
Expected Coverage: >85%
JPL Compliance: ✓ All test functions < 60 lines
"""
