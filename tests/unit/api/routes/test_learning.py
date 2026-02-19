"""
Unit tests for Learning API routes.

Learning Governance UI API.
Tests GWT scenarios and NASA JPL Power of Ten compliance.
"""

import pytest
from unittest.mock import Mock, patch

from ingestforge.api.routes.learning import (
    list_examples,
    approve_example,
    reject_example,
    edit_example,
    get_learning_stats,
    ApproveRequest,
    RejectRequest,
    EditRequest,
    LearningExamplesListResponse,
    ApproveResponse,
    RejectResponse,
    EditResponse,
    MAX_EXAMPLES_PER_PAGE,
    MAX_ENTITIES_IN_RESPONSE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_example() -> Mock:
    """Create a mock GoldenExample."""
    example = Mock()
    example.example_id = "test_abc123_def456"
    example.vertical_id = "legal"
    example.entity_type = "PERSON"
    example.chunk_content = "Dr. John Smith filed the case."
    example.entities = [{"text": "John Smith", "type": "PERSON"}]
    example.metadata = {"source_document": "doc-001"}
    example.approved_at = None
    example.approved_by = "user"
    return example


@pytest.fixture
def mock_registry(mock_example: Mock) -> Mock:
    """Create a mock IFExampleRegistry."""
    registry = Mock()
    registry.list_examples.return_value = [mock_example]
    registry.get_example.return_value = mock_example
    registry.count_examples.return_value = 1
    registry.get_verticals.return_value = ["legal", "medical"]
    return registry


# ---------------------------------------------------------------------------
# GWT Scenario 1: List Examples
# ---------------------------------------------------------------------------


class TestListExamples:
    """Given list_examples endpoint, When called, Then returns paginated examples."""

    def test_given_examples_when_list_called_then_returns_list(
        self, mock_registry: Mock, mock_example: Mock
    ):
        """Given examples exist, When list called, Then returns list response."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                response = list_examples()

                assert isinstance(response, LearningExamplesListResponse)
                assert len(response.examples) == 1
                assert response.total == 1
                assert response.page == 1

    def test_given_filter_when_list_called_then_filters_applied(
        self, mock_registry: Mock
    ):
        """Given filters, When list called, Then filters applied."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                list_examples(vertical_id="legal", entity_type="PERSON")

                mock_registry.list_examples.assert_called_once()
                call_args = mock_registry.list_examples.call_args
                assert call_args.kwargs.get("vertical_id") == "legal"
                assert call_args.kwargs.get("entity_type") == "PERSON"

    def test_given_pagination_when_list_called_then_paginated(
        self, mock_registry: Mock
    ):
        """Given pagination params, When list called, Then paginated correctly."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                response = list_examples(page=2, page_size=5)

                assert response.page == 2
                assert response.page_size == 5


# ---------------------------------------------------------------------------
# GWT Scenario 2: Approve Example
# ---------------------------------------------------------------------------


class TestApproveExample:
    """Given approve endpoint, When called, Then example is approved."""

    def test_given_valid_example_when_approve_then_approved(
        self, mock_registry: Mock, mock_example: Mock
    ):
        """Given valid example, When approve called, Then status is approved."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                request = ApproveRequest(example_id="test_abc123_def456")
                response = approve_example(request)

                assert isinstance(response, ApproveResponse)
                assert response.status == "approved"
                assert response.available_for_learning is True

    def test_given_nonexistent_example_when_approve_then_error(
        self, mock_registry: Mock
    ):
        """Given nonexistent example, When approve called, Then error returned."""
        mock_registry.get_example.return_value = None

        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                request = ApproveRequest(example_id="nonexistent")
                response = approve_example(request)

                assert response.status == "error"
                assert "not found" in response.message


# ---------------------------------------------------------------------------
# GWT Scenario 3: Reject Example
# ---------------------------------------------------------------------------


class TestRejectExample:
    """Given reject endpoint, When called, Then example is rejected."""

    def test_given_valid_example_when_reject_then_rejected(
        self, mock_registry: Mock, mock_example: Mock
    ):
        """Given valid example, When reject called, Then status is rejected."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                request = RejectRequest(
                    example_id="test_abc123_def456", reason="Low quality extraction"
                )
                response = reject_example(request)

                assert isinstance(response, RejectResponse)
                assert response.status == "rejected"

    def test_given_rejection_reason_when_reject_then_reason_stored(
        self, mock_registry: Mock, mock_example: Mock
    ):
        """Given rejection reason, When reject called, Then reason stored."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                request = RejectRequest(
                    example_id="test_abc123_def456", reason="Incorrect entity type"
                )
                reject_example(request)

                assert (
                    mock_example.metadata["rejected_reason"] == "Incorrect entity type"
                )


# ---------------------------------------------------------------------------
# GWT Scenario 4: Edit Example
# ---------------------------------------------------------------------------


class TestEditExample:
    """Given edit endpoint, When called, Then example is updated."""

    def test_given_new_entities_when_edit_then_entities_updated(
        self, mock_registry: Mock, mock_example: Mock
    ):
        """Given new entities, When edit called, Then entities updated."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                new_entities = [{"text": "Dr. Smith", "type": "PERSON"}]
                request = EditRequest(
                    example_id="test_abc123_def456", entities=new_entities
                )
                response = edit_example(request)

                assert isinstance(response, EditResponse)
                assert response.status == "updated"
                assert "entities" in response.message

    def test_given_new_entity_type_when_edit_then_type_updated(
        self, mock_registry: Mock, mock_example: Mock
    ):
        """Given new entity type, When edit called, Then type updated."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                request = EditRequest(
                    example_id="test_abc123_def456", entity_type="ORG"
                )
                edit_example(request)

                assert mock_example.entity_type == "ORG"


# ---------------------------------------------------------------------------
# GWT Scenario 5: Learning Stats
# ---------------------------------------------------------------------------


class TestLearningStats:
    """Given stats endpoint, When called, Then returns statistics."""

    def test_given_examples_when_stats_called_then_returns_counts(
        self, mock_registry: Mock
    ):
        """Given examples exist, When stats called, Then counts returned."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                stats = get_learning_stats()

                assert stats["total_examples"] == 1
                assert stats["verticals"] == 2
                assert "legal" in stats["verticals_list"]


# ---------------------------------------------------------------------------
# JPL Rule #2: Fixed Upper Bounds
# ---------------------------------------------------------------------------


class TestJPLRule2Bounds:
    """Test fixed upper bounds per JPL Rule #2."""

    def test_max_examples_per_page_bounded(self):
        """Given large page_size, When list called, Then bounded."""
        assert MAX_EXAMPLES_PER_PAGE == 100

    def test_max_entities_in_response_bounded(self):
        """Given many entities, When response built, Then bounded."""
        assert MAX_ENTITIES_IN_RESPONSE == 500

    def test_page_size_capped_at_max(self, mock_registry: Mock):
        """Given page_size > max, When list called, Then capped."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                response = list_examples(page_size=500)

                # Should be capped at MAX_EXAMPLES_PER_PAGE
                assert response.page_size <= MAX_EXAMPLES_PER_PAGE


# ---------------------------------------------------------------------------
# JPL Rule #9: Complete Type Hints
# ---------------------------------------------------------------------------


class TestJPLRule9TypeHints:
    """Test complete type hints per JPL Rule #9."""

    def test_list_examples_returns_typed_response(self, mock_registry: Mock):
        """Given list_examples, Then returns typed LearningExamplesListResponse."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                response = list_examples()

                assert isinstance(response, LearningExamplesListResponse)
                assert isinstance(response.examples, list)
                assert isinstance(response.total, int)

    def test_approve_returns_typed_response(self, mock_registry: Mock):
        """Given approve_example, Then returns typed ApproveResponse."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                request = ApproveRequest(example_id="test")
                response = approve_example(request)

                assert isinstance(response, ApproveResponse)
                assert isinstance(response.available_for_learning, bool)


# ---------------------------------------------------------------------------
# Acceptance Criteria Tests
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """Test explicit acceptance criteria from ."""

    def test_ac1_grid_view_shows_original_text(
        self, mock_registry: Mock, mock_example: Mock
    ):
        """AC: Grid view shows Original Text."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                response = list_examples()

                assert len(response.examples) > 0
                example = response.examples[0]
                assert example.chunk_content is not None
                assert len(example.chunk_content) > 0

    def test_ac2_grid_view_shows_extracted_json(
        self, mock_registry: Mock, mock_example: Mock
    ):
        """AC: Grid view shows Extracted JSON."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                response = list_examples()

                assert len(response.examples) > 0
                example = response.examples[0]
                assert example.entities is not None
                assert isinstance(example.entities, list)

    def test_ac3_grid_view_shows_source_document(
        self, mock_registry: Mock, mock_example: Mock
    ):
        """AC: Grid view shows Source Document."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                response = list_examples()

                assert len(response.examples) > 0
                example = response.examples[0]
                # source_document is optional but should be present
                assert hasattr(example, "source_document")

    def test_ac4_approve_makes_available_for_learning(
        self, mock_registry: Mock, mock_example: Mock
    ):
        """AC: Toggling 'Approve' makes example available for SemanticMatcher."""
        with patch(
            "ingestforge.api.routes.learning._LazyDeps.get_registry",
            return_value=mock_registry,
        ):
            with patch(
                "ingestforge.api.routes.learning._LazyDeps.get_logger",
                return_value=Mock(),
            ):
                request = ApproveRequest(example_id="test_abc123_def456")
                response = approve_example(request)

                assert response.available_for_learning is True
                assert response.status == "approved"
