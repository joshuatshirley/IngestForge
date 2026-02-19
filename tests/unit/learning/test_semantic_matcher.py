"""Comprehensive GWT unit tests for SemanticExampleMatcher.

Semantic Example Matcher.
Verifies embedding-based retrieval and JPL compliance.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from ingestforge.learning.models import FewShotExample
from ingestforge.learning.matcher import SemanticExampleMatcher

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_registry():
    """Create a mock registry with sample data."""
    registry = MagicMock()
    examples = [
        FewShotExample(
            id="1",
            input_text="Einstein was a physicist.",
            output_json={},
            domain="legal",
        ),
        FewShotExample(
            id="2",
            input_text="The judge ruled in favor.",
            output_json={},
            domain="legal",
        ),
        FewShotExample(
            id="3",
            input_text="Malware detected in system.",
            output_json={},
            domain="cyber",
        ),
    ]
    registry.list_examples.return_value = examples
    return registry


@pytest.fixture
def matcher(mock_registry):
    """Initialize matcher with a mock registry."""
    with patch("ingestforge.learning.matcher.EmbeddingGenerator") as MockEmbedder:
        # Create a mock instance for the embedder
        embedder_instance = MockEmbedder.return_value
        embedder_instance.generate = MagicMock()

        m = SemanticExampleMatcher(registry=mock_registry)
        # Ensure our matcher instance uses the mocked instance
        m.embedder = embedder_instance
        return m


# =============================================================================
# UNIT TESTS (GWT)
# =============================================================================


def test_semantic_matching_accuracy(matcher, mock_registry):
    """GIVEN a query about 'physics' and a legal domain
    WHEN find_matches is called
    THEN it returns the most relevant example based on cosine similarity.
    """
    # Setup mock embeddings
    target_vec = np.array([1.0, 0.0])
    candidate_vecs = np.array(
        [
            [0.9, 0.1],  # Similar to physics (Example 1)
            [0.1, 0.9],  # Not similar to physics (Example 2)
        ]
    )

    # generate() is called twice: once for input, once for candidates
    matcher.embedder.generate.side_effect = [[target_vec], candidate_vecs]

    # Domain filter simulation (legal)
    original_examples = mock_registry.list_examples.return_value
    mock_registry.list_examples.return_value = original_examples[:2]

    matches = matcher.find_matches("Tell me about physics", domain="legal", limit=1)

    assert len(matches) == 1
    assert matches[0].id == "1"


def test_min_similarity_threshold(matcher, mock_registry):
    """GIVEN an input text that is completely irrelevant to all examples
    WHEN find_matches is called
    THEN it returns an empty list if similarity is below MIN_SIMILARITY_SCORE.
    """
    target_vec = np.array([1.0, 0.0])
    candidate_vecs = np.array([[0.0, 1.0]])  # 0 similarity

    matcher.embedder.generate.side_effect = [[target_vec], candidate_vecs]

    original_examples = mock_registry.list_examples.return_value
    mock_registry.list_examples.return_value = [original_examples[0]]

    matches = matcher.find_matches("Irrelevant text")

    assert len(matches) == 0


def test_matcher_error_fallback(matcher, mock_registry):
    """GIVEN a failure in the embedding service
    WHEN find_matches is called
    THEN it falls back to returning the first N candidates from the registry.
    """
    matcher.embedder.generate.side_effect = Exception("Embedder down")

    matches = matcher.find_matches("Any query", limit=2)

    assert len(matches) == 2
    assert matches[0].id == "1"
    assert matches[1].id == "2"


def test_cosine_similarity_math(matcher):
    """GIVEN two vectors
    WHEN _cosine_similarity is called
    THEN it returns the correct mathematical similarity score.
    """
    target = np.array([1.0, 0.0])
    others = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    scores = matcher._cosine_similarity(target, others)

    assert np.allclose(scores[0], 1.0)
    assert np.allclose(scores[1], 0.0)
    assert np.allclose(scores[2], -1.0)
