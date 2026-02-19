"""
Comprehensive GWT unit tests for FewShotRegistry.

Few-Shot Registry
Verifies storage, retrieval, and atomic operations.
"""

import pytest
from ingestforge.learning.models import FewShotExample
from ingestforge.learning.registry import FewShotRegistry

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_registry(tmp_path):
    """Create a registry in a temporary directory."""
    path = tmp_path / "test_few_shot.jsonl"
    return FewShotRegistry(storage_path=path)


@pytest.fixture
def sample_example():
    """Return a valid few-shot example."""
    return FewShotExample(
        id="ex_1",
        input_text="Sample input",
        output_json={"key": "value"},
        domain="legal",
    )


# =============================================================================
# TESTS (GWT)
# =============================================================================


def test_add_and_list_examples(temp_registry, sample_example):
    """GIVEN an empty registry
    WHEN a new example is added
    THEN it should be retrievable via list_examples.
    """
    success = temp_registry.add_example(sample_example)
    assert success is True

    examples = temp_registry.list_examples()
    assert len(examples) == 1
    assert examples[0].id == "ex_1"
    assert examples[0].domain == "legal"


def test_domain_filtering(temp_registry):
    """GIVEN a registry with multiple domains
    WHEN listing examples with a domain filter
    THEN only matching examples are returned.
    """
    ex1 = FewShotExample(id="1", input_text="i1", output_json={}, domain="legal")
    ex2 = FewShotExample(id="2", input_text="i2", output_json={}, domain="cyber")

    temp_registry.add_example(ex1)
    temp_registry.add_example(ex2)

    legal_examples = temp_registry.list_examples(domain="legal")
    assert len(legal_examples) == 1
    assert legal_examples[0].domain == "legal"


def test_remove_example(temp_registry, sample_example):
    """GIVEN a registry with an example
    WHEN remove_example is called
    THEN the example is no longer in the registry.
    """
    temp_registry.add_example(sample_example)
    assert len(temp_registry.list_examples()) == 1

    removed = temp_registry.remove_example("ex_1")
    assert removed is True
    assert len(temp_registry.list_examples()) == 0


def test_limit_enforcement(temp_registry):
    """GIVEN a registry with many examples
    WHEN listing with a limit
    THEN no more than 'limit' examples are returned (JPL Rule #2).
    """
    for i in range(20):
        ex = FewShotExample(id=str(i), input_text="t", output_json={}, domain="gen")
        temp_registry.add_example(ex)

    results = temp_registry.list_examples(limit=5)
    assert len(results) == 5


def test_corrupt_line_handling(tmp_path):
    """GIVEN a registry file with corrupt JSON
    WHEN listing examples
    THEN corrupt lines are skipped gracefully.
    """
    path = tmp_path / "corrupt.jsonl"
    with open(path, "w") as f:
        f.write(
            '{"id": "1", "domain": "legal"}\n'
        )  # Valid but incomplete for FewShotExample
        f.write("not a json line\n")

    registry = FewShotRegistry(storage_path=path)
    results = registry.list_examples()
    # Pydantic validation should fail for the first line too as it's missing fields
    assert len(results) == 0
