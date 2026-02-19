"""
Tests for Single-Object Aggregator ().

GWT (Given-When-Then) test structure.
NASA JPL Power of Ten compliance verification.
"""

import inspect
import json
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field

from ingestforge.core.pipeline.aggregator import (
    IFAggregator,
    AggregationStrategy,
    AggregationResult,
    aggregate_artifacts,
    aggregate_to_json,
    MAX_AGGREGATION_CHUNKS,
    MAX_FIELDS,
    MAX_LIST_ITEMS,
)
from ingestforge.core.pipeline.artifacts import IFTextArtifact, IFChunkArtifact


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_schema() -> type:
    """Create a sample Pydantic schema for testing."""

    class ArticleInsight(BaseModel):
        title: str
        author: Optional[str] = None
        score: int = 0
        tags: List[str] = Field(default_factory=list)
        published: Optional[str] = None

    return ArticleInsight


@pytest.fixture
def sample_schema_def() -> Dict[str, Any]:
    """Create a sample schema definition dict."""
    return {
        "title": "string",
        "author": "string?",
        "score": "int",
        "tags": "list[string]",
    }


@pytest.fixture
def text_artifact_with_json() -> IFTextArtifact:
    """Create text artifact with JSON content."""
    content = json.dumps(
        {
            "title": "Test Article",
            "author": "John Doe",
            "score": 95,
        }
    )
    return IFTextArtifact(
        artifact_id="text-001",
        content=content,
    )


@pytest.fixture
def chunk_artifact_1() -> IFChunkArtifact:
    """Create first chunk artifact."""
    return IFChunkArtifact(
        artifact_id="chunk-001",
        document_id="doc-001",
        content="title: First Article\nauthor: Alice",
        chunk_index=0,
        total_chunks=3,
        metadata={"tags": ["ai", "ml"]},
    )


@pytest.fixture
def chunk_artifact_2() -> IFChunkArtifact:
    """Create second chunk artifact."""
    return IFChunkArtifact(
        artifact_id="chunk-002",
        document_id="doc-001",
        content="score: 85",
        chunk_index=1,
        total_chunks=3,
        metadata={"tags": ["python"]},
    )


@pytest.fixture
def chunk_artifact_3() -> IFChunkArtifact:
    """Create third chunk artifact."""
    return IFChunkArtifact(
        artifact_id="chunk-003",
        document_id="doc-001",
        content="published: 2026-02-16",
        chunk_index=2,
        total_chunks=3,
        metadata={},
    )


@pytest.fixture
def aggregator() -> IFAggregator:
    """Create default aggregator instance."""
    return IFAggregator()


# ---------------------------------------------------------------------------
# AggregationResult Tests
# ---------------------------------------------------------------------------


class TestAggregationResult:
    """Tests for AggregationResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating an AggregationResult."""
        result = AggregationResult(
            data={"title": "Test"},
            schema_valid=True,
        )
        assert result.data == {"title": "Test"}
        assert result.schema_valid is True
        assert result.missing_fields == []
        assert result.validation_errors == []

    def test_result_with_missing_fields(self) -> None:
        """Test AggregationResult with missing fields."""
        result = AggregationResult(
            data={"title": "Test"},
            schema_valid=False,
            missing_fields=["author", "score"],
        )
        assert result.is_complete is False
        assert len(result.missing_fields) == 2

    def test_result_is_complete_property(self) -> None:
        """Test is_complete property."""
        result = AggregationResult(
            data={"title": "Test", "author": "Alice"},
            schema_valid=True,
            missing_fields=[],
        )
        assert result.is_complete is True

    def test_result_artifact_count(self) -> None:
        """Test artifact_count property."""
        result = AggregationResult(
            data={},
            schema_valid=True,
            source_artifact_ids=["a1", "a2", "a3"],
        )
        assert result.artifact_count == 3

    def test_result_field_sources_tracking(self) -> None:
        """Test field_sources tracks which artifacts contributed."""
        result = AggregationResult(
            data={"title": "Test"},
            schema_valid=True,
            field_sources={"title": ["chunk-001", "chunk-002"]},
        )
        assert "title" in result.field_sources
        assert len(result.field_sources["title"]) == 2


# ---------------------------------------------------------------------------
# AggregationStrategy Tests
# ---------------------------------------------------------------------------


class TestAggregationStrategy:
    """Tests for AggregationStrategy enum."""

    def test_first_wins_strategy(self) -> None:
        """Test FIRST_WINS strategy value."""
        assert AggregationStrategy.FIRST_WINS.value == "first_wins"

    def test_last_wins_strategy(self) -> None:
        """Test LAST_WINS strategy value."""
        assert AggregationStrategy.LAST_WINS.value == "last_wins"

    def test_accumulate_strategy(self) -> None:
        """Test ACCUMULATE strategy value."""
        assert AggregationStrategy.ACCUMULATE.value == "accumulate"

    def test_merge_unique_strategy(self) -> None:
        """Test MERGE_UNIQUE strategy value."""
        assert AggregationStrategy.MERGE_UNIQUE.value == "merge_unique"


# ---------------------------------------------------------------------------
# IFAggregator Initialization Tests
# ---------------------------------------------------------------------------


class TestIFAggregatorInit:
    """Tests for IFAggregator initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        agg = IFAggregator()
        assert agg._default_strategy == AggregationStrategy.FIRST_WINS
        assert agg._deduplicate_lists is True

    def test_init_with_strategy(self) -> None:
        """Test initialization with custom strategy."""
        agg = IFAggregator(default_strategy=AggregationStrategy.LAST_WINS)
        assert agg._default_strategy == AggregationStrategy.LAST_WINS

    def test_init_with_field_strategies(self) -> None:
        """Test initialization with per-field strategies."""
        strategies = {"tags": AggregationStrategy.ACCUMULATE}
        agg = IFAggregator(field_strategies=strategies)
        assert agg._field_strategies["tags"] == AggregationStrategy.ACCUMULATE

    def test_processor_properties(self) -> None:
        """Test IFProcessor interface properties."""
        agg = IFAggregator()
        assert agg.processor_id == "aggregator"
        assert agg.version == "1.0.0"
        assert "aggregate" in agg.capabilities
        assert agg.memory_mb > 0
        assert agg.is_available() is True


# ---------------------------------------------------------------------------
# GWT-1: Multi-Chunk to Single Object Tests
# ---------------------------------------------------------------------------


class TestGWT1MultiChunkToSingleObject:
    """GWT-1: Multi-chunk to single object tests."""

    def test_given_chunks_when_aggregate_then_single_object(
        self,
        aggregator: IFAggregator,
        chunk_artifact_1: IFChunkArtifact,
        chunk_artifact_2: IFChunkArtifact,
        sample_schema_def: Dict[str, Any],
    ) -> None:
        """Given multiple chunks, when aggregate, then produces single object."""
        result = aggregator.aggregate(
            [chunk_artifact_1, chunk_artifact_2],
            schema_def=sample_schema_def,
        )

        assert isinstance(result, AggregationResult)
        assert isinstance(result.data, dict)
        assert result.artifact_count == 2

    def test_given_chunks_when_aggregate_to_artifact_then_text_artifact(
        self,
        aggregator: IFAggregator,
        chunk_artifact_1: IFChunkArtifact,
        chunk_artifact_2: IFChunkArtifact,
        sample_schema_def: Dict[str, Any],
    ) -> None:
        """Given chunks, when aggregate_to_artifact, then returns IFTextArtifact."""
        artifact = aggregator.aggregate_to_artifact(
            [chunk_artifact_1, chunk_artifact_2],
            schema_def=sample_schema_def,
        )

        assert isinstance(artifact, IFTextArtifact)
        assert artifact.artifact_id.startswith("aggregated:")
        assert artifact.metadata.get("aggregation_result") is True

    def test_aggregated_artifact_has_lineage(
        self,
        aggregator: IFAggregator,
        chunk_artifact_1: IFChunkArtifact,
    ) -> None:
        """Test aggregated artifact has proper lineage."""
        artifact = aggregator.aggregate_to_artifact([chunk_artifact_1])

        assert artifact.parent_id == chunk_artifact_1.artifact_id
        assert "aggregator" in artifact.provenance


# ---------------------------------------------------------------------------
# GWT-2: Field Extraction Across Chunks Tests
# ---------------------------------------------------------------------------


class TestGWT2FieldExtractionAcrossChunks:
    """GWT-2: Field extraction across chunks tests."""

    def test_given_fields_in_different_chunks_when_aggregate_then_combined(
        self,
        aggregator: IFAggregator,
        chunk_artifact_1: IFChunkArtifact,
        chunk_artifact_2: IFChunkArtifact,
        chunk_artifact_3: IFChunkArtifact,
        sample_schema_def: Dict[str, Any],
    ) -> None:
        """Given fields in different chunks, when aggregate, then combined."""
        result = aggregator.aggregate(
            [chunk_artifact_1, chunk_artifact_2, chunk_artifact_3],
            schema_def=sample_schema_def,
        )

        # title from chunk 1, score from chunk 2
        assert "title" in result.data
        assert "score" in result.data

    def test_field_sources_tracked(
        self,
        aggregator: IFAggregator,
        chunk_artifact_1: IFChunkArtifact,
        chunk_artifact_2: IFChunkArtifact,
        sample_schema_def: Dict[str, Any],
    ) -> None:
        """Test that field sources are tracked correctly."""
        result = aggregator.aggregate(
            [chunk_artifact_1, chunk_artifact_2],
            schema_def=sample_schema_def,
        )

        # Check field sources are recorded
        assert len(result.source_artifact_ids) == 2

    def test_extract_from_json_content(
        self,
        aggregator: IFAggregator,
        text_artifact_with_json: IFTextArtifact,
        sample_schema_def: Dict[str, Any],
    ) -> None:
        """Test field extraction from JSON content."""
        result = aggregator.aggregate(
            [text_artifact_with_json],
            schema_def=sample_schema_def,
        )

        assert result.data.get("title") == "Test Article"
        assert result.data.get("author") == "John Doe"
        assert result.data.get("score") == 95

    def test_extract_from_metadata(
        self,
        aggregator: IFAggregator,
        chunk_artifact_1: IFChunkArtifact,
        sample_schema_def: Dict[str, Any],
    ) -> None:
        """Test field extraction from artifact metadata."""
        result = aggregator.aggregate(
            [chunk_artifact_1],
            schema_def=sample_schema_def,
        )

        # tags should come from metadata
        assert "tags" in result.data


# ---------------------------------------------------------------------------
# GWT-3: List Field Accumulation Tests
# ---------------------------------------------------------------------------


class TestGWT3ListFieldAccumulation:
    """GWT-3: List field accumulation tests."""

    def test_given_list_field_when_accumulate_then_combined(self) -> None:
        """Given list field, when accumulate strategy, then combined."""
        aggregator = IFAggregator(
            field_strategies={"tags": AggregationStrategy.ACCUMULATE}
        )

        chunk1 = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content="",
            metadata={"tags": ["ai", "ml"]},
        )
        chunk2 = IFChunkArtifact(
            artifact_id="c2",
            document_id="d1",
            content="",
            metadata={"tags": ["python", "data"]},
        )

        result = aggregator.aggregate(
            [chunk1, chunk2],
            schema_def={"tags": "list[string]"},
        )

        tags = result.data.get("tags", [])
        assert len(tags) == 4
        assert "ai" in tags
        assert "python" in tags

    def test_merge_unique_deduplicates(self) -> None:
        """Test MERGE_UNIQUE strategy deduplicates values."""
        aggregator = IFAggregator(
            field_strategies={"tags": AggregationStrategy.MERGE_UNIQUE}
        )

        chunk1 = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content="",
            metadata={"tags": ["ai", "ml"]},
        )
        chunk2 = IFChunkArtifact(
            artifact_id="c2",
            document_id="d1",
            content="",
            metadata={"tags": ["ai", "python"]},  # "ai" is duplicate
        )

        result = aggregator.aggregate(
            [chunk1, chunk2],
            schema_def={"tags": "list[string]"},
        )

        tags = result.data.get("tags", [])
        assert tags.count("ai") == 1  # Only one "ai"

    def test_list_bounded_by_max_items(self) -> None:
        """Test list accumulation is bounded by MAX_LIST_ITEMS."""
        aggregator = IFAggregator(
            field_strategies={"items": AggregationStrategy.ACCUMULATE}
        )

        # Create artifact with many items
        large_list = list(range(MAX_LIST_ITEMS + 100))
        chunk = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content="",
            metadata={"items": large_list},
        )

        result = aggregator.aggregate(
            [chunk],
            schema_def={"items": "list[int]"},
        )

        items = result.data.get("items", [])
        assert len(items) <= MAX_LIST_ITEMS


# ---------------------------------------------------------------------------
# GWT-4: Validation Against Schema Tests
# ---------------------------------------------------------------------------


class TestGWT4ValidationAgainstSchema:
    """GWT-4: Validation against schema tests."""

    def test_given_valid_data_when_validate_then_schema_valid_true(
        self,
        aggregator: IFAggregator,
        sample_schema: type,
    ) -> None:
        """Given valid data, when validate, then schema_valid is True."""
        chunk = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content=json.dumps({"title": "Test", "score": 10}),
        )

        result = aggregator.aggregate([chunk], schema=sample_schema)

        assert result.schema_valid is True

    def test_given_missing_required_field_when_validate_then_reports_missing(
        self,
        aggregator: IFAggregator,
        sample_schema: type,
    ) -> None:
        """Given missing required field, when validate, then reports missing."""
        chunk = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content=json.dumps({"score": 10}),  # Missing required "title"
        )

        result = aggregator.aggregate([chunk], schema=sample_schema)

        assert "title" in result.missing_fields
        assert result.schema_valid is False

    def test_given_invalid_type_when_validate_then_reports_error(
        self,
        aggregator: IFAggregator,
        sample_schema: type,
    ) -> None:
        """Given invalid type, when validate, then reports validation error."""
        chunk = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content=json.dumps({"title": "Test", "score": "not_an_int"}),
        )

        result = aggregator.aggregate([chunk], schema=sample_schema)

        assert result.schema_valid is False
        assert len(result.validation_errors) > 0


# ---------------------------------------------------------------------------
# Merge Strategy Tests
# ---------------------------------------------------------------------------


class TestMergeStrategies:
    """Tests for different merge strategies."""

    def test_first_wins_keeps_first_value(self) -> None:
        """Test FIRST_WINS keeps first non-null value."""
        aggregator = IFAggregator(default_strategy=AggregationStrategy.FIRST_WINS)

        chunk1 = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content=json.dumps({"title": "First Title"}),
        )
        chunk2 = IFChunkArtifact(
            artifact_id="c2",
            document_id="d1",
            content=json.dumps({"title": "Second Title"}),
        )

        result = aggregator.aggregate(
            [chunk1, chunk2],
            schema_def={"title": "string"},
        )

        assert result.data.get("title") == "First Title"

    def test_last_wins_keeps_last_value(self) -> None:
        """Test LAST_WINS keeps last value."""
        aggregator = IFAggregator(default_strategy=AggregationStrategy.LAST_WINS)

        chunk1 = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content=json.dumps({"title": "First Title"}),
        )
        chunk2 = IFChunkArtifact(
            artifact_id="c2",
            document_id="d1",
            content=json.dumps({"title": "Second Title"}),
        )

        result = aggregator.aggregate(
            [chunk1, chunk2],
            schema_def={"title": "string"},
        )

        assert result.data.get("title") == "Second Title"

    def test_per_field_strategy_overrides_default(self) -> None:
        """Test per-field strategy overrides default."""
        aggregator = IFAggregator(
            default_strategy=AggregationStrategy.FIRST_WINS,
            field_strategies={"title": AggregationStrategy.LAST_WINS},
        )

        chunk1 = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content=json.dumps({"title": "First", "author": "Alice"}),
        )
        chunk2 = IFChunkArtifact(
            artifact_id="c2",
            document_id="d1",
            content=json.dumps({"title": "Second", "author": "Bob"}),
        )

        result = aggregator.aggregate(
            [chunk1, chunk2],
            schema_def={"title": "string", "author": "string"},
        )

        # title uses LAST_WINS, author uses FIRST_WINS
        assert result.data.get("title") == "Second"
        assert result.data.get("author") == "Alice"


# ---------------------------------------------------------------------------
# Convenience Function Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_aggregate_artifacts_function(self) -> None:
        """Test aggregate_artifacts convenience function."""
        chunk = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content=json.dumps({"title": "Test"}),
        )

        result = aggregate_artifacts(
            [chunk],
            schema_def={"title": "string"},
        )

        assert isinstance(result, AggregationResult)
        assert result.data.get("title") == "Test"

    def test_aggregate_artifacts_with_strategy(self) -> None:
        """Test aggregate_artifacts with custom strategy."""
        chunk1 = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content=json.dumps({"title": "First"}),
        )
        chunk2 = IFChunkArtifact(
            artifact_id="c2",
            document_id="d1",
            content=json.dumps({"title": "Second"}),
        )

        result = aggregate_artifacts(
            [chunk1, chunk2],
            schema_def={"title": "string"},
            strategy=AggregationStrategy.LAST_WINS,
        )

        assert result.data.get("title") == "Second"

    def test_aggregate_to_json_function(self) -> None:
        """Test aggregate_to_json convenience function."""
        chunk = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content=json.dumps({"title": "Test", "score": 95}),
        )

        json_str = aggregate_to_json(
            [chunk],
            schema_def={"title": "string", "score": "int"},
        )

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed.get("title") == "Test"
        assert parsed.get("score") == 95


# ---------------------------------------------------------------------------
# JPL Power of Ten Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for NASA JPL Power of Ten rule compliance."""

    def test_jpl_rule_2_fixed_bounds(self) -> None:
        """JPL Rule #2: Fixed upper bounds are defined."""
        assert MAX_AGGREGATION_CHUNKS == 1000
        assert MAX_FIELDS == 64
        assert MAX_LIST_ITEMS == 10000

    def test_jpl_rule_2_bounds_enforced(self) -> None:
        """JPL Rule #2: Bounds are enforced."""
        aggregator = IFAggregator()

        # Create more chunks than allowed
        chunks = [
            IFChunkArtifact(
                artifact_id=f"c{i}",
                document_id="d1",
                content="",
            )
            for i in range(MAX_AGGREGATION_CHUNKS + 1)
        ]

        with pytest.raises(AssertionError):
            aggregator.aggregate(chunks)

    def test_jpl_rule_4_function_length(self) -> None:
        """JPL Rule #4: All functions < 60 lines."""
        import ingestforge.core.pipeline.aggregator as module

        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                try:
                    source_lines = inspect.getsourcelines(obj)[0]
                    code_lines = [
                        line
                        for line in source_lines
                        if line.strip()
                        and not line.strip().startswith(("#", "@", '"""', "'''"))
                    ]
                    assert (
                        len(code_lines) < 60
                    ), f"Function {name} has {len(code_lines)} lines"
                except (OSError, TypeError):
                    pass

    def test_jpl_rule_5_assertions_present(self) -> None:
        """JPL Rule #5: Assertions are present for preconditions."""
        import ingestforge.core.pipeline.aggregator as module

        source = inspect.getsource(module)
        assert "assert" in source, "No assertions found in module"

    def test_jpl_rule_9_type_hints(self) -> None:
        """JPL Rule #9: Key functions have type hints."""
        # Check specific functions
        for func_name in ["aggregate_artifacts", "aggregate_to_json"]:
            from ingestforge.core.pipeline import aggregator

            func = getattr(aggregator, func_name)
            hints = func.__annotations__
            assert "return" in hints, f"Function {func_name} missing return type hint"

    def test_jpl_rule_1_no_recursion(self) -> None:
        """JPL Rule #1: No recursion in key methods."""

        # Check that aggregate doesn't call itself
        aggregate_source = inspect.getsource(IFAggregator.aggregate)
        # Direct self.aggregate call would indicate recursion
        assert "self.aggregate(" not in aggregate_source.replace(
            "def aggregate(", ""
        ), "Recursive call detected in aggregate"


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_artifact_list_rejected(self) -> None:
        """Test empty artifact list is rejected."""
        aggregator = IFAggregator()

        with pytest.raises(AssertionError):
            aggregator.aggregate_to_artifact([])

    def test_none_artifacts_rejected(self) -> None:
        """Test None artifacts rejected."""
        aggregator = IFAggregator()

        with pytest.raises(AssertionError):
            aggregator.aggregate(None)  # type: ignore

    def test_no_schema_still_works(self) -> None:
        """Test aggregation without schema still extracts data."""
        aggregator = IFAggregator()

        chunk = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content=json.dumps({"title": "Test"}),
        )

        result = aggregator.aggregate([chunk])

        assert result.schema_valid is True  # No schema = valid
        assert "title" in result.data

    def test_artifact_with_empty_content(self) -> None:
        """Test artifact with empty content doesn't crash."""
        aggregator = IFAggregator()

        chunk = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content="",
        )

        result = aggregator.aggregate([chunk], schema_def={"title": "string"})

        assert isinstance(result, AggregationResult)

    def test_process_passthrough(self) -> None:
        """Test process() method is passthrough."""
        aggregator = IFAggregator()

        chunk = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content="test",
        )

        result = aggregator.process(chunk)

        assert result is chunk  # Same object returned

    def test_key_value_extraction_from_text(self) -> None:
        """Test key-value extraction from plain text."""
        aggregator = IFAggregator()

        chunk = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content="title: My Document\nauthor: John Doe\nscore = 95",
        )

        result = aggregator.aggregate(
            [chunk],
            schema_def={"title": "string", "author": "string", "score": "string"},
        )

        assert result.data.get("title") == "My Document"
        assert result.data.get("author") == "John Doe"
        assert result.data.get("score") == "95"


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests with schema generator."""

    def test_integration_with_pydantic_schema(self) -> None:
        """Test full integration with Pydantic schema."""

        class FactSheet(BaseModel):
            company: str
            revenue: Optional[float] = None
            employees: int = 0
            products: List[str] = Field(default_factory=list)

        aggregator = IFAggregator(
            field_strategies={"products": AggregationStrategy.MERGE_UNIQUE}
        )

        chunk1 = IFChunkArtifact(
            artifact_id="c1",
            document_id="d1",
            content=json.dumps(
                {
                    "company": "Acme Corp",
                    "revenue": 1000000.0,
                    "products": ["Widget A", "Widget B"],
                }
            ),
        )
        chunk2 = IFChunkArtifact(
            artifact_id="c2",
            document_id="d1",
            content=json.dumps(
                {
                    "employees": 500,
                    "products": ["Widget B", "Widget C"],
                }
            ),
        )

        result = aggregator.aggregate([chunk1, chunk2], schema=FactSheet)

        assert result.schema_valid is True
        assert result.data.get("company") == "Acme Corp"
        assert result.data.get("employees") == 500
        # Widget B should be deduplicated
        products = result.data.get("products", [])
        assert len(products) == 3
