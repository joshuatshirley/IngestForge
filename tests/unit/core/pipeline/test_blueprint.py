"""
Tests for Blueprint Parser & Validator ().

GWT (Given-When-Then) test structure.
NASA JPL Power of Ten compliance verification.
"""

import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import yaml

from ingestforge.core.pipeline.blueprint import (
    VerticalBlueprint,
    StageConfig,
    BlueprintParser,
    BlueprintRegistry,
    BlueprintValidationError,
    BlueprintLoadError,
    load_blueprint,
    load_blueprint_from_dict,
    get_blueprint_registry,
    MAX_STAGES,
    MAX_BLUEPRINTS,
    MAX_BLUEPRINT_SIZE,
    MAX_VERTICAL_ID_LENGTH,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_blueprint_dict() -> Dict[str, Any]:
    """Create a valid blueprint dictionary."""
    return {
        "vertical_id": "legal-discovery",
        "name": "Legal Document Discovery",
        "version": "1.0.0",
        "description": "Pipeline for legal document analysis",
        "stages": [
            {"processor": "IFLegalChunker", "config": {"preserve_citations": True}},
            {
                "processor": "IFEntityExtractor",
                "config": {"entity_types": ["PERSON", "ORG"]},
            },
        ],
        "output_schema": {
            "case_name": "string",
            "parties": "list[string]",
            "filing_date": "date?",
        },
    }


@pytest.fixture
def minimal_blueprint_dict() -> Dict[str, Any]:
    """Create a minimal valid blueprint dictionary."""
    return {
        "vertical_id": "minimal",
        "name": "Minimal Pipeline",
        "stages": [
            {"processor": "IFProcessor"},
        ],
    }


@pytest.fixture
def temp_blueprint_file(valid_blueprint_dict: Dict[str, Any]) -> Path:
    """Create a temporary blueprint YAML file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(valid_blueprint_dict, f)
        return Path(f.name)


@pytest.fixture
def temp_blueprint_dir(valid_blueprint_dict: Dict[str, Any]) -> Path:
    """Create a temporary directory with blueprint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)

        # Create valid blueprint
        with open(dir_path / "legal.yaml", "w", encoding="utf-8") as f:
            yaml.dump(valid_blueprint_dict, f)

        # Create another valid blueprint
        research_bp = {
            "vertical_id": "research",
            "name": "Research Pipeline",
            "stages": [{"processor": "IFTextExtractor"}],
        }
        with open(dir_path / "research.yml", "w", encoding="utf-8") as f:
            yaml.dump(research_bp, f)

        yield dir_path


@pytest.fixture
def parser() -> BlueprintParser:
    """Create a BlueprintParser instance."""
    return BlueprintParser()


@pytest.fixture
def registry() -> BlueprintRegistry:
    """Create a fresh BlueprintRegistry instance."""
    reg = BlueprintRegistry()
    reg.clear()
    return reg


# ---------------------------------------------------------------------------
# StageConfig Tests
# ---------------------------------------------------------------------------


class TestStageConfig:
    """Tests for StageConfig model."""

    def test_stage_config_creation(self):
        """Given valid processor, When creating StageConfig, Then stores all fields."""
        config = StageConfig(
            processor="IFLegalChunker",
            config={"preserve_citations": True},
            enabled=True,
        )

        assert config.processor == "IFLegalChunker"
        assert config.config["preserve_citations"] is True
        assert config.enabled is True

    def test_stage_config_defaults(self):
        """Given only processor, When creating StageConfig, Then uses defaults."""
        config = StageConfig(processor="IFProcessor")

        assert config.processor == "IFProcessor"
        assert config.config == {}
        assert config.enabled is True

    def test_stage_config_empty_processor_fails(self):
        """Given empty processor, When creating StageConfig, Then raises error."""
        with pytest.raises(Exception):
            StageConfig(processor="")

    def test_stage_config_disabled(self):
        """Given enabled=False, When creating StageConfig, Then stage is disabled."""
        config = StageConfig(processor="IFProcessor", enabled=False)

        assert config.enabled is False


# ---------------------------------------------------------------------------
# VerticalBlueprint Tests
# ---------------------------------------------------------------------------


class TestVerticalBlueprint:
    """Tests for VerticalBlueprint model."""

    def test_blueprint_creation(self, valid_blueprint_dict: Dict[str, Any]):
        """Given valid data, When creating VerticalBlueprint, Then stores all fields."""
        stages = [StageConfig(**s) for s in valid_blueprint_dict["stages"]]
        blueprint = VerticalBlueprint(
            vertical_id=valid_blueprint_dict["vertical_id"],
            name=valid_blueprint_dict["name"],
            version=valid_blueprint_dict["version"],
            stages=stages,
            output_schema=valid_blueprint_dict["output_schema"],
        )

        assert blueprint.vertical_id == "legal-discovery"
        assert blueprint.name == "Legal Document Discovery"
        assert len(blueprint.stages) == 2
        assert blueprint.output_schema is not None

    def test_blueprint_is_immutable(self, valid_blueprint_dict: Dict[str, Any]):
        """Given blueprint, When trying to modify, Then raises error."""
        stages = [StageConfig(**s) for s in valid_blueprint_dict["stages"]]
        blueprint = VerticalBlueprint(
            vertical_id="test",
            name="Test",
            stages=stages,
        )

        with pytest.raises(Exception):
            blueprint.vertical_id = "modified"

    def test_blueprint_vertical_id_validation(self):
        """Given invalid vertical_id, When creating blueprint, Then raises error."""
        with pytest.raises(Exception):
            VerticalBlueprint(
                vertical_id="invalid id with spaces",
                name="Test",
                stages=[StageConfig(processor="IFProcessor")],
            )

    def test_blueprint_empty_stages_fails(self):
        """Given empty stages, When creating blueprint, Then raises error."""
        with pytest.raises(Exception):
            VerticalBlueprint(
                vertical_id="test",
                name="Test",
                stages=[],
            )

    def test_blueprint_output_schema_validation(self):
        """Given invalid output_schema, When creating blueprint, Then raises error."""
        with pytest.raises(Exception):
            VerticalBlueprint(
                vertical_id="test",
                name="Test",
                stages=[StageConfig(processor="IFProcessor")],
                output_schema={"field": "invalid_type_xyz"},
            )

    def test_blueprint_vertical_id_normalization(self):
        """Given uppercase vertical_id, When creating blueprint, Then lowercases it."""
        blueprint = VerticalBlueprint(
            vertical_id="Legal-Discovery",
            name="Test",
            stages=[StageConfig(processor="IFProcessor")],
        )

        assert blueprint.vertical_id == "legal-discovery"


# ---------------------------------------------------------------------------
# BlueprintParser Tests
# ---------------------------------------------------------------------------


class TestBlueprintParser:
    """Tests for BlueprintParser class."""

    def test_parser_creation(self, parser: BlueprintParser):
        """Given no args, When creating parser, Then initializes correctly."""
        assert parser is not None

    def test_parse_dict_valid(
        self, parser: BlueprintParser, valid_blueprint_dict: Dict[str, Any]
    ):
        """Given valid dict, When parsing, Then returns VerticalBlueprint."""
        blueprint = parser.parse_dict(valid_blueprint_dict)

        assert isinstance(blueprint, VerticalBlueprint)
        assert blueprint.vertical_id == "legal-discovery"

    def test_parse_dict_minimal(
        self, parser: BlueprintParser, minimal_blueprint_dict: Dict[str, Any]
    ):
        """Given minimal dict, When parsing, Then returns VerticalBlueprint."""
        blueprint = parser.parse_dict(minimal_blueprint_dict)

        assert isinstance(blueprint, VerticalBlueprint)
        assert blueprint.vertical_id == "minimal"
        assert blueprint.version == "1.0.0"  # Default

    def test_parse_dict_missing_vertical_id(self, parser: BlueprintParser):
        """Given dict without vertical_id, When parsing, Then raises error."""
        with pytest.raises(BlueprintValidationError) as exc_info:
            parser.parse_dict({"name": "Test", "stages": [{"processor": "P"}]})

        assert "vertical_id" in str(exc_info.value)

    def test_parse_dict_missing_name(self, parser: BlueprintParser):
        """Given dict without name, When parsing, Then raises error."""
        with pytest.raises(BlueprintValidationError) as exc_info:
            parser.parse_dict({"vertical_id": "test", "stages": [{"processor": "P"}]})

        assert "name" in str(exc_info.value)

    def test_parse_dict_missing_stages(self, parser: BlueprintParser):
        """Given dict without stages, When parsing, Then raises error."""
        with pytest.raises(BlueprintValidationError) as exc_info:
            parser.parse_dict({"vertical_id": "test", "name": "Test"})

        assert "stages" in str(exc_info.value)

    def test_parse_dict_stages_not_list(self, parser: BlueprintParser):
        """Given stages not a list, When parsing, Then raises error."""
        with pytest.raises(BlueprintValidationError) as exc_info:
            parser.parse_dict(
                {
                    "vertical_id": "test",
                    "name": "Test",
                    "stages": "not a list",
                }
            )

        assert "stages" in str(exc_info.value)

    def test_parse_file_valid(self, parser: BlueprintParser, temp_blueprint_file: Path):
        """Given valid YAML file, When parsing, Then returns VerticalBlueprint."""
        blueprint = parser.parse_file(temp_blueprint_file)

        assert isinstance(blueprint, VerticalBlueprint)
        assert blueprint.vertical_id == "legal-discovery"

        # Cleanup
        temp_blueprint_file.unlink()

    def test_parse_file_not_found(self, parser: BlueprintParser):
        """Given non-existent file, When parsing, Then raises BlueprintLoadError."""
        with pytest.raises(BlueprintLoadError) as exc_info:
            parser.parse_file(Path("/nonexistent/file.yaml"))

        assert "not found" in str(exc_info.value)

    def test_parse_file_invalid_yaml(self, parser: BlueprintParser):
        """Given invalid YAML, When parsing, Then raises BlueprintLoadError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("invalid: yaml: content: [")
            path = Path(f.name)

        with pytest.raises(BlueprintLoadError) as exc_info:
            parser.parse_file(path)

        assert "YAML" in str(exc_info.value)
        path.unlink()

    def test_parse_file_empty(self, parser: BlueprintParser):
        """Given empty file, When parsing, Then raises BlueprintValidationError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            path = Path(f.name)

        with pytest.raises(BlueprintValidationError) as exc_info:
            parser.parse_file(path)

        assert "empty" in str(exc_info.value).lower()
        path.unlink()

    def test_validate_warnings_duplicate_processors(
        self, parser: BlueprintParser, valid_blueprint_dict: Dict[str, Any]
    ):
        """Given duplicate processors, When validating, Then returns warnings."""
        valid_blueprint_dict["stages"].append(
            {"processor": "IFLegalChunker", "config": {}}
        )
        blueprint = parser.parse_dict(valid_blueprint_dict)

        warnings = parser.validate(blueprint)

        assert len(warnings) > 0
        assert any("Duplicate" in w for w in warnings)

    def test_validate_warnings_disabled_stages(
        self, parser: BlueprintParser, valid_blueprint_dict: Dict[str, Any]
    ):
        """Given disabled stages, When validating, Then returns warnings."""
        valid_blueprint_dict["stages"][0]["enabled"] = False
        blueprint = parser.parse_dict(valid_blueprint_dict)

        warnings = parser.validate(blueprint)

        assert len(warnings) > 0
        assert any("disabled" in w for w in warnings)


# ---------------------------------------------------------------------------
# BlueprintRegistry Tests
# ---------------------------------------------------------------------------


class TestBlueprintRegistry:
    """Tests for BlueprintRegistry class."""

    def test_registry_singleton(self):
        """Given multiple instantiations, When creating, Then returns same instance."""
        reg1 = BlueprintRegistry()
        reg2 = BlueprintRegistry()

        assert reg1 is reg2

    def test_registry_register(
        self, registry: BlueprintRegistry, valid_blueprint_dict: Dict[str, Any]
    ):
        """Given valid blueprint, When registering, Then stores it."""
        blueprint = load_blueprint_from_dict(valid_blueprint_dict)
        registry.register(blueprint)

        assert registry.get("legal-discovery") is not None
        assert registry.get("legal-discovery").name == "Legal Document Discovery"

    def test_registry_get_not_found(self, registry: BlueprintRegistry):
        """Given unknown vertical_id, When getting, Then returns None."""
        result = registry.get("nonexistent")

        assert result is None

    def test_registry_list_verticals(
        self, registry: BlueprintRegistry, valid_blueprint_dict: Dict[str, Any]
    ):
        """Given registered blueprints, When listing, Then returns all IDs."""
        bp1 = load_blueprint_from_dict(valid_blueprint_dict)
        valid_blueprint_dict["vertical_id"] = "research"
        bp2 = load_blueprint_from_dict(valid_blueprint_dict)

        registry.register(bp1)
        registry.register(bp2)

        verticals = registry.list_verticals()

        assert "legal-discovery" in verticals
        assert "research" in verticals

    def test_registry_clear(
        self, registry: BlueprintRegistry, valid_blueprint_dict: Dict[str, Any]
    ):
        """Given registered blueprints, When clearing, Then removes all."""
        blueprint = load_blueprint_from_dict(valid_blueprint_dict)
        registry.register(blueprint)

        registry.clear()

        assert registry.list_verticals() == []

    def test_registry_get_output_model(
        self, registry: BlueprintRegistry, valid_blueprint_dict: Dict[str, Any]
    ):
        """Given blueprint with output_schema, When getting model, Then returns it."""
        blueprint = load_blueprint_from_dict(valid_blueprint_dict)
        registry.register(blueprint)

        model = registry.get_output_model("legal-discovery")

        assert model is not None
        assert hasattr(model, "model_fields")
        assert "case_name" in model.model_fields

    def test_registry_get_output_model_not_found(self, registry: BlueprintRegistry):
        """Given unknown vertical, When getting model, Then returns None."""
        model = registry.get_output_model("nonexistent")

        assert model is None


# ---------------------------------------------------------------------------
# GWT Behavioral Tests
# ---------------------------------------------------------------------------


class TestGWTBehavior:
    """
    Given-When-Then behavioral tests for .

    GWT:
    - GWT-1: Valid blueprint parsing returns validated model.
    - GWT-2: Invalid blueprint raises BlueprintValidationError.
    - GWT-3: Directory scan registers all valid blueprints.
    """

    def test_gwt1_valid_blueprint_parsing(
        self, parser: BlueprintParser, valid_blueprint_dict: Dict[str, Any]
    ):
        """
        GWT-1: Given a valid YAML conforming to schema.
        When: The parser loads it.
        Then: Returns validated VerticalBlueprint with all fields typed.
        """
        blueprint = parser.parse_dict(valid_blueprint_dict)

        # Verify it's a validated VerticalBlueprint
        assert isinstance(blueprint, VerticalBlueprint)

        # Verify all fields are properly typed
        assert isinstance(blueprint.vertical_id, str)
        assert isinstance(blueprint.name, str)
        assert isinstance(blueprint.stages, list)
        assert all(isinstance(s, StageConfig) for s in blueprint.stages)
        assert isinstance(blueprint.output_schema, dict)

    def test_gwt2_invalid_blueprint_rejection(self, parser: BlueprintParser):
        """
        GWT-2: Given a YAML with missing required fields.
        When: The parser attempts to load it.
        Then: Raises BlueprintValidationError with clear message.
        """
        invalid_data = {
            "vertical_id": "test",
            # Missing 'name' and 'stages'
        }

        with pytest.raises(BlueprintValidationError) as exc_info:
            parser.parse_dict(invalid_data)

        # Verify clear error message
        error = exc_info.value
        assert error.field is not None or "name" in str(error) or "stages" in str(error)

    def test_gwt3_directory_scan_registers_all(self, temp_blueprint_dir: Path):
        """
        GWT-3: Given a directory with multiple blueprint files.
        When: The registry scans the directory.
        Then: All valid blueprints are registered by vertical_id.
        """
        registry = BlueprintRegistry()
        registry.clear()

        result = registry.scan_directory(temp_blueprint_dir)

        # Verify scan results
        assert result["loaded"] == 2
        assert result["failed"] == 0

        # Verify registration by vertical_id
        assert registry.get("legal-discovery") is not None
        assert registry.get("research") is not None


# ---------------------------------------------------------------------------
# JPL Power of Ten Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for NASA JPL Power of Ten compliance."""

    def test_jpl_rule_2_max_stages(self):
        """JPL Rule #2: Verify MAX_STAGES bound."""
        assert MAX_STAGES == 20

    def test_jpl_rule_2_max_stages_enforced(self, parser: BlueprintParser):
        """JPL Rule #2: Too many stages raises error."""
        data = {
            "vertical_id": "test",
            "name": "Test",
            "stages": [{"processor": f"P{i}"} for i in range(MAX_STAGES + 1)],
        }

        with pytest.raises(Exception):
            parser.parse_dict(data)

    def test_jpl_rule_2_max_blueprints(self):
        """JPL Rule #2: Verify MAX_BLUEPRINTS bound."""
        assert MAX_BLUEPRINTS == 64

    def test_jpl_rule_2_max_blueprint_size(self):
        """JPL Rule #2: Verify MAX_BLUEPRINT_SIZE bound."""
        assert MAX_BLUEPRINT_SIZE == 65536

    def test_jpl_rule_2_max_vertical_id_length(self):
        """JPL Rule #2: Verify MAX_VERTICAL_ID_LENGTH bound."""
        assert MAX_VERTICAL_ID_LENGTH == 64

    def test_jpl_rule_2_vertical_id_length_enforced(self, parser: BlueprintParser):
        """JPL Rule #2: Too long vertical_id raises error."""
        data = {
            "vertical_id": "a" * (MAX_VERTICAL_ID_LENGTH + 1),
            "name": "Test",
            "stages": [{"processor": "P"}],
        }

        with pytest.raises(Exception):
            parser.parse_dict(data)

    def test_jpl_rule_4_parser_method_sizes(self):
        """JPL Rule #4: All parser methods should be < 60 lines."""
        import inspect

        methods = [
            "parse_file",
            "parse_dict",
            "validate",
        ]

        for method_name in methods:
            method = getattr(BlueprintParser, method_name)
            source = inspect.getsource(method)
            lines = len(source.split("\n"))
            assert lines < 60, f"{method_name} has {lines} lines (limit: 60)"

    def test_jpl_rule_5_assertions_in_parse_file(self):
        """JPL Rule #5: parse_file has assertions."""
        import inspect

        source = inspect.getsource(BlueprintParser.parse_file)

        assert "assert" in source

    def test_jpl_rule_5_assertions_in_parse_dict(self):
        """JPL Rule #5: parse_dict has assertions."""
        import inspect

        source = inspect.getsource(BlueprintParser.parse_dict)

        assert "assert" in source

    def test_jpl_rule_9_type_hints(self):
        """JPL Rule #9: Verify complete type hints."""
        import inspect

        # Check BlueprintParser methods
        methods = ["parse_file", "parse_dict", "validate"]
        for method_name in methods:
            method = getattr(BlueprintParser, method_name)
            sig = inspect.signature(method)
            assert (
                sig.return_annotation != inspect.Parameter.empty
            ), f"BlueprintParser.{method_name} missing return type hint"

        # Check BlueprintRegistry methods
        methods = ["register", "get", "scan_directory", "list_verticals"]
        for method_name in methods:
            method = getattr(BlueprintRegistry, method_name)
            sig = inspect.signature(method)
            assert (
                sig.return_annotation != inspect.Parameter.empty
            ), f"BlueprintRegistry.{method_name} missing return type hint"


# ---------------------------------------------------------------------------
# Convenience Function Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_load_blueprint(self, temp_blueprint_file: Path):
        """Given valid file, When using load_blueprint, Then returns blueprint."""
        blueprint = load_blueprint(temp_blueprint_file)

        assert isinstance(blueprint, VerticalBlueprint)
        assert blueprint.vertical_id == "legal-discovery"

        temp_blueprint_file.unlink()

    def test_load_blueprint_from_dict(self, valid_blueprint_dict: Dict[str, Any]):
        """Given valid dict, When using load_blueprint_from_dict, Then returns blueprint."""
        blueprint = load_blueprint_from_dict(valid_blueprint_dict)

        assert isinstance(blueprint, VerticalBlueprint)
        assert blueprint.vertical_id == "legal-discovery"

    def test_get_blueprint_registry(self):
        """Given function call, When getting registry, Then returns singleton."""
        reg1 = get_blueprint_registry()
        reg2 = get_blueprint_registry()

        assert reg1 is reg2
        assert isinstance(reg1, BlueprintRegistry)


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_blueprint_with_special_characters_in_id(self, parser: BlueprintParser):
        """Given vertical_id with allowed special chars, When parsing, Then succeeds."""
        data = {
            "vertical_id": "legal-doc_v2",
            "name": "Test",
            "stages": [{"processor": "P"}],
        }

        blueprint = parser.parse_dict(data)

        assert blueprint.vertical_id == "legal-doc_v2"

    def test_blueprint_with_invalid_special_chars(self, parser: BlueprintParser):
        """Given vertical_id with invalid chars, When parsing, Then fails."""
        data = {
            "vertical_id": "legal.doc",  # Period not allowed
            "name": "Test",
            "stages": [{"processor": "P"}],
        }

        with pytest.raises(BlueprintValidationError):
            parser.parse_dict(data)

    def test_scan_nonexistent_directory(self, registry: BlueprintRegistry):
        """Given nonexistent directory, When scanning, Then returns empty result."""
        result = registry.scan_directory(Path("/nonexistent/dir"))

        assert result["loaded"] == 0
        assert result["failed"] == 0

    def test_scan_directory_with_invalid_files(self, registry: BlueprintRegistry):
        """Given directory with invalid blueprints, When scanning, Then tracks failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            # Create invalid blueprint
            with open(dir_path / "invalid.yaml", "w", encoding="utf-8") as f:
                yaml.dump({"invalid": "data"}, f)

            result = registry.scan_directory(dir_path)

            assert result["failed"] == 1
            assert len(result["errors"]) == 1

    def test_blueprint_with_empty_config(self, parser: BlueprintParser):
        """Given stage with empty config, When parsing, Then uses default."""
        data = {
            "vertical_id": "test",
            "name": "Test",
            "stages": [{"processor": "P", "config": {}}],
        }

        blueprint = parser.parse_dict(data)

        assert blueprint.stages[0].config == {}

    def test_blueprint_without_output_schema(self, parser: BlueprintParser):
        """Given blueprint without output_schema, When parsing, Then succeeds."""
        data = {
            "vertical_id": "test",
            "name": "Test",
            "stages": [{"processor": "P"}],
        }

        blueprint = parser.parse_dict(data)

        assert blueprint.output_schema is None

    def test_registry_blueprints_property(
        self, registry: BlueprintRegistry, valid_blueprint_dict: Dict[str, Any]
    ):
        """Given registered blueprints, When accessing .blueprints, Then returns copy."""
        blueprint = load_blueprint_from_dict(valid_blueprint_dict)
        registry.register(blueprint)

        blueprints = registry.blueprints

        # Verify it's a copy (modifications don't affect registry)
        blueprints["test"] = None
        assert registry.get("test") is None
        assert registry.get("legal-discovery") is not None
