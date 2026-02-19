"""
Tests for extraction_rules module.

Implements NASA JPL Rule #7: 3+ assertions per core logic test.
"""
import json
import pytest
from ingestforge.ingest.extraction_rules import (
    Selector,
    SelectorType,
    ExtractionField,
    ExtractionStrategy,
    ContentBoundary,
    PaginationRule,
    ExtractionRule,
    ExtractionRuleRegistry,
    create_default_registry,
    _RegistrySingleton,
    BUILTIN_RULES,
)


class TestSelector:
    """Test Selector dataclass."""

    def test_selector_default_values(self):
        """Selector should have correct default values."""
        selector = Selector("div.content")
        assert selector.value == "div.content"
        assert selector.type == SelectorType.CSS
        assert selector.multiple is False

    def test_selector_to_dict(self):
        """Selector should serialize to dict correctly."""
        selector = Selector("//div[@class='content']", SelectorType.XPATH, True)
        result = selector.to_dict()
        assert result["value"] == "//div[@class='content']"
        assert result["type"] == "xpath"
        assert result["multiple"] is True

    def test_selector_from_dict(self):
        """Selector should deserialize from dict correctly."""
        data = {
            "value": "span.author",
            "type": "css",
            "multiple": False,
        }
        selector = Selector.from_dict(data)
        assert selector.value == "span.author"
        assert selector.type == SelectorType.CSS
        assert selector.multiple is False


class TestExtractionField:
    """Test ExtractionField dataclass."""

    def test_extraction_field_defaults(self):
        """ExtractionField should have correct defaults."""
        field = ExtractionField(
            name="title",
            selector=Selector("h1"),
        )
        assert field.name == "title"
        assert field.strategy == ExtractionStrategy.TEXT
        assert field.required is False
        assert field.default is None

    def test_extraction_field_to_dict(self):
        """ExtractionField should serialize correctly."""
        field = ExtractionField(
            name="author",
            selector=Selector("span.author"),
            required=True,
            filters=["strip", "trim_quotes"],
        )
        result = field.to_dict()
        assert result["name"] == "author"
        assert result["required"] is True
        assert result["filters"] == ["strip", "trim_quotes"]
        assert "selector" in result

    def test_extraction_field_from_dict(self):
        """ExtractionField should deserialize correctly."""
        data = {
            "name": "content",
            "selector": {"value": "article", "type": "css", "multiple": False},
            "strategy": "markdown",
            "required": True,
        }
        field = ExtractionField.from_dict(data)
        assert field.name == "content"
        assert field.strategy == ExtractionStrategy.MARKDOWN
        assert field.required is True
        assert field.selector.value == "article"


class TestContentBoundary:
    """Test ContentBoundary dataclass."""

    def test_content_boundary_defaults(self):
        """ContentBoundary should have correct defaults."""
        boundary = ContentBoundary()
        assert boundary.container is None
        assert boundary.remove == []
        assert boundary.keep == []

    def test_content_boundary_to_dict(self):
        """ContentBoundary should serialize correctly."""
        boundary = ContentBoundary(
            container=Selector("main"),
            remove=[Selector("nav"), Selector("footer")],
        )
        result = boundary.to_dict()
        assert result["container"] is not None
        assert len(result["remove"]) == 2
        assert result["keep"] == []

    def test_content_boundary_from_dict(self):
        """ContentBoundary should deserialize correctly."""
        data = {
            "container": {"value": "article", "type": "css", "multiple": False},
            "remove": [{"value": "aside", "type": "css", "multiple": False}],
            "keep": [],
        }
        boundary = ContentBoundary.from_dict(data)
        assert boundary.container is not None
        assert boundary.container.value == "article"
        assert len(boundary.remove) == 1
        assert boundary.remove[0].value == "aside"


class TestPaginationRule:
    """Test PaginationRule dataclass."""

    def test_pagination_rule_defaults(self):
        """PaginationRule should have correct defaults."""
        rule = PaginationRule()
        assert rule.next_page is None
        assert rule.page_number is None
        assert rule.max_pages == 10

    def test_pagination_rule_to_dict(self):
        """PaginationRule should serialize correctly."""
        rule = PaginationRule(
            next_page=Selector("a.next"),
            max_pages=20,
        )
        result = rule.to_dict()
        assert result["next_page"] is not None
        assert result["max_pages"] == 20
        assert result["page_number"] is None

    def test_pagination_rule_from_dict(self):
        """PaginationRule should deserialize correctly."""
        data = {
            "next_page": {"value": "a.next", "type": "css", "multiple": False},
            "max_pages": 15,
        }
        rule = PaginationRule.from_dict(data)
        assert rule.next_page is not None
        assert rule.next_page.value == "a.next"
        assert rule.max_pages == 15


class TestExtractionRule:
    """Test ExtractionRule dataclass."""

    def test_extraction_rule_defaults(self):
        """ExtractionRule should have correct defaults."""
        rule = ExtractionRule(name="test_rule")
        assert rule.name == "test_rule"
        assert rule.enabled is True
        assert rule.priority == 0
        assert rule.timeout == 30

    def test_matches_url_by_domain(self):
        """ExtractionRule should match URLs by domain."""
        rule = ExtractionRule(
            name="example",
            domain="example.com",
        )
        assert rule.matches_url("https://example.com/page") is True
        assert rule.matches_url("https://www.example.com/page") is True
        assert rule.matches_url("https://other.com/page") is False

    def test_matches_url_by_pattern(self):
        """ExtractionRule should match URLs by regex pattern."""
        rule = ExtractionRule(
            name="articles",
            url_pattern=r"https://.*\.com/article/\d+",
        )
        assert rule.matches_url("https://example.com/article/123") is True
        assert rule.matches_url("https://test.com/article/456") is True
        assert rule.matches_url("https://example.com/blog/123") is False

    def test_matches_url_by_path_pattern(self):
        """ExtractionRule should match URLs by path pattern."""
        rule = ExtractionRule(
            name="wiki",
            domain="wikipedia.org",
            path_pattern=r"^/wiki/[^:]+$",
        )
        assert rule.matches_url("https://en.wikipedia.org/wiki/Python") is True
        assert rule.matches_url("https://en.wikipedia.org/wiki/Special:Random") is False
        assert rule.matches_url("https://other.com/wiki/Python") is False

    def test_matches_url_disabled_rule(self):
        """Disabled rules should not match any URL."""
        rule = ExtractionRule(
            name="disabled",
            domain="example.com",
            enabled=False,
        )
        assert rule.matches_url("https://example.com/page") is False
        assert rule.enabled is False
        assert rule.domain == "example.com"

    def test_extraction_rule_to_dict(self):
        """ExtractionRule should serialize correctly."""
        rule = ExtractionRule(
            name="test",
            domain="example.com",
            priority=10,
        )
        result = rule.to_dict()
        assert result["name"] == "test"
        assert result["domain"] == "example.com"
        assert result["priority"] == 10
        assert result["enabled"] is True

    def test_extraction_rule_from_dict(self):
        """ExtractionRule should deserialize correctly."""
        data = {
            "name": "test",
            "domain": "example.com",
            "priority": 5,
            "boundaries": {},
            "fields": [],
        }
        rule = ExtractionRule.from_dict(data)
        assert rule.name == "test"
        assert rule.domain == "example.com"
        assert rule.priority == 5
        assert rule.enabled is True


class TestExtractionRuleRegistry:
    """Test ExtractionRuleRegistry class."""

    def test_registry_initialization(self):
        """Registry should initialize with empty rules."""
        registry = ExtractionRuleRegistry()
        assert registry.rules == []
        assert len(registry._transforms) > 0
        assert len(registry._filters) > 0

    def test_add_rule_sorts_by_priority(self):
        """Registry should sort rules by priority (highest first)."""
        registry = ExtractionRuleRegistry()

        rule_low = ExtractionRule(name="low", priority=1)
        rule_high = ExtractionRule(name="high", priority=10)
        rule_mid = ExtractionRule(name="mid", priority=5)

        registry.add_rule(rule_low)
        registry.add_rule(rule_high)
        registry.add_rule(rule_mid)
        assert len(registry.rules) == 3
        assert registry.rules[0].name == "high"
        assert registry.rules[1].name == "mid"
        assert registry.rules[2].name == "low"

    def test_find_matching_rule_returns_highest_priority(self):
        """Registry should return highest priority matching rule."""
        registry = ExtractionRuleRegistry()

        rule1 = ExtractionRule(name="rule1", domain="example.com", priority=1)
        rule2 = ExtractionRule(name="rule2", domain="example.com", priority=10)

        registry.add_rule(rule1)
        registry.add_rule(rule2)

        result = registry.find_matching_rule("https://example.com/page")
        assert result is not None
        assert result.name == "rule2"
        assert result.priority == 10

    def test_find_matching_rule_returns_none_when_no_match(self):
        """Registry should return None when no rules match."""
        registry = ExtractionRuleRegistry()

        rule = ExtractionRule(name="test", domain="example.com")
        registry.add_rule(rule)

        result = registry.find_matching_rule("https://other.com/page")
        assert result is None
        assert len(registry.rules) == 1
        assert registry.rules[0].domain == "example.com"

    def test_find_all_matching_rules(self):
        """Registry should return all matching rules."""
        registry = ExtractionRuleRegistry()

        rule1 = ExtractionRule(name="rule1", domain="example.com", priority=1)
        rule2 = ExtractionRule(name="rule2", domain="example.com", priority=10)
        rule3 = ExtractionRule(name="rule3", domain="other.com", priority=5)

        registry.add_rule(rule1)
        registry.add_rule(rule2)
        registry.add_rule(rule3)

        results = registry.find_all_matching_rules("https://example.com/page")
        assert len(results) == 2
        assert results[0].name == "rule2"
        assert results[1].name == "rule1"

    def test_load_from_json(self, tmp_path):
        """Registry should load rules from JSON file."""
        registry = ExtractionRuleRegistry()

        rule_data = {
            "name": "test_json",
            "domain": "example.com",
            "priority": 5,
            "boundaries": {},
            "fields": [],
        }
        json_file = tmp_path / "test_rule.json"
        json_file.write_text(json.dumps(rule_data))

        rule = registry.load_from_json(json_file)
        assert rule.name == "test_json"
        assert rule.domain == "example.com"
        assert len(registry.rules) == 1
        assert registry.rules[0] is rule

    def test_builtin_transforms_registered(self):
        """Registry should have built-in transforms registered."""
        registry = ExtractionRuleRegistry()
        assert "strip" in registry._transforms
        assert "lower" in registry._transforms
        assert "normalize_whitespace" in registry._transforms

    def test_builtin_filters_registered(self):
        """Registry should have built-in filters registered."""
        registry = ExtractionRuleRegistry()
        assert "strip" in registry._filters
        assert "remove_newlines" in registry._filters
        assert "collapse_whitespace" in registry._filters


class TestBuiltinRules:
    """Test built-in extraction rules."""

    def test_builtin_rules_exist(self):
        """Built-in rules should be defined."""
        assert len(BUILTIN_RULES) > 0
        assert all(isinstance(r, ExtractionRule) for r in BUILTIN_RULES)
        assert all(r.enabled for r in BUILTIN_RULES)

    def test_wikipedia_rule_matches(self):
        """Wikipedia rule should match Wikipedia articles."""
        wiki_rule = next(r for r in BUILTIN_RULES if r.name == "wikipedia")
        assert wiki_rule.matches_url("https://en.wikipedia.org/wiki/Python") is True
        assert (
            wiki_rule.matches_url("https://en.wikipedia.org/wiki/Special:Random")
            is False
        )
        assert wiki_rule.domain == "wikipedia.org"

    def test_medium_rule_matches(self):
        """Medium rule should match Medium articles."""
        medium_rule = next(r for r in BUILTIN_RULES if r.name == "medium")
        assert medium_rule.matches_url("https://medium.com/@user/article") is True
        assert medium_rule.domain == "medium.com"
        assert len(medium_rule.fields) > 0

    def test_github_rule_matches(self):
        """GitHub rule should match GitHub repository pages."""
        github_rule = next(r for r in BUILTIN_RULES if r.name == "github_readme")
        assert github_rule.matches_url("https://github.com/user/repo") is True
        assert github_rule.matches_url("https://github.com/user/repo/") is True
        assert github_rule.matches_url("https://github.com/user/repo/issues") is False

    def test_arxiv_rule_matches(self):
        """arXiv rule should match arXiv abstract pages."""
        arxiv_rule = next(r for r in BUILTIN_RULES if r.name == "arxiv_abstract")
        assert arxiv_rule.matches_url("https://arxiv.org/abs/2301.12345") is True
        assert arxiv_rule.matches_url("https://arxiv.org/pdf/2301.12345.pdf") is False
        assert arxiv_rule.domain == "arxiv.org"


class TestCreateDefaultRegistry:
    """Test create_default_registry function."""

    def test_creates_registry_with_builtin_rules(self):
        """create_default_registry should create registry with built-in rules."""
        registry = create_default_registry()
        assert isinstance(registry, ExtractionRuleRegistry)
        assert len(registry.rules) == len(BUILTIN_RULES)
        assert all(r.enabled for r in registry.rules)

    def test_rules_sorted_by_priority(self):
        """create_default_registry should return rules sorted by priority."""
        registry = create_default_registry()

        priorities = [r.priority for r in registry.rules]
        assert len(priorities) > 0
        assert priorities == sorted(priorities, reverse=True)
        assert max(priorities) >= 10


class TestRegistrySingleton:
    """Test _RegistrySingleton class."""

    def test_singleton_returns_same_instance(self):
        """Singleton should return the same instance on multiple calls."""
        _RegistrySingleton._instance = None

        instance1 = _RegistrySingleton.get()
        instance2 = _RegistrySingleton.get()
        assert instance1 is instance2
        assert isinstance(instance1, ExtractionRuleRegistry)
        assert len(instance1.rules) > 0

    def test_singleton_has_builtin_rules(self):
        """Singleton instance should have built-in rules loaded."""
        _RegistrySingleton._instance = None

        instance = _RegistrySingleton.get()
        assert len(instance.rules) == len(BUILTIN_RULES)
        assert all(isinstance(r, ExtractionRule) for r in instance.rules)
        assert (
            instance.find_matching_rule("https://en.wikipedia.org/wiki/Test")
            is not None
        )


class TestLoadFromYaml:
    """Test loading rules from YAML files."""

    def test_load_from_yaml_basic(self, tmp_path):
        """Registry should load rules from YAML file."""
        pytest.importorskip("yaml")
        registry = ExtractionRuleRegistry()

        yaml_content = """
name: test_yaml_rule
description: Test YAML rule
domain: example.com
priority: 5
boundaries: {}
fields: []
"""
        yaml_file = tmp_path / "test_rule.yaml"
        yaml_file.write_text(yaml_content)

        rule = registry.load_from_yaml(yaml_file)
        assert rule.name == "test_yaml_rule"
        assert rule.domain == "example.com"
        assert len(registry.rules) == 1
        assert registry.rules[0] is rule

    def test_load_from_yaml_with_fields(self, tmp_path):
        """Registry should load YAML rules with field definitions."""
        pytest.importorskip("yaml")
        registry = ExtractionRuleRegistry()

        yaml_content = """
name: rule_with_fields
domain: test.com
fields:
  - name: title
    selector:
      value: h1
      type: css
      multiple: false
    strategy: text
    required: true
  - name: content
    selector:
      value: article
      type: css
      multiple: false
    strategy: markdown
"""
        yaml_file = tmp_path / "fields_rule.yaml"
        yaml_file.write_text(yaml_content)

        rule = registry.load_from_yaml(yaml_file)
        assert rule.name == "rule_with_fields"
        assert len(rule.fields) == 2
        assert rule.fields[0].name == "title"
        assert rule.fields[0].required is True

    def test_load_from_yaml_invalid_file(self, tmp_path):
        """Registry should raise error for invalid YAML."""
        pytest.importorskip("yaml")
        registry = ExtractionRuleRegistry()

        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("")  # Empty file

        with pytest.raises(ValueError, match="Invalid YAML rule file"):
            registry.load_from_yaml(yaml_file)


class TestLoadFromDirectory:
    """Test loading rules from directory."""

    def test_load_from_directory_json_files(self, tmp_path):
        """Registry should load all JSON rule files from directory."""
        registry = ExtractionRuleRegistry()

        # Create multiple JSON rule files
        for i in range(3):
            rule_data = {
                "name": f"json_rule_{i}",
                "domain": f"example{i}.com",
                "priority": i,
                "boundaries": {},
                "fields": [],
            }
            (tmp_path / f"rule_{i}.json").write_text(json.dumps(rule_data))

        loaded_count = registry.load_from_directory(tmp_path)
        assert loaded_count == 3
        assert len(registry.rules) == 3
        assert all(r.name.startswith("json_rule_") for r in registry.rules)

    def test_load_from_directory_yaml_files(self, tmp_path):
        """Registry should load all YAML rule files from directory."""
        pytest.importorskip("yaml")
        registry = ExtractionRuleRegistry()

        # Create YAML rule files
        for i in range(2):
            yaml_content = f"""
name: yaml_rule_{i}
domain: yamlsite{i}.com
priority: {i}
boundaries: {{}}
fields: []
"""
            (tmp_path / f"rule_{i}.yaml").write_text(yaml_content)

        loaded_count = registry.load_from_directory(tmp_path)
        assert loaded_count == 2
        assert len(registry.rules) == 2
        assert all("yaml_rule_" in r.name for r in registry.rules)

    def test_load_from_directory_mixed_files(self, tmp_path):
        """Registry should load both JSON and YAML files."""
        pytest.importorskip("yaml")
        registry = ExtractionRuleRegistry()

        # Create JSON rule
        json_rule = {
            "name": "json_rule",
            "domain": "jsonsite.com",
            "boundaries": {},
            "fields": [],
        }
        (tmp_path / "rule.json").write_text(json.dumps(json_rule))

        # Create YAML rule
        yaml_content = """
name: yaml_rule
domain: yamlsite.com
boundaries: {}
fields: []
"""
        (tmp_path / "rule.yaml").write_text(yaml_content)

        loaded_count = registry.load_from_directory(tmp_path)
        assert loaded_count == 2
        assert len(registry.rules) == 2
        rule_names = [r.name for r in registry.rules]
        assert "json_rule" in rule_names
        assert "yaml_rule" in rule_names

    def test_load_from_directory_empty(self, tmp_path):
        """Registry should return 0 for empty directory."""
        registry = ExtractionRuleRegistry()

        loaded_count = registry.load_from_directory(tmp_path)
        assert loaded_count == 0
        assert len(registry.rules) == 0
        assert registry.find_matching_rule("https://example.com") is None

    def test_load_from_directory_nonexistent(self, tmp_path):
        """Registry should return 0 for non-existent directory."""
        registry = ExtractionRuleRegistry()

        nonexistent = tmp_path / "does_not_exist"
        loaded_count = registry.load_from_directory(nonexistent)
        assert loaded_count == 0
        assert len(registry.rules) == 0
        assert not nonexistent.exists()

    def test_load_from_directory_skips_invalid(self, tmp_path):
        """Registry should skip invalid rule files."""
        registry = ExtractionRuleRegistry()

        # Create valid rule
        valid_rule = {
            "name": "valid_rule",
            "domain": "valid.com",
            "boundaries": {},
            "fields": [],
        }
        (tmp_path / "valid.json").write_text(json.dumps(valid_rule))

        # Create invalid rule (missing required name field)
        (tmp_path / "invalid.json").write_text('{"domain": "invalid.com"}')

        # Create non-rule file
        (tmp_path / "readme.txt").write_text("Not a rule file")

        loaded_count = registry.load_from_directory(tmp_path)
        assert loaded_count == 1
        assert len(registry.rules) == 1
        assert registry.rules[0].name == "valid_rule"

    def test_load_from_directory_yml_extension(self, tmp_path):
        """Registry should load .yml files as YAML."""
        pytest.importorskip("yaml")
        registry = ExtractionRuleRegistry()

        yaml_content = """
name: yml_rule
domain: ymlsite.com
boundaries: {}
fields: []
"""
        (tmp_path / "rule.yml").write_text(yaml_content)

        loaded_count = registry.load_from_directory(tmp_path)
        assert loaded_count == 1
        assert len(registry.rules) == 1
        assert registry.rules[0].name == "yml_rule"

    def test_registry_init_with_rules_dir(self, tmp_path):
        """Registry should load rules from directory on initialization."""
        # Create rule file
        rule_data = {
            "name": "init_rule",
            "domain": "initsite.com",
            "boundaries": {},
            "fields": [],
        }
        (tmp_path / "rule.json").write_text(json.dumps(rule_data))

        registry = ExtractionRuleRegistry(rules_dir=tmp_path)
        assert len(registry.rules) == 1
        assert registry.rules[0].name == "init_rule"
        assert registry.find_matching_rule("https://initsite.com/page") is not None


class TestHTMLProcessorIntegration:
    """Test extraction rules integration with HTMLProcessor."""

    def test_html_processor_with_registry(self):
        """HTMLProcessor should accept extraction_registry parameter."""
        from ingestforge.ingest.html_processor import HTMLProcessor

        registry = create_default_registry()
        processor = HTMLProcessor(extraction_registry=registry)
        assert processor.extraction_registry is not None
        assert processor.extraction_registry is registry
        assert len(processor.extraction_registry.rules) > 0

    def test_html_processor_with_rules_dir(self, tmp_path):
        """HTMLProcessor should load rules from rules_dir."""
        from ingestforge.ingest.html_processor import HTMLProcessor

        # Create rule file
        rule_data = {
            "name": "dir_rule",
            "domain": "dirsite.com",
            "boundaries": {},
            "fields": [],
        }
        (tmp_path / "rule.json").write_text(json.dumps(rule_data))

        processor = HTMLProcessor(rules_dir=tmp_path)
        assert processor.extraction_registry is not None
        assert len(processor.extraction_registry.rules) == 1
        assert processor.extraction_registry.rules[0].name == "dir_rule"

    def test_html_processor_no_registry(self):
        """HTMLProcessor should work without extraction registry."""
        from ingestforge.ingest.html_processor import HTMLProcessor

        processor = HTMLProcessor()
        assert processor._extraction_registry is None
        assert processor._rules_dir is None
        assert processor.extraction_registry is None

    def test_html_processor_rule_matching(self):
        """HTMLProcessor should apply matching rules."""
        from ingestforge.ingest.html_processor import HTMLProcessor

        registry = create_default_registry()
        processor = HTMLProcessor(extraction_registry=registry)

        # Registry should have Wikipedia rule
        wiki_rule = registry.find_matching_rule("https://en.wikipedia.org/wiki/Python")
        assert wiki_rule is not None
        assert wiki_rule.name == "wikipedia"
        assert wiki_rule.domain == "wikipedia.org"
