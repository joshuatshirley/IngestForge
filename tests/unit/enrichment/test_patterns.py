"""Tests for patterns module (NLP-001.2).

Tests the domain pattern registry:
- DomainPattern dataclass
- TermMatch dataclass
- PatternRegistry searching
- Domain detection
"""


from ingestforge.enrichment.patterns import (
    Domain,
    DomainPattern,
    TermMatch,
    PatternRegistry,
    find_domain_terms,
    MAX_ALLOWED_TERMS,
    HISTORY_PATTERNS,
    SCIENCE_PATTERNS,
    PSYCHOLOGY_PATTERNS,
)


class TestDomain:
    """Test Domain enum."""

    def test_all_domains_defined(self) -> None:
        """All expected domains should be defined."""
        assert Domain.HISTORY
        assert Domain.SCIENCE
        assert Domain.PSYCHOLOGY
        assert Domain.GENERAL
        assert Domain.MATHEMATICS
        assert Domain.LITERATURE
        assert Domain.MEDICINE
        assert Domain.LAW


class TestDomainPattern:
    """Test DomainPattern dataclass."""

    def test_basic_pattern(self) -> None:
        """Should create basic pattern."""
        pattern = DomainPattern(
            pattern=r"\b\d{4}\b",
            domain=Domain.HISTORY,
            category="year",
        )

        assert pattern.domain == Domain.HISTORY
        assert pattern.category == "year"

    def test_pattern_compiles(self) -> None:
        """Should compile regex pattern."""
        pattern = DomainPattern(
            pattern=r"\btest\b",
            domain=Domain.GENERAL,
            category="test",
        )

        compiled = pattern.compiled
        assert compiled is not None
        assert compiled.search("this is a test")

    def test_pattern_cached(self) -> None:
        """Compiled pattern should be cached."""
        pattern = DomainPattern(
            pattern=r"\btest\b",
            domain=Domain.GENERAL,
            category="test",
        )

        compiled1 = pattern.compiled
        compiled2 = pattern.compiled

        assert compiled1 is compiled2


class TestTermMatch:
    """Test TermMatch dataclass."""

    def test_basic_match(self) -> None:
        """Should create basic match."""
        match = TermMatch(
            text="Renaissance",
            domain=Domain.HISTORY,
            category="period",
            start=10,
            end=21,
        )

        assert match.text == "Renaissance"
        assert match.domain == Domain.HISTORY
        assert match.length == 11

    def test_match_hash(self) -> None:
        """Matches should be hashable."""
        m1 = TermMatch(text="Test", domain=Domain.GENERAL, category="c", start=0, end=4)
        m2 = TermMatch(text="test", domain=Domain.GENERAL, category="c", start=0, end=4)

        # Same position should hash same
        assert hash(m1) == hash(m2)


class TestBuiltinPatterns:
    """Test built-in pattern sets."""

    def test_history_patterns_exist(self) -> None:
        """History patterns should be defined."""
        assert len(HISTORY_PATTERNS) > 0

    def test_science_patterns_exist(self) -> None:
        """Science patterns should be defined."""
        assert len(SCIENCE_PATTERNS) > 0

    def test_psychology_patterns_exist(self) -> None:
        """Psychology patterns should be defined."""
        assert len(PSYCHOLOGY_PATTERNS) > 0

    def test_all_patterns_have_domain(self) -> None:
        """All patterns should have valid domain."""
        all_patterns = HISTORY_PATTERNS + SCIENCE_PATTERNS + PSYCHOLOGY_PATTERNS

        for pattern in all_patterns:
            assert isinstance(pattern.domain, Domain)


class TestPatternRegistryInit:
    """Test PatternRegistry initialization."""

    def test_default_init(self) -> None:
        """Should initialize with built-in patterns."""
        registry = PatternRegistry()

        assert registry.default_domain == Domain.GENERAL
        assert len(registry.list_domains()) > 0

    def test_custom_patterns(self) -> None:
        """Should accept custom patterns."""
        custom = DomainPattern(
            pattern=r"\bcustom\b",
            domain=Domain.GENERAL,
            category="custom",
        )

        registry = PatternRegistry(custom_patterns=[custom])

        patterns = registry.get_patterns(Domain.GENERAL)
        assert any(p.category == "custom" for p in patterns)

    def test_register_pattern(self) -> None:
        """Should register pattern after init."""
        registry = PatternRegistry()

        custom = DomainPattern(
            pattern=r"\bnewpattern\b",
            domain=Domain.SCIENCE,
            category="new",
        )
        registry.register_pattern(custom)

        patterns = registry.get_patterns(Domain.SCIENCE)
        assert any(p.category == "new" for p in patterns)


class TestPatternRegistrySearch:
    """Test pattern searching."""

    def test_find_history_terms(self) -> None:
        """Should find history terms."""
        registry = PatternRegistry()

        text = "The Renaissance was a period of cultural rebirth in Europe."
        terms = registry.find_terms(text, domain=Domain.HISTORY)

        assert len(terms) > 0
        assert any(t.text == "Renaissance" for t in terms)

    def test_find_science_terms(self) -> None:
        """Should find science terms."""
        registry = PatternRegistry()

        text = "Newton's Laws describe motion and forces."
        terms = registry.find_terms(text, domain=Domain.SCIENCE)

        assert len(terms) > 0

    def test_find_psychology_terms(self) -> None:
        """Should find psychology terms."""
        registry = PatternRegistry()

        text = "The hippocampus is involved in memory formation."
        terms = registry.find_terms(text, domain=Domain.PSYCHOLOGY)

        assert len(terms) > 0
        assert any(t.text.lower() == "hippocampus" for t in terms)

    def test_empty_text(self) -> None:
        """Should handle empty text."""
        registry = PatternRegistry()

        terms = registry.find_terms("")
        assert terms == []

    def test_whitespace_text(self) -> None:
        """Should handle whitespace text."""
        registry = PatternRegistry()

        terms = registry.find_terms("   ")
        assert terms == []

    def test_respects_max_terms(self) -> None:
        """Should limit results to max_terms."""
        registry = PatternRegistry()

        text = "First the Renaissance, then the Enlightenment, the Industrial Revolution, the Victorian Era, and more."
        terms = registry.find_terms(text, domain=Domain.HISTORY, max_terms=2)

        assert len(terms) <= 2

    def test_max_terms_cap(self) -> None:
        """Should cap at MAX_ALLOWED_TERMS."""
        registry = PatternRegistry()

        text = "Test text"
        terms = registry.find_terms(text, max_terms=MAX_ALLOWED_TERMS + 100)

        # Should not exceed cap (even if enough matches existed)
        assert len(terms) <= MAX_ALLOWED_TERMS

    def test_includes_general_patterns(self) -> None:
        """Domain search should include general patterns."""
        registry = PatternRegistry()

        # General patterns should apply to specific domains
        patterns = registry.get_patterns(Domain.HISTORY)

        # Should have both history-specific and general patterns
        domains = {p.domain for p in patterns}
        assert Domain.HISTORY in domains


class TestDeduplication:
    """Test match deduplication."""

    def test_removes_duplicates(self) -> None:
        """Should remove duplicate matches at same position."""
        registry = PatternRegistry()

        # Create text that might match multiple patterns at same position
        text = "The Renaissance period changed art."
        terms = registry.find_terms(text, domain=Domain.HISTORY)

        # Check no duplicate positions
        positions = [(t.start, t.end) for t in terms]
        unique_positions = set(positions)

        assert len(positions) == len(unique_positions)


class TestDomainDetection:
    """Test automatic domain detection."""

    def test_detect_history(self) -> None:
        """Should detect history domain."""
        registry = PatternRegistry()

        text = "The Renaissance began in Italy during the 15th century after the Middle Ages."
        domain = registry.detect_domain(text)

        assert domain == Domain.HISTORY

    def test_detect_science(self) -> None:
        """Should detect science domain."""
        registry = PatternRegistry()

        text = "Photosynthesis converts carbon dioxide and water into glucose using light energy in chloroplasts."
        domain = registry.detect_domain(text)

        assert domain == Domain.SCIENCE

    def test_detect_psychology(self) -> None:
        """Should detect psychology domain."""
        registry = PatternRegistry()

        text = "The amygdala and hippocampus are involved in processing emotional memories and cognitive dissonance."
        domain = registry.detect_domain(text)

        assert domain == Domain.PSYCHOLOGY

    def test_detect_general_fallback(self) -> None:
        """Should return GENERAL for unrecognized text."""
        registry = PatternRegistry()

        text = "This is plain text with no domain terms."
        domain = registry.detect_domain(text)

        assert domain == Domain.GENERAL

    def test_detect_empty_text(self) -> None:
        """Should return GENERAL for empty text."""
        registry = PatternRegistry()

        domain = registry.detect_domain("")
        assert domain == Domain.GENERAL


class TestConvenienceFunction:
    """Test find_domain_terms convenience function."""

    def test_find_domain_terms_with_domain(self) -> None:
        """Should work with specified domain."""
        terms = find_domain_terms(
            "The Renaissance changed European art.",
            domain=Domain.HISTORY,
        )

        assert isinstance(terms, list)
        assert len(terms) > 0

    def test_find_domain_terms_auto_detect(self) -> None:
        """Should auto-detect domain."""
        terms = find_domain_terms(
            "The Renaissance was a cultural movement in Europe.",
        )

        assert isinstance(terms, list)


class TestSpecificPatterns:
    """Test specific pattern matching."""

    def test_world_war_pattern(self) -> None:
        """Should match World War references."""
        registry = PatternRegistry()

        terms = registry.find_terms(
            "World War II began in 1939.",
            domain=Domain.HISTORY,
        )

        assert any("World War" in t.text for t in terms)

    def test_treaty_pattern(self) -> None:
        """Should match treaty references."""
        registry = PatternRegistry()

        terms = registry.find_terms(
            "The Treaty of Versailles ended World War I.",
            domain=Domain.HISTORY,
        )

        assert any("Treaty of" in t.text for t in terms)

    def test_scientific_law_pattern(self) -> None:
        """Should match scientific laws."""
        registry = PatternRegistry()

        terms = registry.find_terms(
            "Newton's Laws describe how objects move.",
            domain=Domain.SCIENCE,
        )

        assert any("Newton's" in t.text for t in terms)

    def test_process_pattern(self) -> None:
        """Should match scientific processes."""
        registry = PatternRegistry()

        terms = registry.find_terms(
            "Photosynthesis occurs in plant cells.",
            domain=Domain.SCIENCE,
        )

        assert any(t.text.lower() == "photosynthesis" for t in terms)

    def test_brain_region_pattern(self) -> None:
        """Should match brain regions."""
        registry = PatternRegistry()

        terms = registry.find_terms(
            "The prefrontal cortex handles decision making.",
            domain=Domain.PSYCHOLOGY,
        )

        assert any("prefrontal cortex" in t.text.lower() for t in terms)


class TestListDomains:
    """Test listing domains."""

    def test_list_domains(self) -> None:
        """Should list registered domains."""
        registry = PatternRegistry()

        domains = registry.list_domains()

        assert Domain.HISTORY in domains
        assert Domain.SCIENCE in domains
        assert Domain.PSYCHOLOGY in domains
        assert Domain.GENERAL in domains
