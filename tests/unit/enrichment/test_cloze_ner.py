"""Tests for cloze_ner module (NLP-001.1).

Tests the NER-targeted cloze deletion generator:
- ClozeCandidate and Cloze dataclasses
- ClozeGenerator with mock spaCy
- Fallback regex patterns
- Priority-based selection
"""

import pytest
from unittest.mock import Mock, patch

from ingestforge.enrichment.cloze_ner import (
    ClozeCandidate,
    Cloze,
    ClozeResult,
    ClozeType,
    ClozeGenerator,
    generate_clozes,
    DEFAULT_MAX_CLOZES,
    MAX_ALLOWED_CLOZES,
    ENTITY_TYPE_MAP,
    CLOZE_PRIORITIES,
)


class TestClozeType:
    """Test ClozeType enum."""

    def test_all_types_defined(self) -> None:
        """All expected types should be defined."""
        assert ClozeType.DATE
        assert ClozeType.PERSON
        assert ClozeType.LOCATION
        assert ClozeType.ORGANIZATION
        assert ClozeType.EVENT
        assert ClozeType.CONCEPT


class TestEntityTypeMap:
    """Test entity type mapping."""

    def test_date_mapping(self) -> None:
        """DATE should map to ClozeType.DATE."""
        assert ENTITY_TYPE_MAP["DATE"] == ClozeType.DATE

    def test_person_mapping(self) -> None:
        """PERSON should map to ClozeType.PERSON."""
        assert ENTITY_TYPE_MAP["PERSON"] == ClozeType.PERSON

    def test_gpe_mapping(self) -> None:
        """GPE should map to location."""
        assert ENTITY_TYPE_MAP["GPE"] == ClozeType.LOCATION


class TestClozePriorities:
    """Test cloze priorities."""

    def test_date_highest_priority(self) -> None:
        """DATE should have highest priority."""
        assert CLOZE_PRIORITIES[ClozeType.DATE] == 1.0

    def test_all_priorities_positive(self) -> None:
        """All priorities should be positive."""
        for priority in CLOZE_PRIORITIES.values():
            assert priority > 0


class TestClozeCandidate:
    """Test ClozeCandidate dataclass."""

    def test_basic_candidate(self) -> None:
        """Should create basic candidate."""
        candidate = ClozeCandidate(
            text="1066",
            cloze_type=ClozeType.DATE,
            start=10,
            end=14,
        )

        assert candidate.text == "1066"
        assert candidate.cloze_type == ClozeType.DATE
        assert candidate.length == 4

    def test_candidate_hash(self) -> None:
        """Candidates should be hashable."""
        c1 = ClozeCandidate(text="Test", cloze_type=ClozeType.CONCEPT, start=0, end=4)
        c2 = ClozeCandidate(text="test", cloze_type=ClozeType.CONCEPT, start=0, end=4)

        # Same position and text (case insensitive) should hash same
        assert hash(c1) == hash(c2)


class TestCloze:
    """Test Cloze dataclass."""

    def test_basic_cloze(self) -> None:
        """Should create basic cloze."""
        cloze = Cloze(
            original_text="The year was 1066.",
            cloze_text="The year was ___.",
            answer="1066",
            cloze_type=ClozeType.DATE,
        )

        assert cloze.answer == "1066"
        assert "___" in cloze.cloze_text

    def test_cloze_str(self) -> None:
        """String should show cloze and answer."""
        cloze = Cloze(
            original_text="Original",
            cloze_text="The year was ___.",
            answer="1066",
            cloze_type=ClozeType.DATE,
        )

        result = str(cloze)
        assert "___" in result
        assert "1066" in result


class TestClozeResult:
    """Test ClozeResult dataclass."""

    def test_empty_result(self) -> None:
        """Empty result should report zero."""
        result = ClozeResult()

        assert result.count == 0
        assert not result.has_clozes

    def test_with_clozes(self) -> None:
        """Should track cloze count."""
        clozes = [
            Cloze(
                original_text="Test",
                cloze_text="Test",
                answer="A",
                cloze_type=ClozeType.CONCEPT,
            )
        ]

        result = ClozeResult(clozes=clozes)

        assert result.count == 1
        assert result.has_clozes


class TestClozeGeneratorInit:
    """Test ClozeGenerator initialization."""

    def test_default_init(self) -> None:
        """Should initialize with defaults."""
        generator = ClozeGenerator()

        assert generator.max_clozes == DEFAULT_MAX_CLOZES
        assert generator.cloze_marker == "___"

    def test_custom_max_clozes(self) -> None:
        """Should accept custom max_clozes."""
        generator = ClozeGenerator(max_clozes=10)

        assert generator.max_clozes == 10

    def test_max_clozes_exceeds_limit(self) -> None:
        """Should reject max_clozes exceeding Rule #2 limit."""
        with pytest.raises(ValueError, match="cannot exceed"):
            ClozeGenerator(max_clozes=MAX_ALLOWED_CLOZES + 1)

    def test_max_clozes_zero(self) -> None:
        """Should reject zero max_clozes."""
        with pytest.raises(ValueError, match="positive"):
            ClozeGenerator(max_clozes=0)

    def test_custom_marker(self) -> None:
        """Should accept custom cloze marker."""
        generator = ClozeGenerator(cloze_marker="[BLANK]")

        assert generator.cloze_marker == "[BLANK]"


class TestClozeGeneratorWithMockSpacy:
    """Test ClozeGenerator with mocked spaCy."""

    def _create_mock_entity(
        self,
        text: str,
        label: str,
        start: int,
        end: int,
    ) -> Mock:
        """Create mock spaCy entity."""
        ent = Mock()
        ent.text = text
        ent.label_ = label
        ent.start_char = start
        ent.end_char = end
        return ent

    def test_generate_with_spacy(self) -> None:
        """Should generate clozes using spaCy entities."""
        # Create mock spaCy model and doc
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            self._create_mock_entity("1066", "DATE", 10, 14),
            self._create_mock_entity("William", "PERSON", 20, 27),
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp)
        result = generator.generate_clozes("The year was 1066 when William arrived.")

        assert result.has_clozes
        assert result.candidate_count == 2

    def test_target_types_filter(self) -> None:
        """Should filter by target types."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            self._create_mock_entity("1066", "DATE", 10, 14),
            self._create_mock_entity("William", "PERSON", 20, 27),
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp)
        result = generator.generate_clozes(
            "The year was 1066 when William arrived.",
            target_types={ClozeType.DATE},
        )

        # Should only include DATE
        assert result.count == 1
        assert result.clozes[0].cloze_type == ClozeType.DATE

    def test_respects_max_clozes(self) -> None:
        """Should limit to max_clozes."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            self._create_mock_entity(f"Entity{i}", "PERSON", i * 10, i * 10 + 7)
            for i in range(10)
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp, max_clozes=3)
        result = generator.generate_clozes("Many entities in text.")

        assert result.count <= 3


class TestClozeGeneratorFallback:
    """Test fallback when spaCy unavailable."""

    def test_fallback_finds_dates(self) -> None:
        """Fallback should find year patterns."""
        generator = ClozeGenerator()
        generator._spacy_available = False
        generator._spacy_model = None

        # Patch the property to force fallback
        with patch.object(
            ClozeGenerator,
            "is_available",
            new_callable=lambda: property(lambda s: False),
        ):
            result = generator.generate_clozes(
                "The Battle of Hastings occurred in 1066."
            )

        # Should find 1066 as a date
        assert result.candidate_count > 0

    def test_empty_text(self) -> None:
        """Should handle empty text."""
        generator = ClozeGenerator()
        result = generator.generate_clozes("")

        assert not result.has_clozes

    def test_whitespace_text(self) -> None:
        """Should handle whitespace-only text."""
        generator = ClozeGenerator()
        result = generator.generate_clozes("   ")

        assert not result.has_clozes


class TestClozeSelection:
    """Test candidate selection logic."""

    def test_no_overlapping_selections(self) -> None:
        """Should not select overlapping candidates."""
        mock_nlp = Mock()
        mock_doc = Mock()
        # Two overlapping entities
        mock_doc.ents = [
            Mock(text="New York", label_="GPE", start_char=0, end_char=8),
            Mock(text="York", label_="GPE", start_char=4, end_char=8),
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp)
        result = generator.generate_clozes("New York City")

        # Should only select one due to overlap
        assert result.count == 1

    def test_priority_based_selection(self) -> None:
        """Higher priority candidates should be selected first."""
        mock_nlp = Mock()
        mock_doc = Mock()
        # DATE has higher priority than QUANTITY
        mock_doc.ents = [
            Mock(text="100", label_="CARDINAL", start_char=0, end_char=3),
            Mock(text="1066", label_="DATE", start_char=10, end_char=14),
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp, max_clozes=1)
        result = generator.generate_clozes("Has 100 and 1066.")

        # DATE should be selected over CARDINAL
        assert result.clozes[0].cloze_type == ClozeType.DATE


class TestClozeCreation:
    """Test cloze text generation."""

    def test_cloze_text_replacement_anki(self) -> None:
        """Should replace term with Anki format marker."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(text="1066", label_="DATE", start_char=13, end_char=17),
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp, use_anki_format=True)
        result = generator.generate_clozes("The year was 1066.")

        assert result.count == 1
        assert "{{c1::1066}}" in result.clozes[0].cloze_text
        assert result.clozes[0].cloze_text == "The year was {{c1::1066}}."

    def test_cloze_text_replacement_simple(self) -> None:
        """Should replace term with simple marker when Anki disabled."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(text="1066", label_="DATE", start_char=13, end_char=17),
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp, use_anki_format=False)
        result = generator.generate_clozes("The year was 1066.")

        assert result.count == 1
        assert "___" in result.clozes[0].cloze_text
        assert "1066" not in result.clozes[0].cloze_text

    def test_hint_generation(self) -> None:
        """Should generate type-specific hints."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(text="1066", label_="DATE", start_char=0, end_char=4),
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp)
        result = generator.generate_clozes("1066 was the year.")

        assert result.clozes[0].hint is not None
        assert "date" in result.clozes[0].hint.lower()

    def test_multiple_cloze_numbering(self) -> None:
        """Should number clozes sequentially (c1, c2, c3...)."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(text="Napoleon", label_="PERSON", start_char=0, end_char=8),
            Mock(text="1769", label_="DATE", start_char=21, end_char=25),
            Mock(text="Corsica", label_="GPE", start_char=29, end_char=36),
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp, use_anki_format=True)
        result = generator.generate_clozes("Napoleon was born in 1769 in Corsica.")

        assert result.count == 3
        assert "{{c1::Napoleon}}" in result.clozes[0].cloze_text
        assert "{{c2::1769}}" in result.clozes[1].cloze_text
        assert "{{c3::Corsica}}" in result.clozes[2].cloze_text


class TestAnkiClozeGeneration:
    """Test generate_anki_cloze method for multi-cloze cards."""

    def test_generate_single_anki_card(self) -> None:
        """Should generate single text with all clozes numbered."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(text="Napoleon", label_="PERSON", start_char=0, end_char=8),
            Mock(text="1769", label_="DATE", start_char=21, end_char=25),
            Mock(text="Corsica", label_="GPE", start_char=29, end_char=36),
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp)
        result = generator.generate_anki_cloze("Napoleon was born in 1769 in Corsica.")

        assert "{{c1::Napoleon}}" in result
        assert "{{c2::1769}}" in result
        assert "{{c3::Corsica}}" in result
        assert result == "{{c1::Napoleon}} was born in {{c2::1769}} in {{c3::Corsica}}."

    def test_anki_cloze_empty_text(self) -> None:
        """Should return original text when no entities found."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp)
        result = generator.generate_anki_cloze("No entities here.")

        assert result == "No entities here."

    def test_anki_cloze_respects_max_clozes(self) -> None:
        """Should limit to max_clozes."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(
                text=f"Entity{i}",
                label_="PERSON",
                start_char=i * 10,
                end_char=i * 10 + 7,
            )
            for i in range(10)
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp, max_clozes=3)
        result = generator.generate_anki_cloze("Many entities in text.")

        # Should have at most 3 cloze markers
        import re

        cloze_count = len(re.findall(r"\{\{c\d+::", result))
        assert cloze_count <= 3


class TestConvenienceFunction:
    """Test generate_clozes convenience function."""

    def test_generate_clozes_function_returns_strings(self) -> None:
        """Should return list of cloze text strings."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(text="1066", label_="DATE", start_char=9, end_char=13),
        ]
        mock_nlp.return_value = mock_doc

        with patch("ingestforge.enrichment.cloze_ner.ClozeGenerator") as MockGen:
            mock_gen_instance = Mock()
            mock_result = Mock()
            mock_cloze = Mock()
            mock_cloze.cloze_text = "The year {{c1::1066}} saw changes."
            mock_result.clozes = [mock_cloze]
            mock_gen_instance.generate_clozes.return_value = mock_result
            MockGen.return_value = mock_gen_instance

            clozes = generate_clozes(
                "The year 1066 saw major changes.",
                max_clozes=3,
            )

            assert isinstance(clozes, list)
            assert len(clozes) == 1
            assert isinstance(clozes[0], str)


class TestMinTermLength:
    """Test minimum term length filtering."""

    def test_filters_short_terms(self) -> None:
        """Should filter terms shorter than min_term_length."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(text="I", label_="PERSON", start_char=0, end_char=1),  # Too short
            Mock(text="William", label_="PERSON", start_char=5, end_char=12),
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp, min_term_length=2)
        result = generator.generate_clozes("I and William.")

        # Should only include William (length 7 > 2)
        assert result.count == 1
        assert result.clozes[0].answer == "William"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_returns_empty_result(self) -> None:
        """Empty text should return empty ClozeResult."""
        generator = ClozeGenerator()
        result = generator.generate_clozes("")

        assert result.count == 0
        assert not result.has_clozes
        assert result.source_text == ""

    def test_text_with_no_entities(self) -> None:
        """Text with no entities should return empty result."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp)
        result = generator.generate_clozes("This has no entities.")

        assert result.count == 0
        assert result.candidate_count == 0

    def test_max_clozes_limit_enforced(self) -> None:
        """Should respect max_clozes parameter in generate_clozes."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(
                text=f"Name{i}", label_="PERSON", start_char=i * 10, end_char=i * 10 + 5
            )
            for i in range(10)
        ]
        mock_nlp.return_value = mock_doc

        generator = ClozeGenerator(spacy_model=mock_nlp)
        result = generator.generate_clozes("Text with many names.", max_clozes=2)

        assert result.count <= 2
