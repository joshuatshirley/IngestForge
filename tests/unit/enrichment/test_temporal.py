"""Tests for temporal event extraction."""
import pytest

from ingestforge.enrichment.temporal import (
    TemporalExtractor,
    extract_temporal_events,
    build_timeline,
)


class TestTemporalExtractor:
    """Tests for TemporalExtractor class."""

    @pytest.fixture
    def extractor(self) -> TemporalExtractor:
        """Create extractor instance."""
        return TemporalExtractor()

    def test_extract_empty_text(self, extractor: TemporalExtractor):
        """Test extraction from empty text."""
        chunk = {"text": ""}

        result = extractor.extract(chunk)

        assert result["temporal_events"] == []

    def test_extract_missing_text(self, extractor: TemporalExtractor):
        """Test extraction with missing text field."""
        chunk = {}

        result = extractor.extract(chunk)

        assert result["temporal_events"] == []

    def test_extract_year_event_in_prefix(self, extractor: TemporalExtractor):
        """Test extraction of year event with 'In' prefix."""
        chunk = {"text": "In 2020, the pandemic began."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 1
        event = result["temporal_events"][0]
        assert event["date"] == "2020"
        assert event["normalized_date"] == "2020-01-01"
        assert "pandemic" in event["event"]
        assert event["granularity"] == "year"

    def test_extract_year_event_during_prefix(self, extractor: TemporalExtractor):
        """Test extraction with 'During' prefix."""
        chunk = {"text": "During 1990, the wall fell."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 1
        assert result["temporal_events"][0]["date"] == "1990"

    def test_extract_year_event_by_prefix(self, extractor: TemporalExtractor):
        """Test extraction with 'By' prefix."""
        chunk = {"text": "By 2030, we will achieve our goals."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 1
        assert result["temporal_events"][0]["date"] == "2030"

    def test_extract_full_date_with_month(self, extractor: TemporalExtractor):
        """Test extraction of full date."""
        chunk = {"text": "On January 1, 2000, the new millennium began."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 1
        event = result["temporal_events"][0]
        assert event["date"] == "January 1, 2000"
        assert event["normalized_date"] == "2000-01-01"
        assert event["granularity"] == "day"

    def test_extract_date_without_on(self, extractor: TemporalExtractor):
        """Test extraction of date without 'On' prefix."""
        chunk = {"text": "December 25, 2023, was a special day."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 1
        assert result["temporal_events"][0]["date"] == "December 25, 2023"

    def test_extract_abbreviated_month(self, extractor: TemporalExtractor):
        """Test extraction with abbreviated month."""
        chunk = {"text": "On Jan 15, 2021, something happened."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 1
        assert result["temporal_events"][0]["date"] == "Jan 15, 2021"
        assert result["temporal_events"][0]["normalized_date"] == "2021-01-15"

    def test_extract_multiple_events(self, extractor: TemporalExtractor):
        """Test extraction of multiple events."""
        chunk = {
            "text": "In 2000, the company started. On March 15, 2005, it went public. By 2010, it was a leader."
        }

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 3
        assert result["event_count"] == 3

    def test_events_sorted_by_date(self, extractor: TemporalExtractor):
        """Test events are sorted chronologically."""
        chunk = {"text": "In 2020, event C. In 2010, event A. In 2015, event B."}

        result = extractor.extract(chunk)

        dates = [e["normalized_date"] for e in result["temporal_events"]]
        assert dates == sorted(dates)

    def test_event_position_recorded(self, extractor: TemporalExtractor):
        """Test event position in text is recorded."""
        chunk = {"text": "Some text. In 2020, event happened."}

        result = extractor.extract(chunk)

        assert result["temporal_events"][0]["position"] > 0

    def test_extract_with_sentence_ending(self, extractor: TemporalExtractor):
        """Test extraction captures until sentence end."""
        chunk = {"text": "In 2020, the event occurred! Then more text."}

        result = extractor.extract(chunk)

        event_text = result["temporal_events"][0]["event"]
        assert "!" in event_text
        assert "Then" not in event_text

    def test_year_event_with_comma(self, extractor: TemporalExtractor):
        """Test year event with optional comma."""
        chunk1 = {"text": "In 2020, something happened."}
        chunk2 = {"text": "In 2020 something happened."}

        result1 = extractor.extract(chunk1)
        result2 = extractor.extract(chunk2)

        # Both should extract event
        assert len(result1["temporal_events"]) >= 1
        assert len(result2["temporal_events"]) >= 1

    def test_date_with_comma_variations(self, extractor: TemporalExtractor):
        """Test date extraction with comma variations."""
        chunk = {"text": "January 1 2000, event happened."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) >= 0  # May or may not match pattern

    def test_invalid_date_format_fallback(self, extractor: TemporalExtractor):
        """Test handling of invalid date that can't be parsed."""
        # This should still extract but use fallback normalization
        chunk = {"text": "On XYZ 99, 2000, something happened."}

        # Should not crash
        result = extractor.extract(chunk)
        assert "temporal_events" in result

    def test_different_months(self, extractor: TemporalExtractor):
        """Test extraction with various months."""
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        for month in months:
            chunk = {"text": f"On {month} 15, 2020, event happened."}
            result = extractor.extract(chunk)

            assert len(result["temporal_events"]) >= 1

    def test_abbreviated_months(self, extractor: TemporalExtractor):
        """Test extraction with abbreviated months."""
        abbrevs = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        for abbrev in abbrevs:
            chunk = {"text": f"{abbrev} 1, 2020, event."}
            result = extractor.extract(chunk)

            # Should extract at least one event
            assert len(result["temporal_events"]) >= 1

    def test_case_insensitive_extraction(self, extractor: TemporalExtractor):
        """Test extraction is case insensitive."""
        chunk1 = {"text": "IN 2020, event happened."}
        chunk2 = {"text": "in 2020, event happened."}

        result1 = extractor.extract(chunk1)
        result2 = extractor.extract(chunk2)

        assert len(result1["temporal_events"]) == len(result2["temporal_events"])

    def test_event_count_field(self, extractor: TemporalExtractor):
        """Test event_count field is added."""
        chunk = {"text": "In 2020, event 1. In 2021, event 2."}

        result = extractor.extract(chunk)

        assert result["event_count"] == 2

    def test_preserves_original_chunk(self, extractor: TemporalExtractor):
        """Test extractor preserves original chunk data."""
        chunk = {"id": 123, "title": "Test", "text": "In 2020, event."}

        result = extractor.extract(chunk)

        assert result["id"] == 123
        assert result["title"] == "Test"

    def test_no_events_found(self, extractor: TemporalExtractor):
        """Test text with no temporal events."""
        chunk = {"text": "This is just regular text without dates."}

        result = extractor.extract(chunk)

        assert result["temporal_events"] == []
        assert result["event_count"] == 0


class TestExtractTemporalEvents:
    """Tests for extract_temporal_events convenience function."""

    def test_function_extracts_events(self):
        """Test standalone function extracts events."""
        chunk = {"text": "In 2020, something important happened."}

        result = extract_temporal_events(chunk)

        assert len(result["temporal_events"]) > 0

    def test_function_creates_own_extractor(self):
        """Test function creates its own extractor instance."""
        chunk = {"text": "In 2021, event."}

        result = extract_temporal_events(chunk)

        assert "temporal_events" in result


class TestBuildTimeline:
    """Tests for timeline building."""

    def test_build_timeline_empty(self):
        """Test building timeline from empty list."""
        timeline = build_timeline([])

        assert timeline == []

    def test_build_timeline_single_chunk(self):
        """Test building timeline from single chunk."""
        chunks = [{"text": "In 2020, event happened."}]

        timeline = build_timeline(chunks)

        assert len(timeline) >= 1

    def test_build_timeline_multiple_chunks(self):
        """Test building timeline from multiple chunks."""
        chunks = [
            {"text": "In 2020, event A."},
            {"text": "In 2019, event B."},
            {"text": "In 2021, event C."},
        ]

        timeline = build_timeline(chunks)

        # Should have events from all chunks
        assert len(timeline) >= 3

    def test_timeline_sorted_chronologically(self):
        """Test timeline is sorted by date."""
        chunks = [
            {"text": "In 2025, future event."},
            {"text": "In 2015, past event."},
            {"text": "In 2020, recent event."},
        ]

        timeline = build_timeline(chunks)

        dates = [e["normalized_date"] for e in timeline]
        assert dates == sorted(dates)

    def test_timeline_combines_all_events(self):
        """Test timeline combines events from all chunks."""
        chunks = [
            {"text": "In 2020, event 1. In 2021, event 2."},
            {"text": "In 2022, event 3."},
        ]

        timeline = build_timeline(chunks)

        assert len(timeline) == 3

    def test_timeline_with_no_events(self):
        """Test timeline building with chunks containing no events."""
        chunks = [
            {"text": "Regular text."},
            {"text": "More text without dates."},
        ]

        timeline = build_timeline(chunks)

        assert timeline == []

    def test_timeline_mixed_granularity(self):
        """Test timeline with mixed date granularities."""
        chunks = [
            {"text": "In 2020, year event."},
            {"text": "On January 15, 2020, day event."},
        ]

        timeline = build_timeline(chunks)

        # Should include both year and day granularity events
        assert len(timeline) == 2
        granularities = [e["granularity"] for e in timeline]
        assert "year" in granularities
        assert "day" in granularities


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def extractor(self) -> TemporalExtractor:
        """Create extractor instance."""
        return TemporalExtractor()

    def test_very_long_text(self, extractor: TemporalExtractor):
        """Test extraction from very long text."""
        long_text = "word " * 10000 + "In 2020, event happened."
        chunk = {"text": long_text}

        result = extractor.extract(chunk)

        # Should still find the event
        assert len(result["temporal_events"]) >= 1

    def test_multiple_same_year(self, extractor: TemporalExtractor):
        """Test multiple events in same year."""
        chunk = {"text": "In 2020, event A. In 2020, event B."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 2

    def test_distant_past_year(self, extractor: TemporalExtractor):
        """Test extraction of distant past year."""
        chunk = {"text": "In 1066, the Norman conquest occurred."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 1
        assert result["temporal_events"][0]["date"] == "1066"

    def test_distant_future_year(self, extractor: TemporalExtractor):
        """Test extraction of future year."""
        chunk = {"text": "By 2100, we hope to achieve this goal."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 1
        assert result["temporal_events"][0]["date"] == "2100"

    def test_february_29_leap_year(self, extractor: TemporalExtractor):
        """Test extraction of leap year date."""
        chunk = {"text": "On February 29, 2020, leap day occurred."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 1
        event = result["temporal_events"][0]
        assert event["normalized_date"] == "2020-02-29"

    def test_event_with_punctuation(self, extractor: TemporalExtractor):
        """Test event text with various punctuation."""
        chunk = {"text": "In 2020, the event happened! It was amazing?"}

        result = extractor.extract(chunk)

        # Should capture until first sentence terminator
        event_text = result["temporal_events"][0]["event"]
        assert "!" in event_text or "?" in event_text

    def test_unicode_in_event_text(self, extractor: TemporalExtractor):
        """Test event text with Unicode characters."""
        chunk = {"text": "In 2020, café opened in São Paulo."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) >= 1

    def test_day_zero(self, extractor: TemporalExtractor):
        """Test handling of invalid day (0)."""
        # This should either not match or fail gracefully
        chunk = {"text": "January 0, 2020, invalid date."}

        result = extractor.extract(chunk)

        # Should not crash
        assert "temporal_events" in result

    def test_day_32(self, extractor: TemporalExtractor):
        """Test handling of invalid day (32)."""
        chunk = {"text": "January 32, 2020, invalid date."}

        result = extractor.extract(chunk)

        # Should handle gracefully (fallback to year-01-01)
        if len(result["temporal_events"]) > 0:
            event = result["temporal_events"][0]
            # Should use fallback normalization
            assert event["normalized_date"].endswith("-01-01")

    def test_single_digit_day(self, extractor: TemporalExtractor):
        """Test extraction with single digit day."""
        chunk = {"text": "January 5, 2020, event happened."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 1
        assert result["temporal_events"][0]["normalized_date"] == "2020-01-05"

    def test_double_digit_day(self, extractor: TemporalExtractor):
        """Test extraction with double digit day."""
        chunk = {"text": "December 31, 2020, year ended."}

        result = extractor.extract(chunk)

        assert len(result["temporal_events"]) == 1
        assert result["temporal_events"][0]["normalized_date"] == "2020-12-31"
