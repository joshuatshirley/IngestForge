import pytest
from ingestforge.ingest.citation_metadata import LegalMetadataExtractor, SourceType


class TestLegalMetadataExtractor:
    @pytest.fixture
    def extractor(self):
        return LegalMetadataExtractor()

    def test_extract_from_json_courtlistener(self, extractor):
        """Test extraction from CourtListener-like JSON data."""
        sample_data = {
            "case_name": "Roe v. Wade",
            "docket_number": "70-18",
            "court": "Supreme Court of the United States",
            "judge": "Blackmun",
            "date_filed": "1973-01-22",
            "citation": "410 U.S. 113",
            "jurisdiction": "Federal",
        }

        meta = extractor.extract_from_json(sample_data)

        assert meta.title == "Roe v. Wade"
        assert meta.docket_number == "70-18"
        assert meta.court == "Supreme Court of the United States"
        assert meta.judge == "Blackmun"
        assert meta.year == 1973
        assert meta.citation == "410 U.S. 113"
        assert meta.source_type == SourceType.COURT_OPINION
        assert meta.confidence >= 0.7

    def test_extract_from_text_identifiers(self, extractor):
        """Test extraction of legal identifiers from plain text."""
        text = """
        IN THE UNITED STATES DISTRICT COURT FOR THE NORTHERN DISTRICT OF CALIFORNIA
        
        APPLE INC., Plaintiff,
        v.
        SAMSUNG ELECTRONICS CO., LTD., Defendants.
        
        Case No. 11-cv-01846-LHK
        
        ORDER GRANTING MOTION
        
        Judge: Lucy H. Koh
        """

        meta = extractor.extract_from_text(text)

        assert meta.docket_number == "11-cv-01846-LHK"
        assert "district court" in meta.jurisdiction.lower()
        assert meta.judge == "Lucy H. Koh"
        assert meta.source_type == SourceType.COURT_OPINION

    def test_extract_bluebook_citation(self, extractor):
        """Test extraction of Bluebook citations from text."""
        text = "This principle was established in Brown v. Board of Education, 347 U.S. 483 (1954)."

        meta = extractor.extract_from_text(text)

        assert meta.citation == "347 U.S. 483"
        assert meta.reporter == "U.S."
        assert meta.year == 1954
