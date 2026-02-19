import pytest
from ingestforge.ingest.real_estate_extractor import PropertyExtractor


class TestPropertyExtractor:
    @pytest.fixture
    def extractor(self):
        return PropertyExtractor()

    def test_extract_listing_data(self, extractor, tmp_path):
        content = """
        FOR SALE: 123 Maple Street, Springfield
        Price: $450,000
        Size: 2,500 sq ft
        Beautiful 4-bedroom home with large backyard.
        """
        listing_file = tmp_path / "listing.txt"
        listing_file.write_text(content)

        result = extractor.extract(listing_file)
        metadata = result["metadata"]

        assert metadata["property_price"] == 450000.0
        assert metadata["property_sqft"] == 2500.0
        assert metadata["property_address"] == "123 MAPLE ST"
