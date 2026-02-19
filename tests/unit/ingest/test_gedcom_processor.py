import pytest
from ingestforge.ingest.gedcom_processor import GedcomProcessor


class TestGedcomProcessor:
    @pytest.fixture
    def processor(self):
        return GedcomProcessor()

    def test_parse_gedcom_individuals(self, processor, tmp_path):
        gedcom_content = """0 HEAD
1 CHAR UTF-8
0 @I1@ INDI
1 NAME John /Doe/
1 SEX M
1 BIRT
2 DATE 1 JAN 1900
1 DEAT
2 DATE 1 JAN 1980
0 @I2@ INDI
1 NAME Jane /Smith/
1 SEX F
0 TRLR
"""
        ged_file = tmp_path / "test.ged"
        ged_file.write_text(gedcom_content)

        records = processor.process(ged_file)

        assert len(records) == 2

        # Check John Doe
        john = next(r for r in records if "John" in r["name"])
        assert john["birth_date"] == "1 JAN 1900"
        assert john["death_date"] == "1 JAN 1980"
        assert john["sex"] == "M"

        # Check Jane Smith
        jane = next(r for r in records if "Jane" in r["name"])
        assert jane["sex"] == "F"
