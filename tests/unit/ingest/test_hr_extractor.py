import pytest
from ingestforge.ingest.hr_extractor import ResumeMetadataExtractor


class TestResumeMetadataExtractor:
    @pytest.fixture
    def extractor(self):
        return ResumeMetadataExtractor()

    def test_extract_resume_metadata(self, extractor, tmp_path):
        content = """
        John Doe
        Email: john.doe@example.com
        Phone: (555) 123-4567
        
        Education:
        Master of Science in Computer Science, Stanford University
        
        Skills: Python, React, Docker, Kubernetes
        
        Experience:
        Worked on various AI projects using Python and TensorFlow.
        """
        resume_file = tmp_path / "resume.txt"
        resume_file.write_text(content)

        result = extractor.extract(resume_file)
        metadata = result["metadata"]

        assert metadata["resume_email"] == "john.doe@example.com"
        assert metadata["resume_phone"] == "(555) 123-4567"
        assert "Python" in metadata["resume_skills"]
        assert "React" in metadata["resume_skills"]
        assert "Stanford University" in metadata["resume_education"][0]
