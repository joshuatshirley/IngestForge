"""
Resume and HR document metadata extraction.

Extracts contact info, skills, and education from resumes.
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.ingest.text_extractor import TextExtractor
from ingestforge.core.config import Config

logger = logging.getLogger(__name__)


class ResumeMetadataExtractor:
    """
    Specialized extractor for HR documents like resumes and job descriptions.
    """

    # Common HR patterns
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_PATTERN = re.compile(
        r"\(?\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )

    # Common skills keywords for initial discovery
    SKILLS_KEYWORDS = [
        "Python",
        "Java",
        "Javascript",
        "C++",
        "C#",
        "React",
        "Angular",
        "Vue",
        "SQL",
        "NoSQL",
        "Docker",
        "Kubernetes",
        "AWS",
        "Azure",
        "GCP",
        "Machine Learning",
        "AI",
        "Data Science",
        "Project Management",
        "Agile",
        "Scrum",
        "Git",
        "CI/CD",
        "DevOps",
    ]

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.text_extractor = TextExtractor(self.config)

    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract HR-specific metadata from a resume file."""
        text = self.text_extractor.extract(file_path)

        metadata = {
            "resume_email": self._extract_email(text),
            "resume_phone": self._extract_phone(text),
            "resume_skills": self._extract_skills(text),
            "resume_education": self._extract_education(text),
        }

        return {"text": text, "metadata": metadata}

    def _extract_email(self, text: str) -> str:
        match = self.EMAIL_PATTERN.search(text)
        return match.group(0) if match else ""

    def _extract_phone(self, text: str) -> str:
        match = self.PHONE_PATTERN.search(text)
        return match.group(0) if match else ""

    def _extract_skills(self, text: str) -> List[str]:
        found_skills = []
        for skill in self.SKILLS_KEYWORDS:
            if re.search(r"\b" + re.escape(skill) + r"\b", text, re.IGNORECASE):
                found_skills.append(skill)
        return found_skills

    def _extract_education(self, text: str) -> List[str]:
        # Simple heuristic for education extraction
        education_keywords = [
            "Bachelor",
            "Master",
            "PhD",
            "University",
            "College",
            "B.S.",
            "M.S.",
            "B.A.",
        ]
        lines = text.split("\n")
        found_edu = []
        for line in lines:
            if any(keyword in line for keyword in education_keywords):
                found_edu.append(line.strip())
        return found_edu[:5]  # Limit to top 5 hits
