"""
Autonomous Domain Router.

Analyzes text content to determine the most likely vertical/domain
(e.g., 'Legal', 'Cyber', 'Urban') based on weighted heuristic signals.
"""

import re
import logging
from typing import List, Tuple, Optional
from collections import defaultdict

# Import all specialized refiners to access their patterns if needed,
# or define representative signals here.
# For performance and decoupling, we will define lightweight "Signature" patterns here.

logger = logging.getLogger(__name__)


class DomainRouter:
    """
    Routes content to specific domains based on signal density.
    """

    # Domain Signatures: (Domain Key, Regex Pattern, Weight)
    # Weight 5 = Strong unique identifier (e.g., "CVE-2023-1234")
    # Weight 2 = Strong keyword (e.g., "Plaintiff")
    # Weight 1 = Weak context (e.g., "court")

    SIGNATURES = [
        # 1. Legal
        (
            "legal",
            re.compile(
                r"\b(?:Plaintiff|Defendant|Certiorari|Habeas Corpus|v\.|Cir\.)\b",
                re.IGNORECASE,
            ),
            2,
        ),
        (
            "legal",
            re.compile(r"\b\d{1,4}\s+[A-Z][\w\.]+\.?\s+\d{1,4}\b"),
            4,
        ),  # Citation-like: 345 U.S. 123
        # 2. Medical
        (
            "medical",
            re.compile(
                r"\b(?:Diagnosis|Prognosis|Patient|Rx|ICD-10|Symptom)\b", re.IGNORECASE
            ),
            2,
        ),
        # 3. Technical / Code
        (
            "technical",
            re.compile(r"\b(?:function|class|def|return|import|var|const|void)\b"),
            2,
        ),
        ("technical", re.compile(r"[{};=<>]\s*\n"), 1),
        # 4. Financial
        (
            "financial",
            re.compile(
                r"\b(?:EBITDA|Q[1-4]|FY\d{2,4}|Revenue|Profit|Margin|Balance Sheet)\b",
                re.IGNORECASE,
            ),
            3,
        ),
        # 15. Cyber
        ("cyber", re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE), 5),
        (
            "cyber",
            re.compile(
                r"\b(?:Malware|Phishing|XSS|SQLi|Attack Vector|Payload)\b",
                re.IGNORECASE,
            ),
            3,
        ),
        # 17. Wellness
        (
            "wellness",
            re.compile(
                r"\b(?:Calories|Protein|Carbs|Fat|Serving Size|Vitamin)\b",
                re.IGNORECASE,
            ),
            3,
        ),
        # 18. Spiritual
        (
            "spiritual",
            re.compile(
                r"\b(?:Chapter|Verse|Sutra|Surah|Bible|Quran|Veda|Testament)\b",
                re.IGNORECASE,
            ),
            3,
        ),
        # 19. Museum
        (
            "museum",
            re.compile(
                r"\b(?:Artist|Medium|Dimensions|Provenance|Exhibition|Curator)\b",
                re.IGNORECASE,
            ),
            2,
        ),
        (
            "museum",
            re.compile(
                r"\b(?:Oil on Canvas|Bronze|Sculpture|Gallery)\b", re.IGNORECASE
            ),
            3,
        ),
        # 20. Bio/Lab
        (
            "bio",
            re.compile(
                r"\b(?:Experiment|Lab Ref|Sample ID|Reagent|Protocol|Centrifuge)\b",
                re.IGNORECASE,
            ),
            3,
        ),
        # 21. Auto
        (
            "auto",
            re.compile(
                r"\b(?:VIN|Chassis|Odometer|Transmission|Engine|Torque|Horsepower)\b",
                re.IGNORECASE,
            ),
            2,
        ),
        ("auto", re.compile(r"\bP/N[:\s-]+\w+\b", re.IGNORECASE), 3),
        # 22. Gaming
        (
            "gaming",
            re.compile(
                r"\b(?:Buff|Nerf|Patch|DPS|Tank|Healer|Cooldown|Mana|Hitpoints)\b",
                re.IGNORECASE,
            ),
            3,
        ),
        # 23. AI Safety
        (
            "ai_safety",
            re.compile(
                r"\b(?:Model|LLM|Transformer|Parameters|Trillion|Billion|Benchmark)\b",
                re.IGNORECASE,
            ),
            2,
        ),
        ("ai_safety", re.compile(r"\b(?:MMLU|GSM8K|TruthfulQA|HumanEval)\b"), 4),
        # 24. Urban
        (
            "urban",
            re.compile(
                r"\b(?:Zoning|FAR|Floor Area Ratio|Density|Ordinance|Setback)\b",
                re.IGNORECASE,
            ),
            3,
        ),
        (
            "urban",
            re.compile(r"\b[RCIM]-\d+(?:-[A-Z])?\b"),
            4,
        ),  # Zoning codes like R-1, C-2
    ]

    def classify_chunk(self, text: str) -> List[Tuple[str, int]]:
        """
        Analyzes text and returns a ranked list of (domain, score) tuples.
        """
        scores = defaultdict(int)

        for domain, pattern, weight in self.SIGNATURES:
            matches = len(pattern.findall(text))
            if matches > 0:
                scores[domain] += matches * weight

        # Sort by score desc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def get_best_domain(self, text: str, threshold: int = 3) -> Optional[str]:
        """
        Returns the single best domain if score exceeds threshold.
        """
        ranked = self.classify_chunk(text)
        if not ranked:
            return None

        best_domain, score = ranked[0]
        if score >= threshold:
            return best_domain
        return None
