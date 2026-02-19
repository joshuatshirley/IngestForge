"""
Real estate property metadata extraction.

Extracts addresses, prices, and physical specs from property listings.
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ingestforge.ingest.text_extractor import TextExtractor
from ingestforge.core.config import Config
from ingestforge.shared.address_utils import normalize_address

logger = logging.getLogger(__name__)


class PropertyExtractor:
    """
    Extractor for real estate documents (listings, deeds, zoning).
    """

    # Patterns for property data
    PRICE_PATTERN = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
    SQFT_PATTERN = re.compile(
        r"(\d{1,3}(?:,\d{3})*)\s*(?:sqft|sq ft|square feet)", re.IGNORECASE
    )

    # Simple address heuristic (Number + Street Name + Suffix)
    ADDRESS_HEURISTIC = re.compile(
        r"\b\d+\s+[A-Z0-9\s]+(?:ST|STREET|AVE|AVENUE|BLVD|BOULEVARD|DR|DRIVE|RD|ROAD|LN|LANE|CT|COURT|PL|PLACE)\b",
        re.IGNORECASE,
    )

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.text_extractor = TextExtractor(self.config)

    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract property metadata from a file."""
        text = self.text_extractor.extract(file_path)

        metadata = {
            "property_price": self._extract_price(text),
            "property_sqft": self._extract_sqft(text),
            "property_address": self._extract_address(text),
        }

        return {"text": text, "metadata": metadata}

    def _extract_price(self, text: str) -> float:
        match = self.PRICE_PATTERN.search(text)
        if match:
            # Remove commas and convert to float
            return float(match.group(1).replace(",", ""))
        return 0.0

    def _extract_sqft(self, text: str) -> float:
        match = self.SQFT_PATTERN.search(text)
        if match:
            return float(match.group(1).replace(",", ""))
        return 0.0

    def _extract_address(self, text: str) -> str:
        match = self.ADDRESS_HEURISTIC.search(text)
        if match:
            return normalize_address(match.group(0))
        return ""
