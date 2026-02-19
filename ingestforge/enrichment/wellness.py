"""
Nutrition and Diet enrichment.

Extracts calories, macros, and allergens from meal plans and labels.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class WellnessMetadataRefiner:
    """
    Enriches chunks with wellness-specific metadata.
    """

    # Wellness specific patterns
    CALORIE_PATTERN = re.compile(
        r"\b(\d{1,4})\s*(?:kcal|calories|cal)\b", re.IGNORECASE
    )
    MACRO_PATTERN = re.compile(
        r"\b(\d{1,3}(?:\.\d)?)\s*g\s*(Protein|Fat|Carbs|Carbohydrates)\b|\b(Protein|Fat|Carbs|Carbohydrates)[:\s]*(\d{1,3}(?:\.\d)?)\s*g\b",
        re.IGNORECASE,
    )
    ALLERGEN_KEYWORDS = [
        "Gluten",
        "Dairy",
        "Nuts",
        "Peanuts",
        "Soy",
        "Shellfish",
        "Eggs",
    ]

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with wellness metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Calories
        cal_match = self.CALORIE_PATTERN.search(content)
        if cal_match:
            metadata["wellness_calories"] = float(cal_match.group(1))

        # Extract Macros
        macros = {}
        for macro_match in self.MACRO_PATTERN.finditer(content):
            # Check which group matched (1&2 or 3&4)
            if macro_match.group(1):
                val, name = (
                    float(macro_match.group(1)),
                    macro_match.group(2).capitalize(),
                )
            else:
                name, val = (
                    macro_match.group(3).capitalize(),
                    float(macro_match.group(4)),
                )

            if name == "Carbohydrates":
                name = "Carbs"
            macros[name] = val
        if macros:
            metadata["wellness_macros"] = macros

        # Extract Allergens (Keyword based)
        found_allergens = []
        for allergen in self.ALLERGEN_KEYWORDS:
            if re.search(rf"\b{allergen}\b", content, re.IGNORECASE):
                found_allergens.append(allergen)
        if found_allergens:
            metadata["wellness_allergens"] = list(set(found_allergens))

        chunk.metadata = metadata
        return chunk
