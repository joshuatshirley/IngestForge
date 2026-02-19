"""Constants for citation processing."""

import re

# Reporter patterns (simplified for core link extraction)
# Matches Volume Reporter Page (e.g., 347 U.S. 483)
LEGAL_REPORTER_PATTERN = re.compile(
    r"\b(\d+)\s+"  # Volume
    r"(U\.S\.|S\.\s*Ct\.|L\.\s*Ed\.\s*(?:2d)?|F\.\s*(?:2d|3d|4th)?|F\.\s*Supp\.\s*(?:2d|3d)?|Cal\.\s*(?:2d|3d|4th)?|N\.Y\.\s*(?:2d|3d)?|A\.\s*(?:2d|3d)?|P\.\s*(?:2d|3d)?|S\.W\.\s*(?:2d|3d)?)\s+"  # Reporter
    r"(\d+)\b"  # Page
)
