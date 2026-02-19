"""
Address normalization and validation utilities.
"""

import re


def normalize_address(address: str) -> str:
    """
    Standardize an address string.

    Example: "123 Main St., Apt 4" -> "123 MAIN ST APT 4"
    """
    if not address:
        return ""

    # Uppercase
    addr = address.upper().strip()

    # Remove punctuation
    addr = re.sub(r"[.,]", "", addr)

    # Standardize abbreviations
    replacements = {
        r"\bSTREET\b": "ST",
        r"\bAVENUE\b": "AVE",
        r"\bBOULEVARD\b": "BLVD",
        r"\bDRIVE\b": "DR",
        r"\bROAD\b": "RD",
        r"\bLANE\b": "LN",
        r"\bAPARTMENT\b": "APT",
        r"\bSUITE\b": "STE",
        r"\bNORTH\b": "N",
        r"\bSOUTH\b": "S",
        r"\bEAST\b": "E",
        r"\bWEST\b": "W",
    }

    for pattern, replacement in replacements.items():
        addr = re.sub(pattern, replacement, addr)

    # Collapse multiple spaces
    addr = re.sub(r"\s+", " ", addr)

    return addr.strip()
