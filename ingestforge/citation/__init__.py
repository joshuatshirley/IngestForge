"""Citation and bibliography generation.

This package provides citation functionality:
- csl_engine: CSL-based bibliography formatting (CITE-002.2)
"""

from ingestforge.citation.csl_engine import (
    CitationType,
    OutputFormat,
    Author,
    DateParts,
    Reference,
    Citation,
    Bibliography,
    CSLEngine,
    create_engine,
    format_references,
    MAX_REFERENCES,
)

__all__ = [
    # Types
    "CitationType",
    "OutputFormat",
    # Data classes
    "Author",
    "DateParts",
    "Reference",
    "Citation",
    "Bibliography",
    # Engine
    "CSLEngine",
    # Factory functions
    "create_engine",
    "format_references",
    # Constants
    "MAX_REFERENCES",
]
