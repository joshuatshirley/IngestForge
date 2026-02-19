"""Export module for document generation and corpus sharing.

Provides outline mapping, document formatting, and portable corpus packaging.
"""

from ingestforge.core.export.outline_mapper import (
    OutlineMapper,
    OutlinePoint,
    EvidenceMatch,
    MappedOutline,
)
from ingestforge.core.export.packager import (
    CorpusPackager,
    PackageManifest,
    PackageResult,
)
from ingestforge.core.export.importer import (
    CorpusImporter,
    ImportResult,
    ValidationResult,
    get_package_info,
)

__all__ = [
    # Outline mapping
    "OutlineMapper",
    "OutlinePoint",
    "EvidenceMatch",
    "MappedOutline",
    # Corpus packaging
    "CorpusPackager",
    "PackageManifest",
    "PackageResult",
    # Corpus importing
    "CorpusImporter",
    "ImportResult",
    "ValidationResult",
    "get_package_info",
]
