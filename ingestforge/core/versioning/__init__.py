"""Versioning Module.

Automated Semantic Versioning
Epic: EP-26 (Security & Compliance)
"""

from ingestforge.core.versioning.version_manager import (
    VersionManager,
    SemanticVersion,
    BumpType,
    create_version_manager,
    parse_version,
    MAX_COMMITS_TO_ANALYZE,
)

__all__ = [
    "VersionManager",
    "SemanticVersion",
    "BumpType",
    "create_version_manager",
    "parse_version",
    "MAX_COMMITS_TO_ANALYZE",
]
