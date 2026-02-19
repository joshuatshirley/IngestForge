"""
Security Processors Package.

Contains security-related processors for PII redaction and data protection.
"""

from ingestforge.processors.security.redaction import IFRedactionProcessor

__all__ = ["IFRedactionProcessor"]
