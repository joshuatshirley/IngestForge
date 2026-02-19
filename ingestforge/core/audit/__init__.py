"""Audit module.

Append-Only Audit Log
LLM Model Parity Auditor
BUG002: Dependency Integrity Audit

Provides immutable operation tracking, LLM compliance auditing,
and dependency integrity verification.
"""

from ingestforge.core.audit.log import (
    AuditLogEntry,
    AuditOperation,
    AppendOnlyAuditLog,
    AuditQueryResult,
    create_audit_log,
    log_operation,
    MAX_AUDIT_ENTRIES,
    MAX_QUERY_RESULTS,
)

from ingestforge.core.audit.model_auditor import (
    ModelAuditor,
    ModelCapability,
    AuditStatus,
    ProviderStatus,
    AuditReport,
    create_model_auditor,
    PROVIDER_CAPABILITIES,
    MAX_PROVIDERS,
    MAX_CAPABILITIES,
)

from ingestforge.core.audit.dependency_auditor import (
    DependencyAuditor,
    DependencyReport,
    DependencyIssue,
    create_dependency_auditor,
    MAX_FILES_TO_SCAN,
    MAX_IMPORTS_PER_FILE,
)

__all__ = [
    # Audit Log ()
    "AuditLogEntry",
    "AuditOperation",
    "AppendOnlyAuditLog",
    "AuditQueryResult",
    "create_audit_log",
    "log_operation",
    "MAX_AUDIT_ENTRIES",
    "MAX_QUERY_RESULTS",
    # Model Auditor ()
    "ModelAuditor",
    "ModelCapability",
    "AuditStatus",
    "ProviderStatus",
    "AuditReport",
    "create_model_auditor",
    "PROVIDER_CAPABILITIES",
    "MAX_PROVIDERS",
    "MAX_CAPABILITIES",
    # Dependency Auditor (BUG002)
    "DependencyAuditor",
    "DependencyReport",
    "DependencyIssue",
    "create_dependency_auditor",
    "MAX_FILES_TO_SCAN",
    "MAX_IMPORTS_PER_FILE",
]
