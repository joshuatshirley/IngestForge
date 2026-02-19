"""
Verifiable Purge Service.

Verifiable Deletion Cert
Orchestrates data destruction with cryptographic proof and audit logging.

JPL Compliance:
- Rule #4: All functions < 60 lines.
- Rule #7: Strict verification of return values.
- Rule #9: 100% type hints.
"""

import hmac
import hashlib
import os
import uuid
import logging
from typing import Optional, Tuple
from datetime import datetime

from ingestforge.core.audit.models import DeletionCertificate
from ingestforge.core.audit.log import log_operation, AuditOperation
from ingestforge.storage.factory import get_storage_backend
from ingestforge.core.config_loaders import load_config

logger = logging.getLogger(__name__)


class PurgeService:
    """Service for executing and certifying data purges."""

    def __init__(self, secret_key: Optional[str] = None):
        """Initialize with configuration and secret key."""
        self.config = load_config()
        self.repo = get_storage_backend(self.config)
        self.secret_key = secret_key or os.getenv(
            "INGESTFORGE_AUDIT_SECRET", "system_default_purge_secret"
        )

    def execute_verifiable_purge(
        self, chunk_id: str, operator_id: str = "system"
    ) -> Tuple[bool, Optional[DeletionCertificate], str]:
        """
        Executes a purge and returns a signed certificate if successful.

        Rule #4: Main orchestration logic.
        Rule #7: Returns explicit status and certificate.
        """
        # 1. Fetch data before deletion
        chunk = self.repo.get_chunk(chunk_id)
        if not chunk:
            return False, None, f"Chunk {chunk_id} not found in storage"

        # 2. Calculate pre-deletion hash
        content_hash = self._calculate_hash(chunk.content)

        # 3. Perform deletion
        deleted = self.repo.delete_chunk(chunk_id)
        if not deleted:
            return False, None, f"Storage failed to delete chunk {chunk_id}"

        # 4. Verify deletion (JPL Rule #7)
        verified = self._verify_deletion(chunk_id)
        if not verified:
            return False, None, f"Deletion verification failed for {chunk_id}"

        # 5. Generate and sign certificate
        cert = self._create_certificate(chunk_id, content_hash, operator_id, verified)

        # 6. Log to audit log ()
        # JPL Rule #7: Explicitly verify audit log entry creation
        entry = log_operation(
            operation=AuditOperation.DELETE,
            source=operator_id,
            target=chunk_id,
            message=f"Verifiable purge completed. Certificate: {cert.certificate_id}",
            metadata={
                "certificate_id": cert.certificate_id,
                "content_hash": content_hash,
            },
        )

        if not entry:
            logger.error(
                f"Critical: Purge verified but audit logging failed for {chunk_id}"
            )
            return False, None, "Purge verified but audit logging failed"

        return True, cert, "Purge successful and verified"

    def _calculate_hash(self, content: str) -> str:
        """Helper to calculate SHA-256 hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _verify_deletion(self, chunk_id: str) -> bool:
        """Double-check that the chunk is truly gone."""
        return self.repo.get_chunk(chunk_id) is None

    def _create_certificate(
        self, artifact_id: str, content_hash: str, operator_id: str, verified: bool
    ) -> DeletionCertificate:
        """Constructs and signs a deletion certificate."""
        cert_id = f"CERT-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.utcnow()

        # Create canonical string for signing
        data_to_sign = f"{cert_id}|{artifact_id}|{timestamp.isoformat()}|{content_hash}|{operator_id}"
        signature = self._sign_data(data_to_sign)

        return DeletionCertificate(
            certificate_id=cert_id,
            artifact_id=artifact_id,
            purge_timestamp=timestamp,
            content_hash=content_hash,
            operator_id=operator_id,
            storage_verification=verified,
            signature=signature,
        )

    def _sign_data(self, data: str) -> str:
        """Generates HMAC-SHA256 signature."""
        return hmac.new(
            self.secret_key.encode("utf-8"), data.encode("utf-8"), hashlib.sha256
        ).hexdigest()
