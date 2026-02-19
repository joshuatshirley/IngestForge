"""
Audit Models for Verifiable Purge.

Verifiable Deletion Cert
Defines the structure for signed proof of data destruction.

JPL Compliance:
- Rule #9: 100% type hints.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class DeletionCertificate(BaseModel):
    """
    Cryptographically signed proof that an artifact has been purged.

    Rule #9: Complete type hints.
    """

    certificate_id: str = Field(..., description="Unique ID for this certificate")
    artifact_id: str = Field(..., description="ID of the purged artifact/document")
    purge_timestamp: datetime = Field(default_factory=datetime.utcnow)
    content_hash: str = Field(
        ..., description="SHA-256 hash of the content prior to deletion"
    )
    operator_id: str = Field(
        ..., description="ID of the user/system that triggered the purge"
    )
    storage_verification: bool = Field(
        False, description="Whether storage confirmed 0 records remaining"
    )

    # Cryptographic signature of the above fields
    signature: str = Field(
        ..., description="HMAC-SHA256 signature of the certificate data"
    )

    def to_verification_string(self) -> str:
        """Generates a canonical string for signature verification."""
        # Use ISO format for timestamp to ensure consistency
        ts = self.purge_timestamp.isoformat()
        return f"{self.certificate_id}|{self.artifact_id}|{ts}|{self.content_hash}|{self.operator_id}"
