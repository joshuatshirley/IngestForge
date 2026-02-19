"""
Trust Store Manager for Nexus mTLS.

Manages trusted peer certificate fingerprints.
Rule #2: Fixed bounds on trust store size.
Rule #4: Functions under 60 lines.
Rule #9: Complete type hints.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Set

import yaml
from pydantic import BaseModel, Field

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_TRUSTED_PEERS = 1000
MAX_FINGERPRINT_LENGTH = 128


class TrustedPeer(BaseModel):
    """Metadata for a trusted peer in the trust store."""

    id: str = Field(..., description="Unique peer identifier")
    fingerprint: str = Field(..., description="SHA-256 fingerprint of the certificate")
    name: Optional[str] = Field(None, description="Human-readable peer name")


class TrustStore(BaseModel):
    """Schema for the trust_store.yaml file."""

    version: str = "1.0"
    peers: List[TrustedPeer] = Field(default_factory=list)


class TrustStoreManager:
    """
    Manages the local trust store for mTLS certificate validation.
    """

    def __init__(self, trust_store_path: Path) -> None:
        self.path = trust_store_path
        self._trusted_fingerprints: Set[str] = set()
        self.load()

    def load(self) -> None:
        """
        Load trusted peers from the YAML file.
        Rule #4: Compact loading logic.
        """
        if not self.path.exists():
            logger.warning(f"Trust store not found at {self.path}. Initializing empty.")
            self._trusted_fingerprints = set()
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            store = TrustStore(**data)

            # JPL Rule #2: Bound the number of peers
            peers = store.peers[:MAX_TRUSTED_PEERS]

            self._trusted_fingerprints = {
                p.fingerprint.lower()
                for p in peers
                if len(p.fingerprint) <= MAX_FINGERPRINT_LENGTH
            }

            logger.info(
                f"Loaded {len(self._trusted_fingerprints)} trusted peers from {self.path}"
            )

        except Exception as e:
            logger.error(f"Failed to load trust store: {e}")
            self._trusted_fingerprints = set()

    def is_trusted(self, cert_der: bytes) -> bool:
        """
        Validate a certificate's fingerprint against the trust store.
        Fail-fast: Performance < 5ms guaranteed by hash-map lookup.

        Args:
            cert_der: Certificate in DER format.

        Returns:
            True if the certificate is trusted.
        """
        if not cert_der:
            return False

        fingerprint = hashlib.sha256(cert_der).hexdigest().lower()
        return fingerprint in self._trusted_fingerprints

    def reload(self) -> None:
        """Explicitly trigger a reload of the trust store."""
        self.load()
