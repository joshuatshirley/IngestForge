"""
mTLS Security Manager for Workspace Nexus.

Task 119: Implement mTLS peer-to-peer encryption.
NASA JPL Power of Ten: Rule #4 (Small functions), Rule #9 (Type hints).
"""

import ssl
import logging
from typing import Optional
from ingestforge.core.config.nexus import NexusConfig

logger = logging.getLogger(__name__)


class MTLSContextManager:
    """
    Manages SSL/TLS context for mutual authentication.
    """

    def __init__(self, config: NexusConfig):
        self.config = config
        self._server_context: Optional[ssl.SSLContext] = None
        self._client_context: Optional[ssl.SSLContext] = None

    def get_server_context(self) -> ssl.SSLContext:
        """Create or return SSL context for incoming requests."""
        if not self._server_context:
            self._server_context = self._create_context(server_side=True)
        return self._server_context

    def get_client_context(self) -> ssl.SSLContext:
        """Create or return SSL context for outbound queries."""
        if not self._client_context:
            self._client_context = self._create_context(server_side=False)
        return self._client_context

    def _create_context(self, server_side: bool) -> ssl.SSLContext:
        """
        Setup TLS 1.3 context with mutual auth.
        Rule #4: Atomic logic < 60 lines.
        """
        purpose = ssl.Purpose.CLIENT_AUTH if server_side else ssl.Purpose.SERVER_AUTH
        context = ssl.create_default_context(purpose)

        # Enforce TLS 1.3 (JPL Compliance)
        context.minimum_version = ssl.TLSVersion.TLSv1_3

        # Load local identity
        self._load_local_identity(context)

        # Load trust store for mutual auth
        self._load_trust_store(context)

        if server_side:
            context.verify_mode = ssl.CERT_REQUIRED

        return context

    def _load_local_identity(self, context: ssl.SSLContext) -> None:
        """Load cert and key from config."""
        if not self.config.cert_file.exists() or not self.config.key_file.exists():
            raise FileNotFoundError("mTLS identity files missing")
        context.load_cert_chain(
            certfile=str(self.config.cert_file), keyfile=str(self.config.key_file)
        )

    def _load_trust_store(self, context: ssl.SSLContext) -> None:
        """Load trusted peer certificates."""
        if not self.config.trust_store.exists():
            raise FileNotFoundError(f"Trust store not found: {self.config.trust_store}")

        if self.config.trust_store.is_dir():
            context.load_verify_locations(capath=str(self.config.trust_store))
        else:
            context.load_verify_locations(cafile=str(self.config.trust_store))
