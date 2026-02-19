"""
mTLS Policy Enforcement for Nexus.

Enforces TLS 1.3 and client certificate validation.
Rule #4: Functions under 60 lines.
Rule #7: Explicit return checking for certificate extraction.
Rule #9: Complete type hints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from ingestforge.core.config_loaders import load_config
from ingestforge.core.network.trust_store import TrustStoreManager

logger = logging.getLogger(__name__)

# Global trust store instance
_trust_store: Optional[TrustStoreManager] = None


def get_trust_store() -> TrustStoreManager:
    """Get or initialize the global TrustStoreManager."""
    global _trust_store
    if _trust_store is None:
        config = load_config()
        trust_path = Path(config._base_path) / config.api.trust_store_path
        _trust_store = TrustStoreManager(trust_path)
    return _trust_store


async def verify_mtls_peer(
    request: Request, trust_store: TrustStoreManager = Depends(get_trust_store)
) -> str:
    """
    FastAPI dependency to verify mTLS client certificates.

    AC: Fails fast if query to invalid cert is received.
    Rule #7: Check request scope for peer certificate.
    """
    # Extract client certificate from ASGI scope
    # Note: requires proxy (like Nginx) or Uvicorn to populate 'client_cert'
    client_cert = request.scope.get("extensions", {}).get("tls", {}).get("client_cert")

    if not client_cert:
        # Fallback for some Uvicorn configurations/direct connections
        client_cert = request.scope.get("client_cert")

    if not client_cert:
        logger.warning("mTLS verification failed: No client certificate provided")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="mTLS authentication required"
        )

    # Validate against trust store
    # JPL Rule #7: Explicit check
    if not trust_store.is_trusted(client_cert):
        logger.error("mTLS verification failed: Peer certificate not in trust store")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Untrusted peer certificate"
        )

    return "verified_peer"


def enforce_tls_version(ssl_context: "ssl.SSLContext") -> None:
    """
    Configure SSL context to enforce TLS 1.3.
    Rule #4: Compact policy application.
    """
    import ssl

    # Disable older protocols
    ssl_context.options |= ssl.OP_NO_SSLv2
    ssl_context.options |= ssl.OP_NO_SSLv3
    ssl_context.options |= ssl.OP_NO_TLSv1
    ssl_context.options |= ssl.OP_NO_TLSv1_1
    ssl_context.options |= ssl.OP_NO_TLSv1_2

    # Require client certificate
    ssl_context.verify_mode = ssl.CERT_REQUIRED

    logger.info("TLS 1.3 policy enforced with mandatory client certificates")
