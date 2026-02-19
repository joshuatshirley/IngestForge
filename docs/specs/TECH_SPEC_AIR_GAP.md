# Technical Spec: Air-Gap Network Interceptor (SEC-002)

## Goal
Provide a "Hardened Local" mode that physically prevents the Python process from making external network calls, ensuring total privacy for sensitive research.

---

## 1. The Socket Interceptor (Rule #8: Clean Abstraction)
We will use a context manager to "wrap" the standard library socket creation.

**Logic**:
1.  Override `socket.socket`.
2.  On `connect`, check the target `(host, port)`.
3.  **Whitelist**: Only allow `localhost`, `127.0.0.1`, and optionally configured local model endpoints (e.g., `ollama:11434`).
4.  **Enforcement**: If host is external, raise `ingestforge.core.exceptions.SecurityError`.

---

## 2. Global Enforcement (Rule #10)
**Module**: `ingestforge/core/security/network_lock.py`

```python
class NetworkLock:
    def __enter__(self):
        if config.security.air_gap_mode:
            self._patch_sockets
            
    def _patch_sockets(self):
        # Implementation of socket override
        pass
```

---

## 3. Whitelist Configuration
**Path**: `ingestforge.yaml`
```yaml
security:
  air_gap_mode: true
  network_whitelist:
    - "localhost"
    - "127.0.0.1"
    - "ollama.local"
```

---

## 4. Verification Assertion (Rule #5)
The test suite MUST include a test that attempts to fetch `google.com` while `air_gap_mode` is enabled.
*   `with pytest.raises(SecurityError): requests.get("https://google.com")`
*   This ensures the interceptor cannot be bypassed by high-level libraries (requests, httpx, urllib).
