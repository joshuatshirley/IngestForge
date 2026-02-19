# Technical Spec: Backend-to-Frontend API Bridge

## Goal
Provide a high-performance, typed REST API using FastAPI to bridge the IngestForge core engine to the React frontend.

---

## 1. Core Endpoints (JPL Rule #7: Strict Input Validation)

### Ingestion API
*   `POST /v1/ingest/file`: Accept multipart/form-data.
    *   **Logic**: Run `Pipeline.process_file` in a background task.
*   `GET /v1/ingest/status/{document_id}`: Returns real-time state from `StateManager`.

### Retrieval API
*   `GET /v1/search`: Query params: `q` (string), `top_k` (int), `library` (string).
    *   **Logic**: Calls `HybridRetriever.search`.
    *   **Validation**: Assert `top_k > 0` and `q` length < 1000.

### Study API
*   `GET /v1/study/due`: Returns count and list of cards due today.
*   `POST /v1/study/rate`: Accept `card_id` and `quality` (0-5).
    *   **Logic**: Update SRS state via `SM2Algorithm`.

---

## 2. Real-Time Updates (WebSockets)
*   `WS /v1/events`: Broadcast pipeline progress events (e.g., "DOC_EXTRACTED", "EMBEDDING_COMPLETE").
*   **Safety**: Max 10 messages per second per client to prevent browser flood.

---

## 3. Boundary Schema (Rule #9: Pydantic Integration)
All API responses must be typed using Pydantic models that mirror our internal Dataclasses.

```python
from pydantic import BaseModel

class SearchResultSchema(BaseModel):
    chunk_id: str
    content: str
    score: float
    source_file: str
    page_start: Optional[int]
```

---

## 4. Security (Rule #6: Smallest Scope)
*   **CORS**: Only allow `http://localhost:3000` (or configured frontend URL).
*   **Session**: Use JWT tokens for auth-protected endpoints.
