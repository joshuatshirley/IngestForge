# Technical Spec: FastAPI Engine Bridge (Backend)

## Goal
Implement a modular, high-performance API server that exposes IngestForge core services to the React frontend while maintaining JPL Commandment compliance.

---

## 1. Module Structure (`ingestforge/api/`)
To comply with **Rule #4 (Modularity)**, we will use APIRouters to split the API by domain.

```text
ingestforge/api/
├── __init__.py
├── main.py             # App entry point & middleware
├── dependencies.py     # Global deps (Auth, Config)
├── routes/
│   ├── ingestion.py    # Document processing endpoints
│   ├── retrieval.py    # Search and query endpoints
│   ├── study.py        # SRS and Quiz endpoints
│   └── analysis.py     # Knowledge graph endpoints
└── schemas/            # Pydantic models (Rule #9)
    ├── base.py
    └── response.py
```

---

## 2. Rule #1: Flat Control Flow
End-point handlers must be thin wrappers. All heavy logic must remain in the `core` modules.

```python
@router.post("/process", response_model=PipelineResultSchema)
def process_document(file: UploadFile, config: Config = Depends(get_config)):
    """Rule #7: Parameter validation via Pydantic."""
    pipeline = Pipeline(config)
    return pipeline.process_file(file.path) # Logic stays in Core
```

---

## 3. Real-Time Logic (Rule #2: Bounded Loops)
The WebSocket broadcaster must use a bounded heartbeat to prevent orphaned connections.
*   **Assertion**: `assert active_connections < 100` (Rule #2: prevent resource exhaustion).

---

## 4. Implementation Priority
1.  **TICKET-504**: Base FastAPI scaffold with CORS and Logging.
2.  **TICKET-505**: Ingestion Router with background task support.
3.  **TICKET-506**: Retrieval Router with Hybrid Search integration.
