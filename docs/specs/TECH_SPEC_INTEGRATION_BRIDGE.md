# Technical Spec: Integration Bridge (VS Code, Obsidian)

## Goal
Establish a secure, stateless API bridge that allows external IDEs and note-taking apps to ingest and query content without direct access to the `core` internals.

---

## 1. The "Safe Bridge" Pattern (Rule #8)
External tools MUST NOT import `ingestforge` modules directly. They must communicate via the Local API (FastAPI).

**Core Endpoints**:
1.  `POST /v1/ingest/selection`: Accept raw text + source URL.
2.  `GET /v1/query`: Standard search with structured JSON return.
3.  `POST /v1/bookmarks`: Add metadata to existing chunks.

---

## 2. API Security & Validation (Rule #7)
*   **Token Auth**: Every request must include an `X-Forge-Key` (configured via `auth-wizard`).
*   **Path Sanitization**: `POST /ingest` must assert that URLs are valid and not pointing to internal network IP ranges (Rule #7: SSRF prevention).

---

## 3. Extension Architecture
*   **VS Code**: Lightweight TS wrapper that calls the FastAPI endpoints.
*   **Obsidian**: Plugin that syncs the `vault` contents via the Bridge.

---

## 4. JPL Rule #1 Compliance
The Bridge handler must follow a flat "Receive -> Validate -> Dispatch -> Respond" flow.
*   No long-running stateful sessions in the API.
*   Timeout after 30s for any ingestion request.
