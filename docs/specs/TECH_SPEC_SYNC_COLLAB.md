# Technical Spec: Multi-User Sync & Collaboration

## Goal
Enable real-time synchronization of annotations, tags, and project state across multiple users using a shared PostgreSQL backend and a lightweight WebSocket protocol.

---

## 1. Shared Storage Schema (PostgreSQL)
We will extend the `ChunkRepository` to support a relational backend.

**Key Fields for Collaboration**:
*   `user_id`: UUID of the author.
*   `project_id`: UUID of the shared workspace.
*   `version`: Atomic counter for conflict resolution (Last-Write-Wins).

---

## 2. The Sync Protocol (Event-Driven)
Avoid persistent WebSocket sessions where possible. Use a "Push-on-Change, Poll-on-Resume" strategy.

**Sync Events**:
```json
{
  "event": "CHUNK_ANNOTATED",
  "data": {
    "chunk_id": "uuid",
    "user_id": "uuid",
    "note": "New insight here",
    "timestamp": "iso8601"
  }
}
```

---

## 3. Conflict Resolution (Rule #1: Simple Flow)
Do NOT implement complex Operational Transformation (OT).
1.  **Client A** sends update with `version=5`.
2.  **Server** checks if current `version` is 5.
3.  If yes, apply and increment to 6.
4.  If no, reject and force **Client A** to pull latest.

---

## 4. Offline Persistence (Rule #3: Memory Management)
**Module**: `ingestforge/core/sync/queue.py`
*   Unsent changes must be stored in a local SQLite table (`sync_queue`).
*   **Assertion**: `assert queue.size < 1000` (Rule #2).
*   On reconnection, stream the queue using a generator to avoid memory spikes.
