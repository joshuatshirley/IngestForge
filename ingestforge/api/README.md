# Integration Bridge API Usage

## Starting the API Server

```python
from ingestforge.api.bridge import run_server

# Start server on default port (8000)
run_server

# Or specify custom host/port
run_server(host="127.0.0.1", port=8080)
```

Or via command line:
```bash
# Using uvicorn directly
uvicorn ingestforge.api.bridge:app --host 0.0.0.0 --port 8000

# With auto-reload for development
uvicorn ingestforge.api.bridge:app --reload
```

## API Endpoints

### POST /v1/ingest/bridge
Ingest text content directly into the knowledge base.

**Request:**
```json
{
  "text": "Your text content here...",
  "source": "vscode",
  "title": "Optional title",
  "metadata": {
    "library": "research",
    "tags": ["important"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "chunk_count": 3,
  "chunk_ids": ["chunk_1", "chunk_2", "chunk_3"],
  "document_id": "text_vscode_abc123",
  "message": "Successfully ingested 3 chunks",
  "processing_time_ms": 245.3
}
```

**Validation:**
- `text`: Required, 1-1,000,000 characters, non-whitespace
- `source`: Optional (default: "ide"), max 256 characters
- `title`: Optional, max 512 characters
- `metadata`: Optional dictionary

### POST /v1/query
Query the knowledge base.

**Request:**
```json
{
  "query": "machine learning algorithms",
  "top_k": 5,
  "library": "research"
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "content": "Machine learning is...",
      "source": "ml_textbook.pdf",
      "score": 0.95,
      "chunk_id": "chunk_abc_1"
    }
  ],
  "query_time_ms": 42.1
}
```

**Validation:**
- `query`: Required, 1-10,000 characters
- `top_k`: Optional (default: 5), range 1-100
- `library`: Optional filter by library name

### GET /v1/status
Get system status and statistics.

**Response:**
```json
{
  "status": "ok",
  "version": "1.2.0",
  "document_count": 42,
  "chunk_count": 256,
  "timestamp": "2026-02-14T12:34:56.789"
}
```

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Ingest text
response = requests.post(
    f"{BASE_URL}/v1/ingest/bridge",
    json={
        "text": "Neural networks are powerful machine learning models...",
        "source": "vscode",
        "title": "ML Notes",
        "metadata": {"library": "research"}
    }
)
result = response.json
print(f"Created {result['chunk_count']} chunks")

# Query knowledge base
response = requests.post(
    f"{BASE_URL}/v1/query",
    json={
        "query": "neural networks",
        "top_k": 5
    }
)
results = response.json
for r in results["results"]:
    print(f"Score: {r['score']:.2f} - {r['content'][:100]}")

# Check status
response = requests.get(f"{BASE_URL}/v1/status")
status = response.json
print(f"System has {status['chunk_count']} chunks indexed")
```

## TypeScript/JavaScript Client Example

```typescript
const BASE_URL = "http://localhost:8000";

// Ingest text
const ingestResponse = await fetch(`${BASE_URL}/v1/ingest/bridge`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "Deep learning revolutionized AI...",
    source: "vscode",
    title: "AI Research Notes"
  })
});
const result = await ingestResponse.json;
console.log(`Created ${result.chunk_count} chunks`);

// Query
const queryResponse = await fetch(`${BASE_URL}/v1/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "deep learning",
    top_k: 5
  })
});
const results = await queryResponse.json;
results.results.forEach(r => {
  console.log(`${r.score}: ${r.content.substring(0, 100)}`);
});
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "success": false,
  "message": "Text cannot be empty or whitespace-only",
  "processing_time_ms": 1.2
}
```

Common HTTP status codes:
- `200`: Success
- `422`: Validation error (invalid parameters)
- `500`: Internal server error

## 
This API follows the code quality guidelines:

1. **Rule #4**: All endpoint handlers <60 lines
2. **Rule #7**: Input validation via Pydantic models with fixed bounds
   - Text payloads: 1-1,000,000 chars
   - Query strings: 1-10,000 chars
   - top_k: 1-100 results
3. **Rule #9**: Complete type hints in all functions
4. **Rule #1**: Simple control flow with early returns
5. **Rule #6**: Lazy imports to avoid slow startup

## Integration with IDEs

### VS Code Extension
```typescript
// VS Code extension example
async function sendToIngestForge(text: string) {
  const response = await fetch("http://localhost:8000/v1/ingest/bridge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      source: "vscode",
      metadata: {
        workspace: vscode.workspace.name,
        file: vscode.window.activeTextEditor?.document.fileName
      }
    })
  });
  return response.json;
}
```

### Obsidian Plugin
```typescript
// Obsidian plugin example
async function syncNoteToIngestForge(note: TFile) {
  const content = await this.app.vault.read(note);
  const response = await fetch("http://localhost:8000/v1/ingest/bridge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: content,
      source: "obsidian",
      title: note.basename,
      metadata: {
        path: note.path,
        tags: this.app.metadataCache.getFileCache(note)?.tags
      }
    })
  });
  return response.json;
}
```
