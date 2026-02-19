# IngestForge REST API Reference

## Overview

The IngestForge REST API provides programmatic access to all ingestion and query features.

**Base URL:** `http://localhost:8000` (local development)

**Authentication:** API key via `X-API-Key` header (optional, set via `INGESTFORGE_API_KEY` env var)

**Rate Limiting:** Not yet implemented

> **Note:** This document describes both implemented and planned API features.
> Endpoints marked with **(Planned)** are not yet available. Response schemas
> shown are target specifications — actual responses may differ. See the
> auto-generated docs at `http://localhost:8000/docs` for the live API.

**Content Type:** `application/json`

**Note:** For programmatic access, consider using the Python SDK (when available) instead of direct REST calls for better type safety and error handling.

---

## Quick Start

### Start the API Server

```bash
# Start with default settings
ingestforge serve

# Custom port and host
ingestforge serve --host 0.0.0.0 --port 8080

# With API key authentication enabled
INGESTFORGE_REQUIRE_AUTH=true ingestforge serve
```

### Test Connection

```bash
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "0.1.0", "storage_backend": "chromadb"}
```

### Basic Usage Example

```bash
# 1. Ingest a document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "enable_enrichment": true
  }'

# 2. Query the knowledge base
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "top_k": 5
  }'
```

---

## Endpoints

### Health Check

**GET `/health`**

Check API server health and status.

**Parameters:** None

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "storage_backend": "chromadb",
  "total_documents": 15,
  "total_chunks": 1247,
  "uptime_seconds": 3600
}
```

**Status Codes:**
- `200 OK` - Server is healthy and ready
- `503 Service Unavailable` - Server is unhealthy or not ready

**Example:**
```bash
curl http://localhost:8000/health
```

---

### Ingest Document

**POST `/ingest`**

Ingest a document and add it to the knowledge base.

**Request Body:**
```json
{
  "file_path": "/path/to/document.pdf",
  "document_id": "optional-custom-id",
  "chunking_strategy": "semantic",
  "enable_enrichment": true,
  "enrichers": ["entities", "questions", "embeddings"],
  "metadata": {
    "author": "Smith",
    "year": 2023,
    "tags": ["quantum", "computing"]
  }
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file_path` | string | Yes | - | Absolute path to document |
| `document_id` | string | No | auto-generated | Custom document identifier |
| `chunking_strategy` | string | No | `"semantic"` | `semantic`, `fixed`, `legal`, or `code` |
| `enable_enrichment` | boolean | No | `true` | Enable chunk enrichment |
| `enrichers` | array | No | all available | List of enrichers to apply |
| `metadata` | object | No | `{}` | Custom metadata to attach |

**Response:**
```json
{
  "document_id": "doc_abc123",
  "status": "completed",
  "chunks_created": 42,
  "processing_time_seconds": 12.5,
  "storage_backend": "chromadb",
  "enrichment_applied": ["entities", "embeddings"],
  "file_name": "document.pdf",
  "file_size_bytes": 524288,
  "ingested_at": "2026-02-02T12:00:00Z"
}
```

**Status Codes:**
- `201 Created` - Document ingested successfully
- `400 Bad Request` - Invalid file path or parameters
- `404 Not Found` - File not found
- `409 Conflict` - Document with same ID already exists
- `500 Internal Server Error` - Processing failed

**Example:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "./documents/research_paper.pdf",
    "enable_enrichment": true,
    "metadata": {
      "author": "Jane Smith",
      "year": 2023,
      "category": "research"
    }
  }'
```

---

### Query Knowledge Base

**POST `/query`**

Query the knowledge base and get an answer with citations.

**Request Body:**
```json
{
  "query": "What are the main findings?",
  "top_k": 5,
  "retrieval_strategy": "hybrid",
  "semantic_weight": 0.5,
  "bm25_weight": 0.5,
  "include_citations": true,
  "citation_format": "apa",
  "enable_llm_synthesis": false,
  "filters": {
    "document_id": "doc_abc123",
    "author": "Smith",
    "year": 2023
  }
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Natural language query |
| `top_k` | integer | No | `5` | Number of results to return (1-100) |
| `retrieval_strategy` | string | No | `"hybrid"` | `semantic`, `bm25`, or `hybrid` |
| `semantic_weight` | float | No | `0.5` | Semantic search weight (0.0-1.0) |
| `bm25_weight` | float | No | `0.5` | BM25 search weight (0.0-1.0) |
| `include_citations` | boolean | No | `true` | Include source citations |
| `citation_format` | string | No | `"short"` | `short`, `apa`, `mla`, or `chicago` |
| `enable_llm_synthesis` | boolean | No | `false` | Generate LLM answer |
| `filters` | object | No | `{}` | Metadata filters |

**Response:**
```json
{
  "query": "What are the main findings?",
  "results": [
    {
      "chunk_id": "chunk_123",
      "content": "The main findings indicate that quantum computers can achieve exponential speedup for certain algorithms...",
      "score": 0.92,
      "document_id": "doc_abc",
      "source_file": "paper.pdf",
      "page_number": 3,
      "section_title": "Results",
      "citation": "[Smith 2023, p.3]",
      "metadata": {
        "author": "Smith",
        "year": 2023
      }
    }
  ],
  "total_results": 47,
  "returned_results": 5,
  "processing_time_seconds": 0.82,
  "retrieval_strategy": "hybrid",
  "answer": "Based on the research, quantum computers can achieve...",
  "citations": [
    "Smith, J. (2023). Quantum Computing Advances (p. 3). MIT Press."
  ]
}
```

**Status Codes:**
- `200 OK` - Query successful
- `400 Bad Request` - Invalid query parameters
- `404 Not Found` - No results found
- `500 Internal Server Error` - Query processing failed

**Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is semantic chunking?",
    "top_k": 3,
    "include_citations": true
  }'
```

---

### List Documents

**GET `/documents`**

List all documents in the knowledge base.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `offset` | integer | No | `0` | Pagination offset |
| `limit` | integer | No | `50` | Results per page (max: 100) |
| `sort_by` | string | No | `"ingested_at"` | Sort field |
| `sort_order` | string | No | `"desc"` | `asc` or `desc` |
| `filter_author` | string | No | - | Filter by author |
| `filter_year` | integer | No | - | Filter by year |

**Response:**
```json
{
  "documents": [
    {
      "document_id": "doc_abc123",
      "file_name": "research_paper.pdf",
      "file_path": "/path/to/research_paper.pdf",
      "ingested_at": "2026-02-02T12:00:00Z",
      "chunk_count": 42,
      "size_bytes": 524288,
      "metadata": {
        "author": "Smith",
        "year": 2023,
        "title": "Quantum Computing Advances"
      }
    }
  ],
  "total": 156,
  "offset": 0,
  "limit": 50,
  "has_more": true
}
```

**Status Codes:**
- `200 OK` - Documents retrieved successfully
- `400 Bad Request` - Invalid query parameters

**Example:**
```bash
# List first 10 documents
curl "http://localhost:8000/documents?limit=10"

# Filter by author
curl "http://localhost:8000/documents?filter_author=Smith"

# Pagination
curl "http://localhost:8000/documents?offset=50&limit=25"
```

---

### Get Document Details

**GET `/documents/{document_id}`**

Get detailed information about a specific document.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `document_id` | string | Yes | Document identifier |

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `include_chunks` | boolean | No | `false` | Include chunk previews |
| `chunk_limit` | integer | No | `10` | Max chunks to include |

**Response:**
```json
{
  "document_id": "doc_abc123",
  "file_name": "research_paper.pdf",
  "file_path": "/path/to/research_paper.pdf",
  "ingested_at": "2026-02-02T12:00:00Z",
  "chunk_count": 42,
  "size_bytes": 524288,
  "metadata": {
    "author": "Smith",
    "year": 2023,
    "title": "Quantum Computing Advances"
  },
  "chunks": [
    {
      "chunk_id": "chunk_1",
      "content_preview": "Introduction to quantum computing...",
      "word_count": 287,
      "page_number": 1,
      "section_title": "Introduction",
      "entities": ["quantum computing", "qubits"],
      "has_embedding": true
    }
  ],
  "statistics": {
    "total_words": 12045,
    "average_chunk_size": 287,
    "enrichment_coverage": {
      "entities": 42,
      "questions": 38,
      "embeddings": 42
    }
  }
}
```

**Status Codes:**
- `200 OK` - Document found
- `404 Not Found` - Document not found

**Example:**
```bash
# Basic document info
curl http://localhost:8000/documents/doc_abc123

# With chunk previews
curl "http://localhost:8000/documents/doc_abc123?include_chunks=true&chunk_limit=5"
```

---

### Delete Document

**DELETE `/documents/{document_id}`**

Remove a document and all its chunks from the knowledge base.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `document_id` | string | Yes | Document identifier |

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `confirm` | boolean | No | `false` | Must be `true` to confirm deletion |

**Response:**
```json
{
  "document_id": "doc_abc123",
  "status": "deleted",
  "chunks_removed": 42,
  "storage_freed_bytes": 524288,
  "deleted_at": "2026-02-02T13:00:00Z"
}
```

**Status Codes:**
- `200 OK` - Document deleted successfully
- `400 Bad Request` - Confirmation not provided
- `404 Not Found` - Document not found
- `500 Internal Server Error` - Deletion failed

**Example:**
```bash
# Delete document (requires confirmation)
curl -X DELETE "http://localhost:8000/documents/doc_abc123?confirm=true"
```

---

### Get Statistics

**GET `/stats`**

Get knowledge base statistics and metrics.

**Query Parameters:** None

**Response:**
```json
{
  "total_documents": 156,
  "total_chunks": 6543,
  "total_size_bytes": 125829120,
  "storage_backend": "chromadb",
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_provider": "gemini",
  "enrichment_stats": {
    "entities": 6543,
    "questions": 6012,
    "embeddings": 6543
  },
  "document_types": {
    "pdf": 142,
    "html": 10,
    "docx": 4
  },
  "average_chunks_per_document": 42,
  "oldest_document": "2025-12-01T10:00:00Z",
  "newest_document": "2026-02-02T12:00:00Z"
}
```

**Status Codes:**
- `200 OK` - Statistics retrieved

**Example:**
```bash
curl http://localhost:8000/stats
```

---

### Batch Ingest

**POST `/ingest/batch`**

Ingest multiple documents in a single request.

**Request Body:**
```json
{
  "files": [
    {
      "file_path": "/path/to/doc1.pdf",
      "metadata": {"author": "Smith"}
    },
    {
      "file_path": "/path/to/doc2.pdf",
      "metadata": {"author": "Jones"}
    }
  ],
  "chunking_strategy": "semantic",
  "enable_enrichment": true
}
```

**Response:**
```json
{
  "total_files": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "file_path": "/path/to/doc1.pdf",
      "document_id": "doc_001",
      "status": "completed",
      "chunks_created": 38
    },
    {
      "file_path": "/path/to/doc2.pdf",
      "document_id": "doc_002",
      "status": "completed",
      "chunks_created": 45
    }
  ],
  "total_processing_time_seconds": 24.7
}
```

**Status Codes:**
- `201 Created` - Batch ingestion completed (check individual results for failures)
- `400 Bad Request` - Invalid request format

**Example:**
```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{
    "files": [
      {"file_path": "./docs/paper1.pdf"},
      {"file_path": "./docs/paper2.pdf"}
    ]
  }'
```

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {
    "field": "Additional context",
    "suggestion": "How to fix"
  },
  "timestamp": "2026-02-02T12:00:00Z",
  "request_id": "req_xyz789"
}
```

**Common Error Types:**

| Status Code | Error Type | Description | Solution |
|-------------|------------|-------------|----------|
| 400 | `BadRequest` | Invalid request parameters | Check request format and parameters |
| 401 | `Unauthorized` | Missing or invalid API key | Provide valid API key in `X-API-Key` header |
| 404 | `NotFound` | Resource not found | Verify document ID or file path |
| 409 | `Conflict` | Resource already exists | Use different document ID or update existing |
| 413 | `PayloadTooLarge` | Request body too large | Reduce batch size or file size |
| 429 | `RateLimitExceeded` | Too many requests | Wait and retry, check rate limit headers |
| 500 | `InternalServerError` | Server error | Check logs, contact support |
| 503 | `ServiceUnavailable` | Server not ready | Wait for server to be healthy |

**Example Error Response:**
```json
{
  "error": "NotFound",
  "message": "Document with ID 'doc_invalid' not found",
  "details": {
    "document_id": "doc_invalid",
    "suggestion": "Use GET /documents to list available documents"
  },
  "timestamp": "2026-02-02T12:00:00Z",
  "request_id": "req_abc123"
}
```

---

## Rate Limiting

**Default Limits:**
- 100 requests/minute per API key
- 1000 requests/hour per API key
- 10000 requests/day per API key

**Rate Limit Headers:**

Every response includes rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1706875200
X-RateLimit-Reset-After: 45
```

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Requests allowed per window |
| `X-RateLimit-Remaining` | Requests remaining in current window |
| `X-RateLimit-Reset` | Unix timestamp when limit resets |
| `X-RateLimit-Reset-After` | Seconds until reset |

**When Rate Limited:**

```json
{
  "error": "RateLimitExceeded",
  "message": "Rate limit of 100 requests per minute exceeded",
  "details": {
    "retry_after_seconds": 45,
    "limit": 100,
    "window": "minute"
  }
}
```

**Best Practices:**
- Implement exponential backoff on 429 responses
- Check `X-RateLimit-Remaining` before making requests
- Use batch endpoints when possible
- Cache responses when appropriate

---

## Authentication

**API Key Authentication:**

Include API key in request header:

```bash
curl http://localhost:8000/query \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query": "example"}'
```

**Generating API Keys:**

```bash
# Generate new API key
ingestforge api-key create --name "My App"

# Output:
# API Key: if_api_key_prod_example_123
# Created: 2026-02-02T12:00:00Z
# Permissions: read, write
```

**Configuration:**

```yaml
# config.yaml
api:
  require_auth: true
  api_keys:
    - key: "if_api_key_prod_example_123"
      name: "Production Key"
      permissions: ["read", "write"]
      rate_limit: 1000

    - key: "if_api_key_dev_example_456"
      name: "Development Key"
      permissions: ["read"]
      rate_limit: 100
```

**Permissions:**

| Permission | Allows |
|------------|--------|
| `read` | GET requests (query, list, get details) |
| `write` | POST requests (ingest, batch) |
| `delete` | DELETE requests (remove documents) |
| `admin` | All operations + statistics |

---

## Pagination

**Standard Pagination:**

All list endpoints support offset-based pagination:

```bash
# First page
curl "http://localhost:8000/documents?limit=50&offset=0"

# Second page
curl "http://localhost:8000/documents?limit=50&offset=50"
```

**Response Format:**

```json
{
  "items": [...],
  "total": 156,
  "offset": 0,
  "limit": 50,
  "has_more": true,
  "next_offset": 50
}
```

**Pagination Headers:**

```
X-Total-Count: 156
X-Offset: 0
X-Limit: 50
Link: <http://localhost:8000/documents?offset=50&limit=50>; rel="next"
```

---

## Filtering and Sorting

**Metadata Filters:**

```bash
# Filter by author
curl "http://localhost:8000/documents?filter_author=Smith"

# Filter by year
curl "http://localhost:8000/documents?filter_year=2023"

# Multiple filters (AND logic)
curl "http://localhost:8000/documents?filter_author=Smith&filter_year=2023"
```

**Sorting:**

```bash
# Sort by ingestion date (newest first)
curl "http://localhost:8000/documents?sort_by=ingested_at&sort_order=desc"

# Sort by document name
curl "http://localhost:8000/documents?sort_by=file_name&sort_order=asc"
```

**Available Sort Fields:**
- `ingested_at` - Ingestion timestamp
- `file_name` - Document filename
- `chunk_count` - Number of chunks
- `size_bytes` - File size

---

## Webhook Support (Planned)

**Future webhook events:**

```json
{
  "event": "document.ingested",
  "timestamp": "2026-02-02T12:00:00Z",
  "data": {
    "document_id": "doc_abc123",
    "file_name": "paper.pdf",
    "chunks_created": 42
  }
}
```

**Planned Events:**
- `document.ingested` - Document processing completed
- `document.failed` - Document processing failed
- `query.completed` - Query completed
- `enrichment.completed` - Enrichment pipeline finished

**Configuration (Future):**

```yaml
api:
  webhooks:
    - url: "https://example.com/webhooks/ingestforge"
      events: ["document.ingested", "document.failed"]
      secret: "${WEBHOOK_SECRET}"
```

---

## Client Libraries

### Python Client (Planned)

```python
from ingestforge_client import IngestForgeClient

# Initialize client
client = IngestForgeClient(
    api_key="your-api-key",
    base_url="http://localhost:8000"
)

# Ingest document
result = client.ingest("document.pdf", metadata={"author": "Smith"})
print(f"Created {result.chunks_created} chunks")

# Query
response = client.query("What are the main findings?", top_k=5)
print(response.answer)

for citation in response.citations:
    print(f"  - {citation.source_file}, p.{citation.page_number}")
```

### JavaScript/TypeScript Client (Planned)

```typescript
import { IngestForgeClient } from '@ingestforge/client';

// Initialize client
const client = new IngestForgeClient({
  apiKey: 'your-api-key',
  baseURL: 'http://localhost:8000'
});

// Ingest document
const result = await client.ingest({
  filePath: 'document.pdf',
  metadata: { author: 'Smith' }
});

console.log(`Created ${result.chunksCreated} chunks`);

// Query
const response = await client.query({
  query: 'What are the main findings?',
  topK: 5
});

console.log(response.answer);
response.citations.forEach(citation => {
  console.log(`  - ${citation.sourceFile}, p.${citation.pageNumber}`);
});
```

---

## Versioning

The API uses semantic versioning with URL-based versioning:

**Current Version:** `v1`

**Version Header:**
```
X-API-Version: v1
```

**Version in URL (future):**
```
http://localhost:8000/v1/query
http://localhost:8000/v2/query
```

**Breaking Changes:**
- Major version increments for breaking changes
- Deprecated endpoints supported for 2 major versions
- Deprecation warnings in response headers:
  ```
  X-API-Deprecated: true
  X-API-Sunset: 2027-01-01
  ```

---

## CORS Configuration

**Default CORS Settings:**

```yaml
api:
  cors:
    enabled: true
    allow_origins:
      - "http://localhost:3000"
      - "http://localhost:5173"
    allow_methods:
      - GET
      - POST
      - DELETE
    allow_headers:
      - Content-Type
      - X-API-Key
    max_age: 3600
```

**Headers:**
```
Access-Control-Allow-Origin: http://localhost:3000
Access-Control-Allow-Methods: GET, POST, DELETE
Access-Control-Allow-Headers: Content-Type, X-API-Key
Access-Control-Max-Age: 3600
```

---

## Health Monitoring

**Detailed Health Check:**

**GET `/health/detailed`**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "checks": {
    "storage": {
      "status": "healthy",
      "backend": "chromadb",
      "connection": "ok",
      "free_space_gb": 45.2
    },
    "embeddings": {
      "status": "healthy",
      "model": "all-MiniLM-L6-v2",
      "loaded": true
    },
    "llm": {
      "status": "healthy",
      "provider": "gemini",
      "api_reachable": true
    }
  }
}
```

---

## Support

**Documentation:**
- Full API Reference: [docs.ingestforge.io/api](https://docs.ingestforge.io/api)
- User Guide: [docs.ingestforge.io](https://docs.ingestforge.io)
- Architecture: [ARCHITECTURE.md](../ARCHITECTURE.md)

**Community:**
- GitHub Issues: [github.com/yourusername/ingestforge/issues](https://github.com/yourusername/ingestforge/issues)
- Discord: [discord.gg/ingestforge](https://discord.gg/ingestforge)
- Email: support@ingestforge.io

**Status Page:**
- API Status: [status.ingestforge.io](https://status.ingestforge.io)
- Incident History: [status.ingestforge.io/history](https://status.ingestforge.io/history)

---

*Last updated: 2026-02-02*
*API Version: v1*
*Document Version: 1.0*
