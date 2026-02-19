# API Reference

IngestForge provides a robust REST API for integrating research capabilities into your own applications.

## 🚀 Interactive Documentation

When the IngestForge server is running, you can access the interactive **Swagger UI** at:

`http://localhost:8000/docs`

This allows you to test endpoints directly from your browser.

## 📂 Core Endpoints

### 🔎 Search
`POST /v1/search`

Search the knowledge base with metadata filters.

**Request**:
```json
{
  "query": "quantum gravity",
  "top_k": 5,
  "filters": {
    "doc_type": "PDF"
  }
}
```

### 📥 Ingestion
`POST /v1/ingest/upload`

Upload a local file for processing.

**Form Data**:
*   `file`: The document file (binary)

### 🤖 Agent Control
`POST /v1/agent/run`

Launch an autonomous research mission.

**Request**:
```json
{
  "task": "Synthesize the historical context of the Treaty of Versailles",
  "max_steps": 10
}
```

## 🔒 Authentication

IngestForge uses JWT (JSON Web Token) for authentication.

1.  **Login**: `POST /v1/auth/login` with your admin credentials.
2.  **Authorize**: Include the returned token in the `Authorization: Bearer <token>` header for all subsequent requests.
