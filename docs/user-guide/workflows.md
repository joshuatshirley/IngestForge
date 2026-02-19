# Core Workflows

This guide covers the primary ways to interact with IngestForge to build and query your knowledge base.

## 📥 Document Ingestion

IngestForge supports a wide variety of formats, including PDF, EPUB, DOCX, Markdown, and Source Code.

### Local Files
You can ingest individual files or entire directories recursively.

**CLI**:
```bash
ingestforge ingest ./my_research/ --recursive
```

**Web Portal**:
1.  Navigate to the **Ingest Center**.
2.  Drag and drop your files onto the upload zone.
3.  Monitor real-time progress in the queue.

### Cloud Sources
Connect to external platforms to sync your data.

**CLI**:
```bash
ingestforge ingest --source gdrive --folder-id YOUR_ID
```

---

## 🔎 Intelligent Search

IngestForge uses **Hybrid Retrieval**, combining keyword matching (BM25) with semantic vector search.

### Natural Language Queries
You don't need complex syntax. Simply ask a question.

**Example**:
> "What are the latest developments in quantum decoherence mentioned in these papers?"

### Filtering & Sorting
Narrow down your results using metadata.

*   **Date Range**: Filter by publication date.
*   **Document Type**: Search only within PDFs or Source Code.
*   **Source**: Filter by specific libraries or cloud folders.

---

## 🏛️ Domain Verticals

IngestForge automatically detects the domain of your documents during ingestion.

*   **Legal**: Automatically extracts Bluebook citations and case IDs.
*   **Cyber**: Detects CVEs and CVSS scores in security reports.
*   **Medical**: Identifies ICD-10 patterns and dosage information.

You can view domain-specific metadata in the **Result Cards** within the Web Portal.
