# FAQ

Frequently asked questions about IngestForge.

## 🔒 Privacy & Data

### Is my data sent to the cloud?
By default, **no**. IngestForge is designed to be "local-first." Document processing, chunking, and vector storage happen entirely on your machine. Cloud connections only occur if:
1.  You explicitly use a cloud connector (Google Drive, Notion).
2.  You configure a cloud-based LLM provider (OpenAI, Claude).

### Can I run IngestForge completely air-gapped?
Yes. You can configure IngestForge to use local embedding models and local LLMs (via Ollama or Llama.cpp), allowing it to run without any internet connection after the initial setup.

## 🧠 Technical Concepts

### What is Hybrid Retrieval?
Hybrid retrieval combines two search methods:
1.  **Keyword (BM25)**: Excellent for finding exact terms, names, and acronyms.
2.  **Semantic (Vector)**: Finds conceptually related content even if the exact words are different.
IngestForge fuses these results using RRF (Reciprocal Rank Fusion) for maximum accuracy.

### What are "Specialized Verticals"?
Verticals are domain-specific logic layers. For example, the **Legal Vertical** understands court citation formats, while the **Cyber Vertical** knows how to parse log files and CVE reports. These ensure that the most important information in your specific field is prioritized.

## 🛠️ Usage

### How many documents can IngestForge handle?
A typical modern laptop can comfortably handle 10,000 to 50,000 document chunks using the ChromaDB backend. For larger corpora (100k+), we recommend using the PostgreSQL/pgvector backend.

### What LLMs are supported?
IngestForge is provider-agnostic. It officially supports:
*   **OpenAI** (GPT-4, GPT-3.5)
*   **Anthropic** (Claude 3)
*   **Google** (Gemini)
*   **Local** (Ollama, Llama.cpp, Mistral)

---

## 🏗️ Architecture

### Why does IngestForge use NASA JPL rules?
Research and intelligence data are mission-critical. By following JPL's "Power of Ten" rules, we ensure the framework is stable, predictable, and free from the "spaghetti code" that often plagues rapidly developing AI tools.
