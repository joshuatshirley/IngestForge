# IngestForge Repository Guide

## 📂 Directory Structure

### 🏗️ Core Application (`ingestforge/`)
The primary Python package containing the RAG engine and CLI.
- `api/`: FastAPI server, routers, and middleware.
- `agent/`: Agentic reasoning loops and tool adapters.
- `core/`: Critical infrastructure (Pipeline, Artifacts, Security, Config).
- `enrichment/`: Processing stages for knowledge extraction (NER, Linking, Summary).
- `ingest/`: Source-specific connectors and text extractors.
- `llm/`: Unified interface for LLM providers (OpenAI, Claude, Gemini, Ollama).
- `retrieval/`: Search logic (Hybrid, BM25, RRF Scorer).
- `storage/`: Database persistence (ChromaDB, Postgres/pgvector).
- `verticals/`: Domain-specific blueprints and logic (Legal, Cyber).

### 🌐 Web Portal (`frontend/`)
React/Next.js application for the visual workbench.
- `src/app/`: Next.js App Router pages.
- `src/components/`: Reusable UI components (MUI based).
- `src/store/`: Redux Toolkit state management.
- `src/hooks/`: Custom React hooks.

### 🧪 Quality Assurance (`tests/`)
- `unit/`: Isolated tests for individual modules.
- `integration/`: Multi-module workflow tests.
- `fixtures/`: Sample data and mock objects.

### 📚 Documentation (`docs/`)
- `api/`: API endpoint references.
- `guides/`: User and developer tutorials.
- `architecture/`: High-level system design and ADRs.

### 🛠️ Tooling & Automation
- `scripts/`: Maintenance, installation, and compliance tools.
- `.github/workflows/`: CI/CD pipelines (Linting, Testing, Security).
- `.archive/`: Local-only development notes and legacy logs (Git-ignored).

## 🚦 Development Workflow

### 1. Requirements
- Backend: Python 3.10+
- Frontend: Node.js 18+

### 2. Linting & Standards
We enforce strict coding standards:
- **NASA JPL Power of Ten**: Verified via `scripts/jpl_lint.py`.
- **Formatting**: `ruff` for Python, `prettier` for TypeScript.
- **Typing**: `mypy` (strict) for Python, `TSC` for TypeScript.

### 3. Testing
- Run backend tests: `pytest tests/`
- Run frontend tests: `npm test` (Vitest)
- Run E2E tests: `npm run test:e2e` (Playwright)

## 🛡️ Security
Security is baked into the CI via **Security Shield**:
- Automated vulnerability scanning on every PR.
- Mandatory `X-Admin-Session` for high-risk operations.
- HSTS and CSP enforced at the middleware layer.
