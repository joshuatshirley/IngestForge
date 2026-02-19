# IngestForge Documentation

> Single source of truth for all IngestForge documentation

---

## Quick Navigation

### For Users

| Document | Description |
|----------|-------------|
| [guides/GETTING_STARTED.md](guides/GETTING_STARTED.md) | Complete getting started guide |
| [guides/AGENT_USAGE.md](guides/AGENT_USAGE.md) | Autonomous agent guide |
| [guides/YOUTUBE_INGESTION.md](guides/YOUTUBE_INGESTION.md) | YouTube video ingestion |
| [cli.md](cli.md) | CLI command reference |
| [configuration.md](configuration.md) | Configuration options |
| [docker.md](docker.md) | Docker deployment |
| [API.md](API.md) | REST API reference |

### For Developers

| Document | Description |
|----------|-------------|
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | System architecture with Mermaid diagrams |
| [guides/ONBOARDING.md](guides/ONBOARDING.md) | 15-minute developer onboarding |
| [guides/FEATURE_TEMPLATE.md](guides/FEATURE_TEMPLATE.md) | Adding new features checklist |
| [../REFACTORING.md](../REFACTORING.md) | Hexagonal architecture refactoring tracker |
| [../TESTING_REQUIREMENTS.md](../TESTING_REQUIREMENTS.md) | Testing guidelines and standards |

---

## Architecture Decision Records

Key architectural decisions and their rationale:

| ADR | Topic | Status |
|-----|-------|--------|
| [ADR-001](architecture/ADR-001-hexagonal-architecture.md) | Hexagonal architecture adoption | Accepted |
| [ADR-002](architecture/ADR-002-hybrid-retrieval.md) | Hybrid retrieval (BM25 + semantic) | Accepted |
| [ADR-003](architecture/ADR-003-semantic-chunking.md) | Semantic chunking strategy | Accepted |
| [ADR-004](architecture/ADR-004-citation-provenance.md) | Citation and provenance tracking | Accepted |
| [ADR-005](architecture/ADR-005-ienricher-interface.md) | IEnricher interface standardization | Accepted |
| [ADR-006](architecture/ADR-006-llm-retry-consolidation.md) | LLM retry logic consolidation | Accepted |
| [ADR-TEMPLATE](architecture/ADR-TEMPLATE.md) | Template for new ADRs | - |

---

## Module Documentation

Each module has its own README with architecture context, usage examples, and extension points:

| Module | Description | README |
|--------|-------------|--------|
| **core** | Framework infrastructure (config, logging, retry, security) | [README](../ingestforge/core/README.md) |
| **shared** | Reusable patterns and interfaces | [README](../ingestforge/shared/README.md) |
| **ingest** | Document processing (PDF, HTML, OCR) | [README](../ingestforge/ingest/README.md) |
| **chunking** | Semantic text chunking strategies | [README](../ingestforge/chunking/README.md) |
| **enrichment** | Embeddings and metadata extraction | [README](../ingestforge/enrichment/README.md) |
| **storage** | Storage backends (ChromaDB, JSONL) | [README](../ingestforge/storage/README.md) |
| **retrieval** | Search strategies (BM25, semantic, hybrid) | [README](../ingestforge/retrieval/README.md) |
| **query** | Query pipeline and caching | [README](../ingestforge/query/README.md) |
| **llm** | LLM provider integrations | [README](../ingestforge/llm/README.md) |
| **cli** | Command-line interface | [README](../ingestforge/cli/README.md) |

---

## Project Planning

| Document | Description |
|----------|-------------|
| [planning/WORKFLOW.md](planning/WORKFLOW.md) | End-to-end project workflow (14 steps) |
| [planning/BACKLOG.md](planning/BACKLOG.md) | Remaining work (~91 items: P2, P3, dead code) |
| [planning/COMPLETED.md](planning/COMPLETED.md) | Archive of shipped features (190 items) |
| [planning/ROBUSTNESS_ENHANCEMENTS.md](planning/ROBUSTNESS_ENHANCEMENTS.md) | Production-readiness enhancements |
| [planning/ACCURACY_FRAMEWORK.md](planning/ACCURACY_FRAMEWORK.md) | Accuracy-first design principles |
| [planning/AUDIT_TRAIL_DESIGN.md](planning/AUDIT_TRAIL_DESIGN.md) | Source traceability and verification |
| [../FEATURES.md](../FEATURES.md) | Implemented features with examples |
| [../CHANGELOG.md](../CHANGELOG.md) | Version history |

---

## Documentation Standards

### Structure

- **Root README.md**: User-facing quick start and command reference
- **ARCHITECTURE.md**: Technical deep-dive with visual diagrams
- **Module READMEs**: Contextual documentation at point of need
- **ADRs**: Architectural decisions with rationale

### Principles

1. **Single source of truth**: Each topic documented once
2. **Progressive disclosure**: Brief READMEs, detailed in-depth docs
3. **Visual first**: Mermaid diagrams before detailed text
4. **Worked examples**: Complete examples, not snippets

---

*Last updated: 2026-02-07*
