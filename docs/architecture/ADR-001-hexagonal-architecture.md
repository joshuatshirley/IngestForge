# ADR-001: Hexagonal Architecture Adoption

## Status

✅ Accepted

**Date:** 2024-01-15
**Deciders:** Core development team
**Consulted:** Community contributors, early adopters

## Context

### Problem Statement

IngestForge started as a single-file prototype that grew organically into a multi-module system. By late 2023, the codebase exhibited classic "big ball of mud" symptoms:

- **Tight coupling**: Feature modules directly imported from each other, creating circular dependencies
- **Code duplication**: The same utility functions (retry logic, text cleaning, lazy imports) were duplicated across 3-5 modules
- **Hard to test**: Tightly coupled code made unit testing difficult, requiring extensive mocking
- **Hard to extend**: Adding new document processors or enrichers required modifying core files
- **No clear boundaries**: Unclear where new features should go, leading to inconsistent organization

### Background

The codebase had grown to approximately 8,000 LOC across 45 files with:
- 122 LOC of duplicated retry logic (across 3 LLM clients)
- 87 LOC of duplicated text cleaning code (across 4 processors)
- Inconsistent enricher APIs (entities used `.enrich`, questions used `.generate`)
- No interface contracts - implementations varied wildly

**Trigger event**: Attempt to add PostgreSQL storage backend revealed tight coupling to ChromaDB, requiring changes across 7 files.

### Current State

```python
# Before: Tight coupling, no interfaces
from ingestforge.storage.chromadb import ChromaDBStorage  # Concrete class
from ingestforge.enrichment.entities import extract_entities  # Function
from ingestforge.enrichment.questions import QuestionGenerator  # Class

# Inconsistent APIs
entities = extract_entities(text)  # Function call
questions = QuestionGenerator.generate(chunk)  # Method call
embeddings = get_embeddings(texts)  # Function call
```

## Decision

**We adopt hexagonal architecture (ports and adapters) with three layers:**

1. **Core Layer** (`core/`) - Framework infrastructure with zero feature dependencies
   - Configuration, logging, retry, security, provenance
   - Defines core data structures and orchestration

2. **Shared Layer** (`shared/`) - Reusable patterns and interfaces
   - Interface definitions (IEnricher, IProcessor, IChunkingStrategy)
   - Common utilities (text_utils, lazy_imports, metadata_serialization)
   - Design patterns (EnrichmentPipeline, ProcessorFactory)

3. **Feature Layer** (`ingest/`, `enrichment/`, `storage/`, `retrieval/`, etc.)
   - Implement interfaces from shared layer
   - Depend on core and shared, never on each other directly
   - Domain-specific logic isolated within each module

### Implementation Approach

**Phase 1: Extract Core Utilities** (Completed)
- Move retry logic to `core/retry.py` with `@llm_retry` decorator
- Create `shared/text_utils.py` for common text operations
- Create `shared/lazy_imports.py` for dependency management

**Phase 2: Define Interfaces** (Completed)
- `IEnricher` interface for all enrichers
- `IProcessor` interface for document processors
- `IChunkingStrategy` interface for chunking strategies

**Phase 3: Migrate Implementations** (Completed)
- Refactor enrichers to implement `IEnricher`
- Refactor processors to implement `IProcessor`
- Create `EnrichmentPipeline` for composition

**Before:**
```python
# Tight coupling - enrichment depends on storage
from ingestforge.storage.chromadb import ChromaDBStorage

class EntityExtractor:
    def __init__(self):
        self.storage = ChromaDBStorage  # Hardcoded dependency
```

**After:**
```python
# Loose coupling - enrichment depends on interface
from ingestforge.shared.patterns import IEnricher

class EntityExtractor(IEnricher):
    def enrich_chunk(self, chunk):  # Standardized API
        chunk.entities = self.extract(chunk.content)
        return chunk

    def is_available(self):
        return True
```

## Consequences

### Positive ✅

- **Eliminated duplication**: -122 LOC of retry logic, -87 LOC of text utils (total: -209 LOC duplicates)
- **Clear extension points**: Adding new enrichers/processors/chunkers follows obvious pattern
- **Improved testability**: Interfaces enable easy mocking, test coverage increased from 62% to 78%
- **Better modularity**: Features can evolve independently without breaking others
- **Consistent APIs**: All enrichers have same interface (`.enrich_chunk`, `.is_available`)
- **Easier onboarding**: New developers understand where code belongs

### Negative ⚠️

- **More files**: +17 files in `shared/patterns/` (mitigated by better organization)
- **Slight indirection**: One extra hop through interface (negligible performance impact: <1ms)
- **Migration effort**: Required refactoring 2,847 LOC across 23 files over 3 weeks
- **Learning curve**: Developers must learn interface patterns (mitigated by comprehensive READMEs)

### Risks Mitigated 🛡️

- **Feature coupling risk**: Previously, adding storage backend required changing enrichment code. Now isolated.
- **Testing brittleness**: Previously, tests broke when unrelated features changed. Now isolated.
- **Code duplication drift**: Previously, same logic diverged across copies. Now single source of truth.

### Neutral 📊

- **LOC increase**: +1,017 LOC in shared layer, but -209 LOC duplicates = net +808 LOC (8.6% increase)
  - Trade-off: More code, but better organized and less duplication
- **Build complexity**: No change (pure Python, no additional build steps)

## Alternatives Considered

### Alternative 1: Monolithic Refactor

**Description:** Keep single-module structure but clean up imports and reduce coupling through careful dependency management.

**Pros:**
- Simpler structure (fewer directories)
- No interface abstractions needed
- Faster initial implementation

**Cons:**
- Doesn't solve fundamental coupling issues
- Still requires extensive refactoring
- Coupling tends to creep back in over time
- No clear extension points

**Decision:** Rejected because it doesn't provide long-term architectural benefits. Addresses symptoms, not root cause.

### Alternative 2: Microservices Architecture

**Description:** Split into separate services (ingestion service, enrichment service, storage service) communicating via APIs.

**Pros:**
- Maximum isolation between components
- Independently deployable
- Technology-agnostic boundaries

**Cons:**
- Massive overkill for a library/framework
- Introduces network latency and reliability issues
- Complicates local development and testing
- Users want a library, not a distributed system

**Decision:** Rejected as inappropriate for a Python library. IngestForge is meant to be embedded, not distributed.

### Alternative 3: Plugin Architecture

**Description:** Core provides plugin hooks, all features are plugins loaded at runtime.

**Pros:**
- Maximum extensibility
- Users can add features without modifying core
- Clean separation

**Cons:**
- Overcomplicated for current needs
- Slower due to dynamic loading
- Harder to debug
- More complex for users to configure

**Decision:** Rejected as premature. Hexagonal architecture provides similar benefits with less complexity. Could evolve to this later if needed.

## Implementation Notes

### Files Affected

**New files created (17):**
- `ingestforge/shared/patterns/enricher.py` - IEnricher interface + EnrichmentPipeline
- `ingestforge/shared/patterns/processor.py` - IProcessor interface + ProcessorFactory
- `ingestforge/shared/patterns/chunking.py` - IChunkingStrategy interface
- `ingestforge/shared/text_utils.py` - Centralized text utilities
- `ingestforge/shared/lazy_imports.py` - Lazy loading patterns
- `ingestforge/shared/metadata_serialization.py` - JSON helpers
- [11 more in shared/]

**Files refactored (23):**
- `ingestforge/enrichment/entities.py` - Now implements IEnricher
- `ingestforge/enrichment/questions.py` - Now implements IEnricher
- `ingestforge/enrichment/embeddings.py` - Now implements IEnricher
- `ingestforge/ingest/pdf_processor.py` - Now implements IProcessor
- `ingestforge/llm/gemini.py` - Now uses @llm_retry
- `ingestforge/llm/claude.py` - Now uses @llm_retry
- `ingestforge/llm/openai.py` - Now uses @llm_retry
- [16 more files]

### Migration Strategy

**3-Phase approach:**
1. **Phase 1** (Week 1): Extract utilities to `shared/`, update imports (non-breaking)
2. **Phase 2** (Week 2): Define interfaces, create default implementations (non-breaking)
3. **Phase 3** (Week 3): Migrate existing code to use interfaces (breaking changes documented)

**Backward compatibility:**
- Old function-based APIs deprecated with warnings pointing to new classes
- Deprecation period: 2 releases (6 months) before removal
- Migration guide provided in REFACTORING.md

### Testing Strategy

- All new interfaces include abstract test base classes
- Existing integration tests continue to pass (validates refactoring correctness)
- New unit tests for each interface implementation
- Test coverage target: 75% → 80% (achieved: 78%)

## Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| LOC (Total) | 8,247 | 9,055 | +808 (+9.8%) |
| LOC (Duplicates) | 209 | 0 | -209 (-100%) |
| LOC (Shared utilities) | 0 | 1,017 | +1,017 |
| Files | 45 | 62 | +17 (+37.8%) |
| Modules | 8 | 10 | +2 (shared, patterns) |
| Test Coverage | 62% | 78% | +16% |
| Cyclomatic Complexity (avg) | 8.4 | 6.1 | -2.3 (-27.4%) |
| Import Coupling | 34 cross-module | 12 cross-module | -22 (-64.7%) |

## References

- [REFACTORING.md](../../REFACTORING.md) - Phases 1-3 implementation details
- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Hexagonal architecture overview
- PR #45: Phase 1 - Extract core utilities
- PR #52: Phase 2 - Define interfaces
- PR #58: Phase 3 - Migrate implementations
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/) - Original article by Alistair Cockburn
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) - Similar concepts by Robert C. Martin

## Notes

**Lessons learned:**
- Incremental refactoring (3 phases) was crucial - attempting it all at once would have been overwhelming
- Writing interfaces first, then implementations, clarified design
- Extensive testing during refactoring caught several subtle bugs
- Documentation updates (module READMEs) helped solidify the new structure

**Future considerations:**
- Could introduce dependency injection container if complexity increases
- Plugin system could be layered on top if user extensibility becomes priority
- May need to revisit if we add service/daemon mode (currently just a library)

**Impact on onboarding:**
New developers now have clear mental model:
1. Core = Framework (no dependencies)
2. Shared = Patterns & Interfaces (depends on core)
3. Features = Implementations (depend on core + shared)

This matches their intuition about layered architectures from other frameworks (Django, Spring, etc.).
