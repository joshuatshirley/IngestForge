# ADR-XXX: [Short Title]

## Status

🚧 Proposed | ✅ Accepted | ⚠️ Deprecated | ❌ Rejected

**Date:** YYYY-MM-DD
**Deciders:** [Names/Roles]
**Consulted:** [Stakeholders consulted]

## Context

### Problem Statement

[What issue are we facing? What are the forces at play?]

### Background

[Additional context, constraints, requirements that informed this decision]

### Current State (if applicable)

[What's the current approach and why is it problematic?]

## Decision

[What did we decide to do? Be specific and prescriptive.]

### Implementation Approach

[High-level approach to implementing the decision]

**Before:**
```python
# Code example showing old approach (if applicable)
```

**After:**
```python
# Code example showing new approach
```

## Consequences

### Positive ✅

- Benefit 1
- Benefit 2
- Benefit 3

### Negative ⚠️

- Drawback 1 - [How we mitigate this]
- Drawback 2 - [How we mitigate this]

### Risks Mitigated 🛡️

- Risk 1 - How this decision addresses it
- Risk 2 - How this decision addresses it

### Neutral 📊

- Trade-off 1 (neither good nor bad, just different)

## Alternatives Considered

### Alternative 1: [Name]

**Description:** [What this alternative involves]

**Pros:**
- Pro 1
- Pro 2

**Cons:**
- Con 1
- Con 2

**Decision:** Rejected because [reason]

### Alternative 2: [Name]

[Same structure]

## Implementation Notes

### Files Affected

- `path/to/file1.py` - Description of changes
- `path/to/file2.py` - Description of changes
- `path/to/file3.py` - Description of changes

### Migration Strategy (if applicable)

[How do we get from old to new? What's the migration path for existing code/data?]

### Testing Strategy

[How do we verify this decision works? What tests should we add/update?]

## Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| LOC (Lines of Code) | X | Y | +/- Z |
| Test Coverage | X% | Y% | +/- Z% |
| Performance | X ms | Y ms | +/- Z% |
| Code Duplication | X instances | Y instances | -Z |

## References

- [Link to related PR]
- [Link to related issues]
- [Link to related ADRs]
- [Link to external resources, papers, blog posts]
- [REFACTORING.md sections]
- [ARCHITECTURE.md sections]

## Notes

[Additional context, lessons learned, future considerations]

---

## Example Usage

**When to create an ADR:**
- Architectural decisions that affect multiple modules
- Technology/framework choices
- Significant design pattern adoptions
- Breaking changes to public APIs
- Performance/security trade-offs

**When NOT to create an ADR:**
- Simple bug fixes
- Code refactoring without architectural impact
- Documentation updates
- Trivial configuration changes

**ADR Numbering:**
- Use sequential numbering: ADR-001, ADR-002, etc.
- Never reuse numbers
- Deprecated ADRs keep their number with ⚠️ status
