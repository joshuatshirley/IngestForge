# Technical Spec: Forge Pack Registry (Extensibility)

## Goal
Enable multi-vertical expansion (Legal Forge, Cyber Forge, etc.) via a typed registry system. Plugins must be "self-registering" without modifying `ingestforge/core`.

---

## 1. The Registry Pattern (JPL Rule #8)
Avoid complex class inheritance. Use a singleton-registry for each extension point.

**Extension Points**:
1.  **Scrapers**: `ingestforge.core.scraping.registry`
2.  **Refiners**: `ingestforge.ingest.refinement.registry`
3.  **Prompts**: `ingestforge.core.agents.prompts.registry`

---

## 2. Registration Logic
Packs must use an `init_pack` entry point called during system discovery.

```python
# Example: ingestforge/verticals/legal/__init__.py
def init_pack:
    from ingestforge.core.scraping.registry import ScraperRegistry
    from .scrapers import CaseLawScraper
    ScraperRegistry.register("caselaw", CaseLawScraper)
```

---

## 3. Mandatory Interface Compliance (JPL Rule #9)
All vertical extensions MUST implement the base interface and provide typed metadata.

```python
class CaseLawScraper(BaseScraper):
    def get_metadata_schema(self) -> dict:
        """Define vertical-specific fields (e.g., 'jurisdiction', 'court_level')."""
        return {
            "jurisdiction": str,
            "docket_number": str
        }
```

---

## 4. Discovery Mechanism (Rule #1)
The core engine will scan the `verticals/` directory once on startup.
1.  Iterate sub-directories in `ingestforge/verticals/`.
2.  Check for `__init__.py`.
3.  Execute `init_pack` if present.
4.  Limit scan to 1 level deep (Rule #2).
