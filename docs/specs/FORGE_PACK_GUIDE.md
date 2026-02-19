# Forge Pack Development Guide: Vertical Expansion

## Overview
A "Forge Pack" is a vertical-specific extension of the IngestForge engine (e.g., Legal, Cybersecurity, Academic Research). Each pack must maintain zero dependencies on other packs and interact only with the `core` interfaces.

---

## 1. Pack Structure
Each pack should live in `ingestforge/verticals/<pack_name>/`.

```text
verticals/<pack_name>/
├── __init__.py
├── scrapers.py      # Site-specific adapters (Fandom, CaseLaw, etc)
├── refiners.py      # Vertical-specific text cleaning (PII, log patterns)
├── prompts/         # Specialized LLM system messages
└── cli.py           # Typer sub-commands
```

---

## 2. Standard Verticals & Strategic Priority

### 2.1 Legal Forge [P2]
*   **Focus**: Court listener API integration, citation auto-formatting (Bluebook).
*   **Key Task**: `LEGAL-001`: Implement CaseLaw scraper with automatic "Opinion" vs "Fact" tagging.

### 2.2 Cyber Forge [P3]
*   **Focus**: Log parsing (syslog, cloudtrail), CVE discovery.
*   **Key Task**: `CYBER-001`: Implement JSON-Log flattener for semantic search across system events.

### 2.3 Research Forge [P1]
*   **Focus**: arXiv integration, LaTeX math preservation.
*   **Key Task**: `RES-001`: Implement MathJax-to-Unicode converter for local-LLM compatibility.

---

## 3. Implementation Rules
1.  **Rule #8 (Abstractions)**: Do NOT create a complex "VerticalBase" class. Use composition and register extensions via the `BackendRegistry` or `ScraperRegistry`.
2.  **Rule #9 (Types)**: Every vertical-specific metadata field must be added to a typed `Metadata` extension class.
3.  **Rule #4 (Size)**: No vertical module should exceed 200 lines. If a scraper is complex, split it into `parser.py` and `client.py`.

---

## 4. Integration Workflow
1.  Register pack scrapers in `core/scraping/registry.py`.
2.  Register pack CLI in `cli/main.py` using `app.add_typer(pack_app, name="legal")`.
3.  Ensure 100% test coverage for pack-specific refiners.
