# Technical Spec: JS Rendering Pipeline (Playwright)

## Goal
Enable extraction of content from Javascript-heavy SPAs (React, Vue, etc.) while maintaining strict resource cleanup (JPL Rule #3).

---

## 1. Browser Lifecycle (Rule #3: Explicit Cleanup)
Do NOT use global browser instances. Use a context manager for every scrape.

```python
class PlaywrightProvider:
    def scrape(self, url: str) -> str:
        """Rule #1: Max 2 nesting levels."""
        with sync_playwright as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page
            # ... scrape ...
            browser.close
```

---

## 2. Escalation Logic (Rule #7: Parameter Check)
The `ScraperRegistry` should only trigger JS rendering if:
1.  Static scraping returns `< 200` characters.
2.  The `--render` flag is explicitly passed via CLI.
3.  The domain is known to be JS-only (whitelist).

---

## 3. Waiting Strategy (Rule #2: Fixed Upper Bound)
Avoid "Wait for network idle." Use specific element selectors with hard timeouts.
*   **Default Timeout**: 30 seconds.
*   **Assertion**: `assert page.content is not None` before closing.

---

## 4. Implementation Commands
`ingestforge add https://react-docs.com --render`
*   **Verification**: Check RAM usage before/after to ensure no orphaned chromium processes remain.
