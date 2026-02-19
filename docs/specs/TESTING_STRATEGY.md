# IngestForge System Verification Strategy (Testing)

## Goal
Verify non-deterministic outputs (OCR, Retrieval, Agents) using deterministic "Golden Datasets" and regression boundaries.

---

## 1. OCR Layout Verification (The "Pixel-to-Text" Test)
**Golden Dataset**: `tests/data/golden/ocr/`
*   **Input**: `sample_2col.png`, `sample_table.png`
*   **Expected**: `sample_2col.md` (Ground truth markdown).

**Metric**: Levenshtein distance < 5% between generated markdown and ground truth.
**Assertion**: `assert reconstruct_layout(blocks) == expected_md` (Rule #5).

---

## 2. Retrieval Recall (The "Needle-in-Haystack" Test)
**Golden Dataset**: `tests/data/golden/retrieval/`
*   **Dataset**: 100 mixed academic papers.
*   **Query**: Specific technical questions.
*   **Expected**: List of `chunk_id`s that MUST be in top-3 results.

**Metric**: Success if `expected_id in results[:3]`.

---

### 3. Agent Safety & Bounds (ReAct Verification)
**Mocks**:
*   All Agent tests MUST use `MockLLM` to return pre-defined thought/action chains.
*   Verify that if the MockLLM attempts step 11, the engine raises `AssertionError` (Rule #2).

---

## 4. JPL Rule #10 Compliance (Strict Checks)
The test runner (`pytest`) must be configured to fail if:
1.  **Mypy** coverage < 100%.
2.  **Assertion Density** < 1 per 10 lines.
3.  **Cyclomatic Complexity** > 10 for any function.
