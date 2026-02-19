# Technical Spec: Spatial OCR & Layout Reconstruction

## Goal
Convert raw hOCR/ALTO XML into a structured Markdown document that preserves multi-column reading order without using complex recursive tree-traversal (JPL Rule #1).

---

## 1. Coordinate Normalization
All input coordinates (px) must be normalized to a virtual **1000x1000 grid** per page to ensure heuristics work regardless of scan resolution.

**Formula**:
`x_norm = (x_px / page_width_px) * 1000`
`y_norm = (y_px / page_height_px) * 1000`

---

## 2. Column Detection Heuristic (The "Vertical Sweep" Algorithm)
Instead of a tree-walk, use a linear sweep:
1.  **Project**: Collapse all `OCRBlock` objects onto the X-axis.
2.  **Cluster**: Identify "dense" regions on the X-axis where many blocks start/end.
3.  **Threshold**: Gaps in X-projection > 50 units (5% page width) indicate a column break.
4.  **Assign**: Assign every block a `column_id` based on its center-X coordinate.

---

## 3. Reading Order Logic (JPL Rule #2 Compliant)
Sort blocks using a multi-key sort instead of nested loops:
```python
# Rule: Sort by Page -> Column -> Y-coordinate -> X-coordinate
blocks.sort(key=lambda b: (b.page_num, b.column_id, b.y1, b.x1))
```

---

## 4. Validation Assertions (JPL Rule #5)
The parser MUST assert the following before returning a `LayoutMap`:
*   `assert all(0 <= b.x1 <= 1000 for b in blocks)`
*   `assert all(b.x2 >= b.x1 for b in blocks)`
*   `assert len(blocks) < 5000` (Upper bound to prevent OOM on malicious files - Rule #2).

---

## 5. Markdown Transformation
*   **Paragraphs**: If `y_gap` between blocks < 10 units, join with space.
*   **Headers**: If `font_size` > 1.5x median or `is_bold`, wrap in `#`.
*   **Tables**: If 3+ blocks share identical `y1` (tolerance 2 units), trigger Table Mode.
