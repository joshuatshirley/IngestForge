# Technical Spec: Knowledge Graph & concept Mapping

## Goal
Transform extracted entities and relationships into an interactive, traversable graph.

---

## 1. Graph Construction (Rule #1: Simple Flow)
Avoid complex graph-database dependencies for the local engine. Use `NetworkX` for logic and D3.js for rendering.

**Logic**:
1.  **Nodes**: Every unique `Entity` (from EntityExtractor) becomes a node.
2.  **Edges**: Created when two entities appear in the same chunk OR have a semantic similarity > 0.85.
3.  **Metadata**: Store `chunk_ids` on edges as evidence.

---

## 2. Sparse Data Export (Rule #3: Memory Management)
Large graphs must be exported in a sparse format to avoid browser crashes.

```json
{
  "nodes": [{"id": "NER_1", "label": "Quantum gravity", "type": "CONCEPT"}],
  "links": [{"source": "NER_1", "target": "NER_2", "weight": 0.9, "evidence": ["uuid_1"]}]
}
```

---

## 3. Heuristic Hierarchies
*   **Authority Nodes**: Nodes with degree > 10 are styled as "Core Concepts."
*   **Islands**: Nodes with degree 0 are filtered out by default to reduce noise.

---

## 4. Implementation Commands
`ingestforge analyze graph --output concept_map.html`
*   **Assertion**: `assert graph.number_of_nodes < 2000` (Rule #2: prevent massive graph renders).
*   **Validation**: Check that every `evidence` ID exists in the storage backend.
