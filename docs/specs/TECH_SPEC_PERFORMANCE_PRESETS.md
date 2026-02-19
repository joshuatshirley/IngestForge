# Technical Spec: Performance Presets & Benchmarking

## Goal
Enable IngestForge to run efficiently on everything from a Raspberry Pi (Mobile) to a Workstation (Speed) by dynamically adjusting model selection and parallelism.

---

## 1. Preset Definitions (Rule #7: Valid Ranges)

| Preset | Max Workers | Embedding Model | Chunker Overlap | OCR Engine |
|--------|-------------|-----------------|-----------------|------------|
| **Mobile** | 1 | `all-MiniLM-L6-v2` | 10% | Tesseract (Fast) |
| **Balanced**| CPU/2 | `all-mpnet-base-v2`| 15% | Tesseract (Best) |
| **Speed** | CPU-1 | `all-mpnet-base-v2`| 20% | EasyOCR (GPU) |

---

## 2. Dynamic Throttling Logic (Rule #1)
The `apply_performance_preset` function must be called in `Pipeline.__init__`.

```python
def apply_performance_preset(config: Config) -> Config:
    """JPL #7: Parameter validation based on system resources."""
    mode = config.performance_mode.lower
    
    if mode == "mobile":
        config.indexing.parallel_workers = 1
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_summaries = False
    
    return config
```

---

## 3. Automated Benchmarking (The "Engine Audit")
**Module**: `benchmarks/engine_perf.py`
*   **Metric 1**: Tokens per second (TPS) during ingestion.
*   **Metric 2**: Peak RAM usage (Rule #3 verification).
*   **Assertion**: In **Mobile** mode, peak RAM must stay `< 1GB` for a 100-page PDF.

---

## 4. Benchmark CLI
`ingestforge maintenance benchmark --mode [mobile|speed]`
*   Runs a standardized 5-document test suite.
*   Outputs a Markdown table comparing actual vs. target performance.
