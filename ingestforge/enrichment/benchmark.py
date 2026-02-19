"""
GPU acceleration benchmarks for embedding generation.

Compares CPU vs GPU performance to demonstrate speedup.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.enrichment.embeddings import VRAMInfo

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    device: str
    device_name: str
    batch_size: int
    num_texts: int
    total_time_sec: float
    texts_per_second: float
    avg_time_per_text_ms: float

    def __str__(self) -> str:
        return (
            f"{self.device} ({self.device_name}): "
            f"{self.texts_per_second:.1f} texts/sec, "
            f"{self.avg_time_per_text_ms:.2f} ms/text"
        )


@dataclass
class BenchmarkComparison:
    """Comparison of GPU vs CPU performance."""

    cpu_result: BenchmarkResult
    gpu_result: Optional[BenchmarkResult]
    speedup: float  # GPU speedup factor (e.g., 10.5x)

    def __str__(self) -> str:
        if self.gpu_result:
            return (
                f"CPU: {self.cpu_result.texts_per_second:.1f} texts/sec\n"
                f"GPU: {self.gpu_result.texts_per_second:.1f} texts/sec\n"
                f"Speedup: {self.speedup:.1f}x"
            )
        return f"CPU only: {self.cpu_result.texts_per_second:.1f} texts/sec"


def generate_sample_texts(num_texts: int = 100) -> List[str]:
    """
    Generate sample texts for benchmarking.

    Args:
        num_texts: Number of texts to generate

    Returns:
        List of sample texts (varied lengths)
    """
    samples = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "Document processing involves extracting text, images, and metadata from various file formats.",
        "Semantic search uses embeddings to find documents based on meaning rather than exact keywords.",
        "Neural networks are computing systems inspired by biological neural networks in the brain.",
        "Natural language processing enables computers to understand and generate human language.",
        "Vector databases store embeddings and enable fast similarity search across millions of documents.",
        "Retrieval augmented generation combines search with language models for accurate responses.",
    ]

    # Repeat and vary the samples
    texts = []
    for i in range(num_texts):
        base = samples[i % len(samples)]
        # Add variation
        texts.append(f"{base} (Sample {i+1})")

    return texts


def benchmark_device(
    device: str,
    model_name: str = "all-MiniLM-L6-v2",
    num_texts: int = 100,
    batch_size: int = 32,
    warmup: bool = True,
) -> BenchmarkResult:
    """
    Benchmark embedding generation on a specific device.

    Rule #4: Reduced from 66 → 42 lines via helper extraction
    """
    _check_sentence_transformers()
    from sentence_transformers import SentenceTransformer

    # Load model and get device name
    model = SentenceTransformer(model_name, device=device)
    device_name = _get_device_name(device)

    # Generate sample texts and warmup
    texts = generate_sample_texts(num_texts)
    if warmup:
        model.encode(texts[:batch_size], batch_size=batch_size)

    # Benchmark encoding performance
    start_time = time.perf_counter()
    model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    end_time = time.perf_counter()

    # Calculate metrics
    total_time = end_time - start_time
    texts_per_sec = num_texts / total_time
    ms_per_text = (total_time * 1000) / num_texts

    return _create_benchmark_result(
        device,
        device_name,
        batch_size,
        num_texts,
        total_time,
        texts_per_sec,
        ms_per_text,
    )


def _check_sentence_transformers() -> None:
    """
    Check sentence-transformers is installed.

    Rule #4: Extracted to reduce benchmark_device() size
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "sentence-transformers required for benchmarks. "
            "Install with: pip install sentence-transformers"
        )


def _get_device_name(device: str) -> str:
    """
    Get friendly device name for benchmark results.

    Rule #4: Extracted to reduce benchmark_device() size
    Rule #5: Log exception instead of silent swallow
    """
    if device != "cuda":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception as e:
        logger.debug(f"Failed to get CUDA device name: {e}")

    return device


def _create_benchmark_result(
    device: str,
    device_name: str,
    batch_size: int,
    num_texts: int,
    total_time: float,
    texts_per_sec: float,
    ms_per_text: float,
) -> BenchmarkResult:
    """
    Create benchmark result object.

    Rule #4: Extracted to reduce benchmark_device() size
    """
    return BenchmarkResult(
        device=device,
        device_name=device_name,
        batch_size=batch_size,
        num_texts=num_texts,
        total_time_sec=total_time,
        texts_per_second=texts_per_sec,
        avg_time_per_text_ms=ms_per_text,
    )


def run_comparison_benchmark(
    model_name: str = "all-MiniLM-L6-v2",
    num_texts: int = 500,
    cpu_batch_size: int = 32,
    gpu_batch_size: Optional[int] = None,
) -> BenchmarkComparison:
    """
    Run CPU vs GPU comparison benchmark.

    Rule #4: Reduced from 64 → 37 lines via helper extraction
    """
    logger.info(f"Running benchmark with {num_texts} texts...")
    vram_info = VRAMInfo.detect()
    gpu_batch_size = _detect_gpu_batch_size(gpu_batch_size, vram_info)

    # Benchmark CPU
    logger.info("Benchmarking CPU...")
    cpu_result = benchmark_device(
        device="cpu",
        model_name=model_name,
        num_texts=num_texts,
        batch_size=cpu_batch_size,
    )
    logger.info(f"CPU: {cpu_result}")
    gpu_result, speedup = _benchmark_gpu_if_available(
        vram_info, model_name, num_texts, gpu_batch_size, cpu_result
    )

    return BenchmarkComparison(
        cpu_result=cpu_result,
        gpu_result=gpu_result,
        speedup=speedup,
    )


def _detect_gpu_batch_size(
    gpu_batch_size: Optional[int], vram_info: VRAMInfo
) -> Optional[int]:
    """
    Auto-detect optimal GPU batch size.

    Rule #4: Extracted to reduce run_comparison_benchmark() size
    """
    if gpu_batch_size is None and vram_info.available:
        from ingestforge.enrichment.embeddings import calculate_optimal_batch_size

        return calculate_optimal_batch_size(vram_info.free_gb)
    return gpu_batch_size


def _benchmark_gpu_if_available(
    vram_info: VRAMInfo,
    model_name: str,
    num_texts: int,
    gpu_batch_size: Optional[int],
    cpu_result: BenchmarkResult,
) -> tuple:
    """
    Benchmark GPU if available.

    Rule #4: Extracted to reduce run_comparison_benchmark() size

    Returns:
        Tuple of (gpu_result, speedup)
    """
    if not vram_info.available:
        logger.info("No GPU available, CPU-only benchmark")
        return None, 1.0

    logger.info(f"Benchmarking GPU ({vram_info.device_name})...")
    try:
        gpu_result = benchmark_device(
            device="cuda",
            model_name=model_name,
            num_texts=num_texts,
            batch_size=gpu_batch_size or 64,
        )
        speedup = gpu_result.texts_per_second / cpu_result.texts_per_second
        logger.info(f"GPU: {gpu_result}")
        logger.info(f"Speedup: {speedup:.1f}x")
        return gpu_result, speedup
    except Exception as e:
        logger.warning(f"GPU benchmark failed: {e}")
        return None, 1.0


def print_benchmark_report(comparison: BenchmarkComparison) -> str:
    """
    Generate a formatted benchmark report.

    Args:
        comparison: Benchmark comparison results

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "EMBEDDING BENCHMARK REPORT",
        "=" * 60,
        "",
        "Model: all-MiniLM-L6-v2",
        f"Texts: {comparison.cpu_result.num_texts}",
        "",
        "RESULTS:",
        "-" * 40,
        "",
        f"CPU ({comparison.cpu_result.device_name}):",
        f"  Batch size: {comparison.cpu_result.batch_size}",
        f"  Total time: {comparison.cpu_result.total_time_sec:.2f}s",
        f"  Throughput: {comparison.cpu_result.texts_per_second:.1f} texts/sec",
        f"  Latency: {comparison.cpu_result.avg_time_per_text_ms:.2f} ms/text",
        "",
    ]

    if comparison.gpu_result:
        lines.extend(
            [
                f"GPU ({comparison.gpu_result.device_name}):",
                f"  Batch size: {comparison.gpu_result.batch_size}",
                f"  Total time: {comparison.gpu_result.total_time_sec:.2f}s",
                f"  Throughput: {comparison.gpu_result.texts_per_second:.1f} texts/sec",
                f"  Latency: {comparison.gpu_result.avg_time_per_text_ms:.2f} ms/text",
                "",
                "-" * 40,
                f"SPEEDUP: {comparison.speedup:.1f}x faster on GPU",
            ]
        )

        if comparison.speedup >= 10:
            lines.append("✓ Achieved 10x+ speedup target!")
        elif comparison.speedup >= 5:
            lines.append("Good speedup, consider larger batch sizes")
        else:
            lines.append("Lower speedup may be due to small dataset or model")
    else:
        lines.extend(
            [
                "GPU: Not available",
                "",
                "Install PyTorch with CUDA support for GPU acceleration:",
                "  pip install torch --index-url https://download.pytorch.org/whl/cu121",
            ]
        )

    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    # Run benchmark when executed directly
    comparison = run_comparison_benchmark(num_texts=500)
    print(print_benchmark_report(comparison))
