"""Compression utilities for storage optimization.

Provides transparent gzip compression for JSONL files and
float16 embedding quantization for memory savings.
"""

import gzip
import json
from pathlib import Path
from typing import List, Iterator, Any, cast

import numpy as np
import numpy.typing as npt


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """
    Read a JSONL file, transparently handling gzip compression.

    Supports both .jsonl and .jsonl.gz files.

    Args:
        path: Path to the JSONL or JSONL.gz file

    Yields:
        Parsed JSON objects
    """

    def _read_lines(file_handle: Any) -> Iterator[dict[str, Any]]:
        for line in file_handle:
            line = line.strip()
            if line:
                yield json.loads(line)

    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from _read_lines(f)
    else:
        with open(path, "rt", encoding="utf-8") as f:
            yield from _read_lines(f)


def write_jsonl(
    path: Path, records: List[dict[str, Any]], compress: bool = False
) -> None:
    """
    Write records to a JSONL file, optionally with gzip compression.

    Args:
        path: Output file path
        records: List of dicts to write
        compress: Whether to use gzip compression
    """

    def _write_records(file_handle: Any) -> None:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if compress:
        if not str(path).endswith(".gz"):
            path = Path(str(path) + ".gz")
        with gzip.open(path, "wt", encoding="utf-8") as f:
            _write_records(f)
    else:
        with open(path, "wt", encoding="utf-8") as f:
            _write_records(f)


def append_jsonl(path: Path, record: dict[str, Any], compress: bool = False) -> None:
    """
    Append a single record to a JSONL file.

    Args:
        path: Path to the JSONL file
        record: Dict to append
        compress: Whether to use gzip compression
    """
    if compress:
        if not str(path).endswith(".gz"):
            path = Path(str(path) + ".gz")
        with gzip.open(path, "at", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def quantize_embeddings(
    embeddings: npt.NDArray[Any],
    dtype: str = "float16",
) -> npt.NDArray[Any]:
    """
    Quantize embeddings to a smaller data type for memory savings.

    float32 -> float16 gives ~50% memory reduction with minimal quality loss
    for similarity search.

    Args:
        embeddings: Input embeddings (float32)
        dtype: Target data type ("float16")

    Returns:
        Quantized embeddings
    """
    if dtype == "float16":
        return embeddings.astype(np.float16)
    else:
        raise ValueError(f"Unsupported quantization dtype: {dtype}")


def dequantize_embeddings(embeddings: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Convert quantized embeddings back to float32 for computation.

    Args:
        embeddings: Quantized embeddings

    Returns:
        Float32 embeddings
    """
    return embeddings.astype(np.float32)


def get_compression_stats(path: Path) -> dict[str, Any]:
    """
    Get compression statistics for a file.

    Args:
        path: Path to the file

    Returns:
        Dict with size info and compression ratio
    """
    stats = {
        "path": str(path),
        "compressed": str(path).endswith(".gz"),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
    }

    if stats["compressed"] and path.exists():
        # Estimate uncompressed size
        with gzip.open(path, "rb") as f:
            content = f.read()
            uncompressed_size = len(content)
            file_size = cast(int, stats["file_size_bytes"])
            stats["uncompressed_size_bytes"] = uncompressed_size
            stats["compression_ratio"] = (
                file_size / uncompressed_size if uncompressed_size > 0 else 1.0
            )
    else:
        stats["uncompressed_size_bytes"] = stats["file_size_bytes"]
        stats["compression_ratio"] = 1.0

    return stats
