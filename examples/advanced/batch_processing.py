"""
Example: Batch Document Processing

Description:
    Demonstrates parallel processing of large document collections
    with progress tracking, error recovery, and result aggregation.
    Useful for bulk ingestion projects.

Usage:
    python examples/advanced/batch_processing.py \
        --input documents/ \
        --batch-size 100 \
        --workers 4 \
        --output results.jsonl

    python examples/advanced/batch_processing.py \
        --input documents/ \
        --resume-from checkpoint.json

Expected output:
    - Processed chunks saved to output
    - Progress logs and statistics
    - Error reports
    - Checkpoint for resume capability

Requirements:
    - Multiple documents in input directory
    - Optional: multiprocessing setup for parallel processing
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from ingestforge.ingest.processor import DocumentProcessor
from ingestforge.chunking.semantic_chunker import SemanticChunker
from ingestforge.storage.jsonl import JSONLStorage


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""

    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    total_bytes: int = 0
    start_time: float = 0
    end_time: float = 0

    @property
    def elapsed_time(self) -> float:
        """Elapsed time in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0

    @property
    def files_per_second(self) -> float:
        """Processing speed."""
        if self.elapsed_time > 0:
            return self.processed_files / self.elapsed_time
        return 0


def process_document(
    file_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> tuple[Path, list[dict], Optional[str]]:
    """
    Process a single document (worker function).

    Args:
        file_path: Document path
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        (file_path, chunks, error) tuple
    """
    try:
        processor = DocumentProcessor()
        text = processor.process(file_path)

        chunker = SemanticChunker(target_size=chunk_size, overlap=chunk_overlap)
        chunks = chunker.chunk(text)

        # Add source metadata
        for chunk in chunks:
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            chunk["metadata"]["source"] = str(file_path)
            chunk["metadata"]["source_name"] = file_path.name

        return file_path, chunks, None

    except Exception as e:
        return file_path, [], str(e)


def batch_process(
    input_dir: str,
    output_path: str = "results.jsonl",
    batch_size: int = 100,
    num_workers: int = 4,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    resume_from: Optional[str] = None,
) -> None:
    """
    Batch process documents in parallel.

    Args:
        input_dir: Directory containing documents
        output_path: Output JSONL file
        batch_size: Documents per batch
        num_workers: Number of parallel workers
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        resume_from: Checkpoint file to resume from
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return

    # Find documents
    print(f"\n[*] Finding documents in {input_dir}...")
    doc_files = (
        list(input_dir.glob("**/*.pdf"))
        + list(input_dir.glob("**/*.txt"))
        + list(input_dir.glob("**/*.md"))
        + list(input_dir.glob("**/*.docx"))
    )

    if not doc_files:
        print("[!] No documents found")
        return

    doc_files = sorted(doc_files)
    print(f"[✓] Found {len(doc_files)} documents")

    # Load checkpoint if resuming
    stats = ProcessingStats(total_files=len(doc_files))
    processed_paths = set()
    all_chunks = []
    errors = []

    if resume_from and Path(resume_from).exists():
        print(f"\n[*] Loading checkpoint: {resume_from}")
        try:
            with open(resume_from, "r") as f:
                checkpoint = json.load(f)
                processed_paths = set(checkpoint.get("processed_files", []))
                stats.processed_files = len(processed_paths)
                stats.failed_files = len(checkpoint.get("errors", []))
                all_chunks = checkpoint.get("chunks", [])
                errors = checkpoint.get("errors", [])

                print(f"    Resuming from: {stats.processed_files} processed files")

        except Exception as e:
            print(f"[!] Error loading checkpoint: {e}")

    # Filter out already processed
    remaining_files = [f for f in doc_files if str(f) not in processed_paths]

    print(f"\n[*] Processing {len(remaining_files)} remaining documents")
    print(f"    Batch size: {batch_size}")
    print(f"    Workers: {num_workers}")

    stats.start_time = time.time()

    # Process in batches
    for batch_num, batch_start in enumerate(
        range(0, len(remaining_files), batch_size), 1
    ):
        batch_end = min(batch_start + batch_size, len(remaining_files))
        batch_files = remaining_files[batch_start:batch_end]

        print(f"\n[*] Batch {batch_num}: Processing {len(batch_files)} documents")

        # Process in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_document,
                    f,
                    chunk_size,
                    chunk_overlap,
                ): f
                for f in batch_files
            }

            for i, future in enumerate(as_completed(futures), 1):
                file_path, chunks, error = future.result()

                if error:
                    status = f"ERROR: {error[:50]}"
                    stats.failed_files += 1
                    errors.append(
                        {
                            "file": str(file_path),
                            "error": error,
                        }
                    )
                else:
                    status = f"OK ({len(chunks)} chunks)"
                    stats.processed_files += 1
                    stats.total_chunks += len(chunks)
                    stats.total_bytes += sum(len(c["text"]) for c in chunks)
                    all_chunks.extend(chunks)

                # Print progress
                elapsed = time.time() - stats.start_time
                speed = stats.processed_files / elapsed if elapsed > 0 else 0

                print(
                    f"    [{i}/{len(batch_files)}] {file_path.name:<40} {status:<40} ({speed:.1f} files/sec)"
                )

        # Save checkpoint
        checkpoint_path = output_path.with_suffix(".checkpoint")
        save_checkpoint(
            checkpoint_path,
            {
                "processed_files": list(processed_paths)
                + [str(f) for f in batch_files],
                "chunks": all_chunks,
                "errors": errors,
                "stats": {
                    "processed": stats.processed_files,
                    "failed": stats.failed_files,
                    "total_chunks": stats.total_chunks,
                    "total_bytes": stats.total_bytes,
                },
            },
        )

    stats.end_time = time.time()

    # Save final results
    print(f"\n[*] Saving {len(all_chunks)} chunks to {output_path}...")
    storage = JSONLStorage(str(output_path))
    storage.save(all_chunks)

    # Print summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}\n")

    print(f"Total files:        {stats.total_files}")
    print(f"Processed:          {stats.processed_files}")
    print(f"Failed:             {stats.failed_files}")
    print(f"Total chunks:       {stats.total_chunks}")
    print(f"Total data:         {stats.total_bytes / (1024*1024):.2f} MB")
    print(f"Processing time:    {stats.elapsed_time:.2f} seconds")
    print(f"Speed:              {stats.files_per_second:.2f} files/sec")
    print(f"\nOutput:             {output_path}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors[:5]:
            print(f"  - {error['file']}: {error['error'][:60]}")

        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    print("\n[✓] Batch processing complete!")


def save_checkpoint(path: Path, data: dict) -> None:
    """Save processing checkpoint."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch process documents in parallel")
    parser.add_argument(
        "--input", "-i", required=True, help="Input directory with documents"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results.jsonl",
        help="Output file (default: results.jsonl)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Documents per batch (default: 100)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of workers (default: 4)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size (default: 512)"
    )
    parser.add_argument("--resume-from", help="Resume from checkpoint file")

    args = parser.parse_args()

    batch_process(
        args.input,
        args.output,
        args.batch_size,
        args.workers,
        args.chunk_size,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
