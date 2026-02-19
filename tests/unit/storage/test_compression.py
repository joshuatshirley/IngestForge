"""Tests for storage compression utilities."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from ingestforge.storage.compression import (
    read_jsonl,
    write_jsonl,
    append_jsonl,
    quantize_embeddings,
    dequantize_embeddings,
    get_compression_stats,
)


class TestReadJsonl:
    """Tests for read_jsonl function."""

    def test_read_uncompressed_jsonl(self, tmp_path: Path):
        """Test reading uncompressed JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"
        records = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]

        with open(jsonl_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        result = list(read_jsonl(jsonl_file))
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["name"] == "test2"

    def test_read_compressed_jsonl(self, tmp_path: Path):
        """Test reading compressed JSONL.gz file."""
        jsonl_gz_file = tmp_path / "test.jsonl.gz"
        records = [{"id": 1}, {"id": 2}, {"id": 3}]

        with gzip.open(jsonl_gz_file, "wt") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        result = list(read_jsonl(jsonl_gz_file))
        assert len(result) == 3
        assert result[0]["id"] == 1

    def test_read_gz_suffix(self, tmp_path: Path):
        """Test reading file with .gz suffix."""
        gz_file = tmp_path / "data.gz"
        record = {"test": "value"}

        with gzip.open(gz_file, "wt") as f:
            f.write(json.dumps(record) + "\n")

        result = list(read_jsonl(gz_file))
        assert len(result) == 1
        assert result[0]["test"] == "value"

    def test_read_empty_lines_skipped(self, tmp_path: Path):
        """Test empty lines are skipped."""
        jsonl_file = tmp_path / "test.jsonl"

        with open(jsonl_file, "w") as f:
            f.write('{"id": 1}\n')
            f.write("\n")
            f.write("   \n")
            f.write('{"id": 2}\n')

        result = list(read_jsonl(jsonl_file))
        assert len(result) == 2

    def test_read_unicode(self, tmp_path: Path):
        """Test reading Unicode characters."""
        jsonl_file = tmp_path / "test.jsonl"
        record = {"text": "Hello 世界 café"}

        with open(jsonl_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        result = list(read_jsonl(jsonl_file))
        assert result[0]["text"] == "Hello 世界 café"


class TestWriteJsonl:
    """Tests for write_jsonl function."""

    def test_write_uncompressed(self, tmp_path: Path):
        """Test writing uncompressed JSONL."""
        jsonl_file = tmp_path / "output.jsonl"
        records = [{"a": 1}, {"b": 2}]

        write_jsonl(jsonl_file, records, compress=False)

        assert jsonl_file.exists()

        with open(jsonl_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0])["a"] == 1

    def test_write_compressed(self, tmp_path: Path):
        """Test writing compressed JSONL."""
        jsonl_file = tmp_path / "output.jsonl"
        records = [{"x": 1}, {"y": 2}, {"z": 3}]

        write_jsonl(jsonl_file, records, compress=True)

        # Should add .gz extension
        gz_file = tmp_path / "output.jsonl.gz"
        assert gz_file.exists()

        with gzip.open(gz_file, "rt") as f:
            lines = f.readlines()

        assert len(lines) == 3

    def test_write_compressed_explicit_gz_extension(self, tmp_path: Path):
        """Test writing compressed with .gz already in path."""
        jsonl_gz_file = tmp_path / "output.jsonl.gz"
        records = [{"test": "value"}]

        write_jsonl(jsonl_gz_file, records, compress=True)

        assert jsonl_gz_file.exists()
        # Should not double up on .gz extension

    def test_write_unicode(self, tmp_path: Path):
        """Test writing Unicode characters."""
        jsonl_file = tmp_path / "unicode.jsonl"
        records = [{"text": "Hello 世界"}]

        write_jsonl(jsonl_file, records, compress=False)

        result = list(read_jsonl(jsonl_file))
        assert result[0]["text"] == "Hello 世界"

    def test_write_empty_list(self, tmp_path: Path):
        """Test writing empty list."""
        jsonl_file = tmp_path / "empty.jsonl"

        write_jsonl(jsonl_file, [], compress=False)

        assert jsonl_file.exists()
        assert jsonl_file.stat().st_size == 0


class TestAppendJsonl:
    """Tests for append_jsonl function."""

    def test_append_uncompressed(self, tmp_path: Path):
        """Test appending to uncompressed file."""
        jsonl_file = tmp_path / "append.jsonl"

        append_jsonl(jsonl_file, {"id": 1}, compress=False)
        append_jsonl(jsonl_file, {"id": 2}, compress=False)

        result = list(read_jsonl(jsonl_file))
        assert len(result) == 2
        assert result[1]["id"] == 2

    def test_append_compressed(self, tmp_path: Path):
        """Test appending to compressed file."""
        jsonl_file = tmp_path / "append.jsonl"

        append_jsonl(jsonl_file, {"a": 1}, compress=True)
        append_jsonl(jsonl_file, {"b": 2}, compress=True)

        gz_file = tmp_path / "append.jsonl.gz"
        result = list(read_jsonl(gz_file))
        assert len(result) == 2

    def test_append_to_existing_file(self, tmp_path: Path):
        """Test appending to existing file."""
        jsonl_file = tmp_path / "existing.jsonl"

        # Create initial file
        with open(jsonl_file, "w") as f:
            f.write('{"initial": true}\n')

        # Append
        append_jsonl(jsonl_file, {"appended": True}, compress=False)

        result = list(read_jsonl(jsonl_file))
        assert len(result) == 2
        assert result[0]["initial"] is True
        assert result[1]["appended"] is True


class TestEmbeddingQuantization:
    """Tests for embedding quantization functions."""

    def test_quantize_float32_to_float16(self):
        """Test quantizing float32 embeddings to float16."""
        embeddings = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)

        quantized = quantize_embeddings(embeddings, dtype="float16")

        assert quantized.dtype == np.float16
        assert quantized.shape == embeddings.shape

    def test_quantize_preserves_values_approximately(self):
        """Test quantization preserves values (with some precision loss)."""
        embeddings = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        quantized = quantize_embeddings(embeddings, dtype="float16")

        # Values should be close (not exact due to precision loss)
        np.testing.assert_allclose(quantized, embeddings, rtol=0.001)

    def test_quantize_unsupported_dtype(self):
        """Test quantization with unsupported dtype raises error."""
        embeddings = np.array([[1.0, 2.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="Unsupported quantization dtype"):
            quantize_embeddings(embeddings, dtype="int8")

    def test_dequantize_float16_to_float32(self):
        """Test dequantizing float16 back to float32."""
        embeddings = np.array([[1.5, 2.5]], dtype=np.float16)

        dequantized = dequantize_embeddings(embeddings)

        assert dequantized.dtype == np.float32
        assert dequantized.shape == embeddings.shape

    def test_quantize_dequantize_roundtrip(self):
        """Test quantize then dequantize preserves values."""
        original = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        quantized = quantize_embeddings(original)
        dequantized = dequantize_embeddings(quantized)

        # Should be close to original (minor precision loss)
        np.testing.assert_allclose(dequantized, original, rtol=0.001)

    def test_quantize_large_array(self):
        """Test quantization on larger array."""
        # 100 embeddings of dimension 384
        embeddings = np.random.randn(100, 384).astype(np.float32)

        quantized = quantize_embeddings(embeddings)

        assert quantized.dtype == np.float16
        assert quantized.shape == (100, 384)

        # Check memory savings (should be ~50%)
        original_bytes = embeddings.nbytes
        quantized_bytes = quantized.nbytes
        assert quantized_bytes < original_bytes * 0.55  # ~50% reduction


class TestCompressionStats:
    """Tests for compression statistics."""

    def test_stats_uncompressed_file(self, tmp_path: Path):
        """Test stats for uncompressed file."""
        jsonl_file = tmp_path / "data.jsonl"
        records = [{"test": "value"}] * 10

        write_jsonl(jsonl_file, records, compress=False)

        stats = get_compression_stats(jsonl_file)

        assert stats["path"] == str(jsonl_file)
        assert stats["compressed"] is False
        assert stats["file_size_bytes"] > 0
        assert stats["compression_ratio"] == 1.0

    def test_stats_compressed_file(self, tmp_path: Path):
        """Test stats for compressed file."""
        jsonl_file = tmp_path / "data.jsonl"
        records = [{"test": "value" * 100}] * 50  # Make it compressible

        write_jsonl(jsonl_file, records, compress=True)

        gz_file = tmp_path / "data.jsonl.gz"
        stats = get_compression_stats(gz_file)

        assert stats["compressed"] is True
        assert stats["file_size_bytes"] > 0
        assert stats["uncompressed_size_bytes"] > stats["file_size_bytes"]
        assert 0 < stats["compression_ratio"] < 1.0

    def test_stats_missing_file(self, tmp_path: Path):
        """Test stats for non-existent file."""
        missing_file = tmp_path / "missing.jsonl"

        stats = get_compression_stats(missing_file)

        assert stats["file_size_bytes"] == 0
        assert stats["uncompressed_size_bytes"] == 0

    def test_compression_ratio_calculation(self, tmp_path: Path):
        """Test compression ratio is calculated correctly."""
        # Create highly compressible data
        jsonl_file = tmp_path / "repeat.jsonl"
        records = [{"data": "a" * 1000}] * 100

        write_jsonl(jsonl_file, records, compress=True)

        gz_file = tmp_path / "repeat.jsonl.gz"
        stats = get_compression_stats(gz_file)

        # Should have good compression ratio
        assert stats["compression_ratio"] < 0.1  # Less than 10% of original


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_write_read_roundtrip_uncompressed(self, tmp_path: Path):
        """Test write and read roundtrip."""
        jsonl_file = tmp_path / "roundtrip.jsonl"
        records = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
        ]

        write_jsonl(jsonl_file, records, compress=False)
        result = list(read_jsonl(jsonl_file))

        assert result == records

    def test_write_read_roundtrip_compressed(self, tmp_path: Path):
        """Test write and read roundtrip with compression."""
        jsonl_file = tmp_path / "compressed.jsonl"
        records = [{"value": i} for i in range(20)]

        write_jsonl(jsonl_file, records, compress=True)

        gz_file = tmp_path / "compressed.jsonl.gz"
        result = list(read_jsonl(gz_file))

        assert result == records

    def test_append_multiple_then_read(self, tmp_path: Path):
        """Test multiple appends then read."""
        jsonl_file = tmp_path / "multi_append.jsonl"

        for i in range(5):
            append_jsonl(jsonl_file, {"num": i}, compress=False)

        result = list(read_jsonl(jsonl_file))

        assert len(result) == 5
        assert [r["num"] for r in result] == [0, 1, 2, 3, 4]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_read_malformed_json(self, tmp_path: Path):
        """Test reading file with malformed JSON."""
        jsonl_file = tmp_path / "malformed.jsonl"

        with open(jsonl_file, "w") as f:
            f.write('{"valid": true}\n')
            f.write("{invalid json\n")
            f.write('{"also_valid": true}\n')

        with pytest.raises(json.JSONDecodeError):
            list(read_jsonl(jsonl_file))

    def test_write_complex_data(self, tmp_path: Path):
        """Test writing complex nested data."""
        jsonl_file = tmp_path / "complex.jsonl"
        records = [
            {
                "nested": {"deep": {"value": 123}},
                "list": [1, 2, 3],
                "mixed": [{"a": 1}, {"b": 2}],
            }
        ]

        write_jsonl(jsonl_file, records, compress=False)
        result = list(read_jsonl(jsonl_file))

        assert result == records

    def test_empty_embedding_array(self):
        """Test quantization of empty array."""
        embeddings = np.array([], dtype=np.float32).reshape(0, 0)

        quantized = quantize_embeddings(embeddings)

        assert quantized.shape == embeddings.shape

    def test_single_embedding(self):
        """Test single embedding vector."""
        embedding = np.array([[0.5, 0.6, 0.7]], dtype=np.float32)

        quantized = quantize_embeddings(embedding)
        dequantized = dequantize_embeddings(quantized)

        assert quantized.shape == (1, 3)
        np.testing.assert_allclose(dequantized[0], embedding[0], rtol=0.01)
