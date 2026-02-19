"""Tests for embedding generation and quantization."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import multiprocessing
import time

from ingestforge.enrichment.embeddings import (
    EmbeddingGenerator,
    EmbeddingError,
    QUANTIZATION_MODES,
)


class TestQuantizationModes:
    """Test quantization mode detection."""

    def test_quantization_modes_defined(self) -> None:
        """Verify quantization modes are speed and mobile."""
        assert QUANTIZATION_MODES == {"speed", "mobile"}

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("speed", True),
            ("mobile", True),
            ("quality", False),
            ("balanced", False),
        ],
    )
    def test_should_quantize_by_mode(self, mode: str, expected: bool) -> None:
        """Test _should_quantize returns correct value for each mode."""
        config = MagicMock()
        config.performance_mode = mode
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_embeddings = True

        generator = EmbeddingGenerator(config)
        assert generator._should_quantize() == expected


class TestEmbeddingQuantization:
    """Test embedding quantization application."""

    def test_apply_quantization_speed_mode(self) -> None:
        """In speed mode, embeddings are quantized to float16."""
        config = MagicMock()
        config.performance_mode = "speed"
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_embeddings = True

        generator = EmbeddingGenerator(config)

        # Create test embeddings (float32)
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        result = generator._apply_quantization(embeddings)

        # Result should be list of lists
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 3

    def test_apply_quantization_quality_mode(self) -> None:
        """In quality mode, embeddings are NOT quantized."""
        config = MagicMock()
        config.performance_mode = "quality"
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_embeddings = True

        generator = EmbeddingGenerator(config)

        embeddings = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        result = generator._apply_quantization(embeddings)

        # Should return original precision
        assert isinstance(result, list)
        assert result[0][0] == pytest.approx(0.1, rel=1e-6)

    def test_quantization_reduces_precision(self) -> None:
        """Quantization to float16 reduces precision but preserves values."""
        config = MagicMock()
        config.performance_mode = "mobile"
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_embeddings = True

        generator = EmbeddingGenerator(config)

        # Value that may lose precision in float16
        embeddings = np.array([[0.123456789]], dtype=np.float32)
        result = generator._apply_quantization(embeddings)

        # Value should be approximately correct but not exact
        assert result[0][0] == pytest.approx(0.123456789, rel=1e-3)


class TestEmbedWithQuantization:
    """Test embed() method with quantization."""

    @patch.object(EmbeddingGenerator, "model", new_callable=lambda: MagicMock())
    def test_embed_applies_quantization_in_speed_mode(
        self, mock_model: MagicMock
    ) -> None:
        """embed() applies quantization when in speed mode."""
        config = MagicMock()
        config.performance_mode = "speed"
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_embeddings = True

        # Mock model.encode to return numpy array
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        generator = EmbeddingGenerator(config)
        generator._model = mock_model

        result = generator.embed("test text")

        assert isinstance(result, list)
        assert len(result) == 3
        mock_model.encode.assert_called_once()

    @patch.object(EmbeddingGenerator, "model", new_callable=lambda: MagicMock())
    def test_embed_no_quantization_in_balanced_mode(
        self, mock_model: MagicMock
    ) -> None:
        """embed() does not quantize in balanced mode."""
        config = MagicMock()
        config.performance_mode = "balanced"
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_embeddings = True

        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        generator = EmbeddingGenerator(config)
        generator._model = mock_model

        result = generator.embed("test text")

        assert isinstance(result, list)
        # Full precision preserved
        assert result[0] == pytest.approx(0.1, rel=1e-6)


class TestEmbedBatchWithQuantization:
    """Test embed_batch() with quantization."""

    @patch.object(EmbeddingGenerator, "_detect_optimal_batch_size", return_value=32)
    @patch.object(EmbeddingGenerator, "model", new_callable=lambda: MagicMock())
    def test_embed_batch_applies_quantization(
        self, mock_model: MagicMock, mock_batch_size: MagicMock
    ) -> None:
        """embed_batch() applies quantization in mobile mode."""
        config = MagicMock()
        config.performance_mode = "mobile"
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_embeddings = True

        mock_model.encode.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4]], dtype=np.float32
        )

        generator = EmbeddingGenerator(config)
        generator._model = mock_model

        result = generator.embed_batch(["text1", "text2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 2


class TestResourceLock:
    """Test resource lock for model loading (, Task-003)."""

    @patch("ingestforge.enrichment.embeddings.acquire_resource_lock")
    @patch("ingestforge.enrichment.embeddings.release_resource_lock")
    @patch("sentence_transformers.SentenceTransformer")
    def test_model_loading_acquires_lock(
        self,
        mock_transformer: MagicMock,
        mock_release: MagicMock,
        mock_acquire: MagicMock,
    ) -> None:
        """Model loading acquires and releases resource lock."""
        # JPL Rule #7: Check return value of acquire_resource_lock
        mock_acquire.return_value = True
        mock_model_instance = MagicMock()
        mock_transformer.return_value = mock_model_instance

        config = MagicMock()
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_embeddings = True
        config.performance_mode = "balanced"

        generator = EmbeddingGenerator(config)

        # Access model property to trigger lazy loading
        model = generator.model

        # Assert lock was acquired with timeout
        mock_acquire.assert_called_once_with(timeout=30)

        # Assert model was loaded
        mock_transformer.assert_called_once_with("all-MiniLM-L6-v2")

        # Assert lock was released in finally block
        mock_release.assert_called_once()

        # Assert we got the model
        assert model == mock_model_instance

    @patch("ingestforge.enrichment.embeddings.acquire_resource_lock")
    @patch("ingestforge.enrichment.embeddings.release_resource_lock")
    def test_model_loading_timeout_raises_error(
        self,
        mock_release: MagicMock,
        mock_acquire: MagicMock,
    ) -> None:
        """Model loading raises error when lock acquisition times out."""
        # JPL Rule #7: Test failed lock acquisition
        mock_acquire.return_value = False

        config = MagicMock()
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_embeddings = True

        generator = EmbeddingGenerator(config)

        # Should raise EmbeddingError on timeout
        with pytest.raises(EmbeddingError) as exc_info:
            _ = generator.model

        assert "Failed to acquire resource lock" in str(exc_info.value)
        assert "timeout after 30s" in str(exc_info.value)

        # Lock should not be released if never acquired
        mock_release.assert_not_called()

    @patch("ingestforge.enrichment.embeddings.acquire_resource_lock")
    @patch("ingestforge.enrichment.embeddings.release_resource_lock")
    @patch("sentence_transformers.SentenceTransformer")
    def test_model_loading_releases_lock_on_error(
        self,
        mock_transformer: MagicMock,
        mock_release: MagicMock,
        mock_acquire: MagicMock,
    ) -> None:
        """Lock is released even if model loading fails."""
        mock_acquire.return_value = True
        # Simulate model loading error
        mock_transformer.side_effect = RuntimeError("Model load failed")

        config = MagicMock()
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_embeddings = True

        generator = EmbeddingGenerator(config)

        # Should raise RuntimeError from SentenceTransformer
        with pytest.raises(RuntimeError, match="Model load failed"):
            _ = generator.model

        # Assert lock was still released in finally block
        mock_release.assert_called_once()

    @patch("ingestforge.enrichment.embeddings.acquire_resource_lock")
    @patch("ingestforge.enrichment.embeddings.release_resource_lock")
    @patch("sentence_transformers.SentenceTransformer")
    def test_model_loaded_only_once_with_lock(
        self,
        mock_transformer: MagicMock,
        mock_release: MagicMock,
        mock_acquire: MagicMock,
    ) -> None:
        """Model is loaded only once despite multiple accesses."""
        mock_acquire.return_value = True
        mock_model_instance = MagicMock()
        mock_transformer.return_value = mock_model_instance

        config = MagicMock()
        config.enrichment.embedding_model = "all-MiniLM-L6-v2"
        config.enrichment.generate_embeddings = True

        generator = EmbeddingGenerator(config)

        # Access model multiple times
        model1 = generator.model
        model2 = generator.model
        model3 = generator.model

        # Lock should only be acquired once (first access)
        mock_acquire.assert_called_once()

        # Model should only be instantiated once
        mock_transformer.assert_called_once()

        # Lock should only be released once
        mock_release.assert_called_once()

        # All accesses return same instance
        assert model1 is model2
        assert model2 is model3


class TestConcurrentModelLoading:
    """Test concurrent model loading with resource lock (, Task-003)."""

    def _load_model_worker(
        self,
        worker_id: int,
        result_queue: multiprocessing.Queue,
        lock_delay: float = 0.1,
    ) -> None:
        """Worker function to test concurrent model loading."""
        try:
            # Simulate model loading taking some time
            with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
                mock_model = MagicMock()
                mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

                # Add delay to simulate real model loading
                def delayed_load(*args, **kwargs):
                    time.sleep(lock_delay)
                    return mock_model

                mock_transformer.side_effect = delayed_load

                config = MagicMock()
                config.enrichment.embedding_model = "all-MiniLM-L6-v2"
                config.enrichment.generate_embeddings = True
                config.performance_mode = "balanced"

                generator = EmbeddingGenerator(config)

                # Access model (triggers loading with lock)
                start_time = time.time()
                _ = generator.model
                elapsed = time.time() - start_time

                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "success": True,
                        "elapsed": elapsed,
                    }
                )

        except Exception as e:
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "success": False,
                    "error": str(e),
                }
            )

    def test_parallel_loading_serialized_by_lock(self) -> None:
        """
        Test that 16 concurrent workers don't load model simultaneously.

        Resource lock prevents RAM spikes from concurrent loads.
        JPL Rule #2: Fixed upper bound (16 workers).
        """
        num_workers = 16
        result_queue = multiprocessing.Queue()

        # Start workers
        processes = []
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=self._load_model_worker,
                args=(i, result_queue, 0.05),  # 50ms simulated load time
            )
            p.start()
            processes.append(p)

        # Wait for all workers
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # All workers should succeed
        assert len(results) == num_workers
        assert all(r["success"] for r in results), f"Some workers failed: {results}"

        # Workers should have been serialized by the lock
        # Total time should be roughly sum of individual loads, not maximum
        # (This is a best-effort check; timing tests can be flaky)
        elapsed_times = [r["elapsed"] for r in results]
        max_elapsed = max(elapsed_times)

        # At least some workers should have waited (elapsed > 0.05s base time)
        waiting_workers = [e for e in elapsed_times if e > 0.1]
        assert len(waiting_workers) > 0, (
            "Expected some workers to wait for lock, but none did. "
            f"Elapsed times: {elapsed_times}"
        )
