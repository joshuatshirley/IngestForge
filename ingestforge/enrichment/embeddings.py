"""
Embedding generation for chunks.

Uses sentence-transformers for fast, local embedding generation.

Migrated from IEnricher to IFProcessor interface.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.core.retry import embedding_retry
from ingestforge.core.errors import SafeErrorMessage
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFFailureArtifact
from ingestforge.core.pipeline.registry import (
    register_enricher,
    acquire_resource_lock,
    release_resource_lock,
)
from ingestforge.storage.compression import quantize_embeddings

logger = get_logger(__name__)

# Default batch sizes based on VRAM (in GB)
# These are conservative estimates for all-MiniLM-L6-v2
VRAM_BATCH_SIZES = {
    2: 8,  # 2 GB VRAM
    4: 16,  # 4 GB VRAM
    6: 32,  # 6 GB VRAM
    8: 64,  # 8 GB VRAM
    12: 96,  # 12 GB VRAM
    16: 128,  # 16 GB VRAM
    24: 256,  # 24 GB VRAM
}

# CPU fallback batch size
CPU_BATCH_SIZE = 32

# Performance modes that enable embedding quantization
QUANTIZATION_MODES = {"speed", "mobile"}

# Model memory requirements (in GB)
# Approximate RAM/VRAM needed to load each model
MODEL_MEMORY_REQUIREMENTS = {
    "all-MiniLM-L6-v2": 0.25,  # ~250 MB
    "all-mpnet-base-v2": 0.5,  # ~500 MB
    "paraphrase-multilingual-MiniLM-L12-v2": 0.5,
    "all-MiniLM-L12-v2": 0.35,
    "multi-qa-MiniLM-L6-cos-v1": 0.25,
    "msmarco-MiniLM-L-6-v3": 0.25,
    # Larger models
    "sentence-t5-base": 1.0,
    "sentence-t5-large": 2.0,
    "sentence-t5-xl": 4.0,
    "gtr-t5-base": 1.0,
    "gtr-t5-large": 2.0,
    # Default for unknown models
    "_default": 1.0,
}


@dataclass
class VRAMInfo:
    """GPU VRAM information."""

    available: bool
    total_gb: float
    free_gb: float
    device_name: str
    device_index: int

    @classmethod
    def detect(cls) -> "VRAMInfo":
        """
        Detect available GPU VRAM.

        Returns:
            VRAMInfo with current GPU status
        """
        try:
            import torch

            if torch.cuda.is_available():
                device_index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_index)
                total_mem = props.total_memory / (1024**3)

                # Try to get free memory
                try:
                    free_mem, _ = torch.cuda.mem_get_info(device_index)
                    free_gb = free_mem / (1024**3)
                except Exception:
                    # Estimate free as 80% of total if can't query
                    free_gb = total_mem * 0.8

                return cls(
                    available=True,
                    total_gb=total_mem,
                    free_gb=free_gb,
                    device_name=props.name,
                    device_index=device_index,
                )
        except ImportError as e:
            logger.debug(f"torch not available for VRAM detection: {e}")
        except Exception as e:
            logger.debug(f"VRAM detection failed: {e}")

        return cls(
            available=False,
            total_gb=0.0,
            free_gb=0.0,
            device_name="CPU",
            device_index=-1,
        )


@dataclass
class MemoryInfo:
    """System memory (RAM) information."""

    total_gb: float
    available_gb: float
    percent_used: float

    @classmethod
    def detect(cls) -> "MemoryInfo":
        """
        Detect available system RAM.

        Returns:
            MemoryInfo with current RAM status
        """
        try:
            import psutil

            mem = psutil.virtual_memory()
            return cls(
                total_gb=mem.total / (1024**3),
                available_gb=mem.available / (1024**3),
                percent_used=mem.percent,
            )
        except ImportError:
            logger.debug("psutil not available for memory detection")
            # Fallback: assume 8 GB available (conservative)
            return cls(
                total_gb=8.0,
                available_gb=4.0,
                percent_used=50.0,
            )
        except Exception as e:
            logger.debug(f"Memory detection failed: {e}")
            return cls(
                total_gb=8.0,
                available_gb=4.0,
                percent_used=50.0,
            )


def get_model_memory_requirement(model_name: str) -> float:
    """
    Get memory requirement for a model.

    Args:
        model_name: Name of the embedding model

    Returns:
        Estimated memory requirement in GB
    """
    # Check exact match
    if model_name in MODEL_MEMORY_REQUIREMENTS:
        return MODEL_MEMORY_REQUIREMENTS[model_name]

    # Check partial match (for variants like "all-MiniLM-L6-v2-fine-tuned")
    for key, value in MODEL_MEMORY_REQUIREMENTS.items():
        if key != "_default" and key in model_name:
            return value

    return MODEL_MEMORY_REQUIREMENTS["_default"]


def calculate_optimal_batch_size(
    vram_gb: float,
    model_name: str = "all-MiniLM-L6-v2",
    user_override: Optional[int] = None,
) -> int:
    """
    Calculate optimal batch size based on available VRAM.

    Args:
        vram_gb: Available VRAM in GB
        model_name: Embedding model name (for model-specific adjustments)
        user_override: User-specified batch size (takes precedence)

    Returns:
        Recommended batch size
    """
    if user_override is not None and user_override > 0:
        return user_override

    if vram_gb <= 0:
        return CPU_BATCH_SIZE

    # Find the appropriate batch size
    for threshold, batch_size in sorted(VRAM_BATCH_SIZES.items()):
        if vram_gb <= threshold:
            return batch_size

    # For very large VRAM (> 24 GB), use max batch size
    return max(VRAM_BATCH_SIZES.values())


def get_batch_size_recommendation(config: Optional[Config] = None) -> Tuple[int, str]:
    """
    Get batch size recommendation with explanation.

    Args:
        config: Optional configuration for user override

    Returns:
        Tuple of (batch_size, explanation)
    """
    vram_info = VRAMInfo.detect()

    # Check for user override in config
    user_override = None
    if config and hasattr(config.enrichment, "embedding_batch_size"):
        user_override = getattr(config.enrichment, "embedding_batch_size", None)

    if user_override is not None and user_override > 0:
        return user_override, f"Using user-configured batch size: {user_override}"

    if vram_info.available:
        batch_size = calculate_optimal_batch_size(vram_info.free_gb)
        return batch_size, (
            f"Auto-detected {vram_info.device_name} with {vram_info.free_gb:.1f} GB "
            f"free VRAM. Using batch size {batch_size}."
        )

    return CPU_BATCH_SIZE, f"No GPU detected. Using CPU batch size {CPU_BATCH_SIZE}."


class EmbeddingError(Exception):
    """Error during embedding generation."""

    pass


@register_enricher(
    capabilities=["embedding", "semantic-search"],
    priority=100,
)
class EmbeddingGenerator(IFProcessor):
    """
    Generate embeddings for text chunks.

    Uses sentence-transformers models for efficient embedding.
    Automatically adjusts batch sizes based on available VRAM.

    Implements IFProcessor interface for modular pipeline architecture.

    Registered via @register_enricher decorator.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model_name = config.enrichment.embedding_model
        self._model = None
        self._optimal_batch_size: Optional[int] = None
        self._vram_info: Optional[VRAMInfo] = None
        self._version = "1.0.0"

    def _detect_optimal_batch_size(self) -> int:
        """Detect and cache optimal batch size based on VRAM."""
        if self._optimal_batch_size is not None:
            return self._optimal_batch_size

        self._vram_info = VRAMInfo.detect()

        # Check for user override
        user_override = None
        if hasattr(self.config.enrichment, "embedding_batch_size"):
            user_override = getattr(
                self.config.enrichment, "embedding_batch_size", None
            )

        self._optimal_batch_size = calculate_optimal_batch_size(
            vram_gb=self._vram_info.free_gb if self._vram_info.available else 0,
            model_name=self.model_name,
            user_override=user_override,
        )

        if self._vram_info.available:
            logger.info(
                f"VRAM detected: {self._vram_info.device_name} with "
                f"{self._vram_info.free_gb:.1f} GB free. "
                f"Using batch size {self._optimal_batch_size}"
            )
        else:
            logger.info(
                f"No GPU detected. Using CPU batch size {self._optimal_batch_size}"
            )

        return self._optimal_batch_size

    def _check_memory_requirements(self) -> None:
        """
        Check if sufficient memory is available before loading model.

        Raises:
            EmbeddingError: If strict_memory_check is True and memory is insufficient
        """
        required_gb = get_model_memory_requirement(self.model_name)

        # Check VRAM first if GPU is available
        vram_info = VRAMInfo.detect()
        if vram_info.available:
            if vram_info.free_gb >= required_gb:
                logger.debug(
                    f"VRAM check passed: {vram_info.free_gb:.1f} GB available, "
                    f"{required_gb:.1f} GB required for {self.model_name}"
                )
                return
            else:
                logger.warning(
                    f"Insufficient VRAM for GPU loading: {vram_info.free_gb:.1f} GB available, "
                    f"{required_gb:.1f} GB required. Will fall back to CPU."
                )

        # Check system RAM
        mem_info = MemoryInfo.detect()
        if mem_info.available_gb < required_gb:
            msg = (
                f"Insufficient memory for model {self.model_name}: "
                f"{mem_info.available_gb:.1f} GB available, {required_gb:.1f} GB required"
            )
            logger.warning(msg)

            strict_check = getattr(self.config.enrichment, "strict_memory_check", False)
            if strict_check:
                raise EmbeddingError(msg)
        else:
            logger.debug(
                f"Memory check passed: {mem_info.available_gb:.1f} GB available, "
                f"{required_gb:.1f} GB required for {self.model_name}"
            )

    def _should_quantize(self) -> bool:
        """Check if embeddings should be quantized based on performance mode."""
        mode = getattr(self.config, "performance_mode", "balanced").lower()
        return mode in QUANTIZATION_MODES

    def _apply_quantization(self, embeddings: Any) -> List[List[float]]:
        """
        Apply quantization to embeddings if performance mode requires it.

        Args:
            embeddings: NumPy array of embeddings

        Returns:
            List of embedding vectors (quantized if speed/mobile mode)
        """
        if not self._should_quantize():
            return embeddings.tolist()

        quantized = quantize_embeddings(embeddings, dtype="float16")
        logger.debug(
            f"Quantized embeddings from float32 to float16 "
            f"(performance_mode={self.config.performance_mode})"
        )
        return quantized.tolist()

    @property
    def model(self) -> Any:
        """
        Lazy-load the embedding model with resource lock protection.

        Prevents concurrent model loading across workers.
        JPL Rule #7: Check lock acquisition return value.
        JPL Rule #9: Assert lock state for safety.
        """
        if self._model is None:
            # Check memory requirements before loading
            self._check_memory_requirements()

            # Acquire resource lock to prevent concurrent loads
            # JPL Rule #7: Check return value of acquire_resource_lock
            lock_acquired = acquire_resource_lock(timeout=30)

            if not lock_acquired:
                raise EmbeddingError(
                    "Failed to acquire resource lock for model loading (timeout after 30s). "
                    "Another worker may be loading a model."
                )

            # JPL Rule #9: Assert lock was acquired before proceeding
            assert lock_acquired, "Resource lock acquisition check failed"

            try:
                # Double-check pattern: another worker might have loaded the model
                # while we were waiting for the lock
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info(f"Loading embedding model: {self.model_name}")
                    self._model = SentenceTransformer(self.model_name)
                    logger.info(f"Model loaded successfully: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embeddings. "
                    "Install with: pip install sentence-transformers"
                )
            finally:
                # JPL Rule #7: Always release lock in finally block
                release_resource_lock()
                logger.debug("Released resource lock after model loading")

        return self._model

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Applies float16 quantization in speed/mobile performance modes.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        # Reshape for batch-style quantization, then flatten
        result = self._apply_quantization(embedding.reshape(1, -1))
        return result[0]

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding (auto-detected from VRAM if None)

        Returns:
            List of embedding vectors
        """
        if batch_size is None:
            batch_size = self._detect_optimal_batch_size()
        return self._embed_batch_with_retry(texts, batch_size)

    @embedding_retry
    def _embed_batch_with_retry(
        self, texts: List[str], batch_size: int
    ) -> List[List[float]]:
        """Internal batch embedding with retry for transient failures."""
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
            )
            return self._apply_quantization(embeddings)
        except Exception as e:
            logger.error(f"Embedding batch failed: {e}", batch_size=len(texts))
            raise EmbeddingError(f"Failed to generate embeddings: {e}")

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process an artifact and generate embeddings.

        Implements IFProcessor.process().

        Args:
            artifact: Input artifact (must be IFChunkArtifact)

        Returns:
            IFChunkArtifact with embedding in metadata, or IFFailureArtifact on error
        """
        # Type check
        if not isinstance(artifact, IFChunkArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-embedding-failed",
                error_message=f"EmbeddingGenerator requires IFChunkArtifact, got {type(artifact).__name__}",
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
            )

        # Skip if embeddings disabled
        if not self.config.enrichment.generate_embeddings:
            return artifact

        try:
            # Generate embedding
            embedding = self.embed(artifact.content)

            # Store in metadata
            new_metadata = dict(artifact.metadata)
            new_metadata["embedding"] = embedding

            # Return derived artifact
            return artifact.derive(
                self.processor_id,
                artifact_id=f"{artifact.artifact_id}-embedded",
                metadata=new_metadata,
            )

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-embedding-failed",
                # SEC-002: Sanitize error message
                error_message=SafeErrorMessage.sanitize(
                    e, "embedding generation", logger
                ),
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
            )

    def is_available(self) -> bool:
        """Check if embedding generator is available."""
        return self.config.enrichment.generate_embeddings

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "embedding-generator"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Functional capabilities provided by this processor."""
        return ["embedding", "semantic-search"]

    @property
    def memory_mb(self) -> int:
        """
        Estimated memory requirement in megabytes.

        Based on model size from MODEL_MEMORY_REQUIREMENTS.
        """
        required_gb = get_model_memory_requirement(self.model_name)
        # Convert GB to MB and add 100MB overhead for processing
        return int(required_gb * 1024) + 100

    def get_vram_info(self) -> Optional[VRAMInfo]:
        """Get detected VRAM info."""
        if self._vram_info is None:
            self._detect_optimal_batch_size()
        return self._vram_info

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        return self.model.get_sentence_embedding_dimension()

    def teardown(self) -> bool:
        """
        Clean up resources.

        Implements IFProcessor.teardown().
        Optional cleanup method.

        Returns:
            True if cleanup successful
        """
        self._model = None
        self._optimal_batch_size = None
        self._vram_info = None
        return True
