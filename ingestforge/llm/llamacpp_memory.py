"""
LlamaCpp Memory and GPU Management Mixin.

Handles VRAM allocation, GPU layer optimization, and OOM recovery for llama.cpp.

This module is part of the llamacpp refactoring (Sprint 3, Rule #4)
to reduce llamacpp.py from 981 lines to <500 lines.
"""

from typing import Any, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class LlamaCppMemoryMixin:
    """
    Mixin providing memory and GPU management methods for LlamaCppClient.

    Rule #4: Extracted from llamacpp.py to reduce file size
    """

    def _auto_detect_gpu_layers(self) -> None:
        """
        Auto-detect optimal GPU layer count based on available VRAM.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        if not getattr(self, "auto_gpu_layers", True):
            return

        # Try to get available VRAM using torch
        try:
            import torch

            if not torch.cuda.is_available():
                logger.debug("CUDA not available, n_gpu_layers=0")
                self.n_gpu_layers = 0
                return

            # Get free VRAM
            free_vram_gb = torch.cuda.mem_get_info()[0] / (1024**3)
            logger.debug(f"Free VRAM: {free_vram_gb:.2f} GB")

        except ImportError:
            logger.debug("PyTorch not available, skipping GPU layer auto-detection")
            # Fall back to attempting to use all layers
            self.n_gpu_layers = -1
            return

        # Auto-detect model specs from name
        from ingestforge.llm.llamacpp import (
            _detect_model_spec,
            calculate_optimal_gpu_layers,
        )

        model_path = getattr(self, "model_path", "") or ""
        spec = _detect_model_spec(str(model_path))

        # Calculate optimal layers based on VRAM
        optimal_layers = calculate_optimal_gpu_layers(
            free_vram_gb,
            model_name=str(model_path),
            total_layers=spec.get("layers"),
            gb_per_layer=spec.get("gb_per_layer"),
            overhead_gb=spec.get("overhead_gb"),
        )

        self.n_gpu_layers = optimal_layers
        logger.info(
            f"Auto-detected n_gpu_layers={optimal_layers} for ~{free_vram_gb:.1f}GB free VRAM"
        )

    def _get_model_params(self):
        """
        Extract model parameters from loaded model.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation (via assertion)
        Rule #9: Full type hints

        Returns:
            Model params object or None if not available
        """
        assert self._model is not None, "Model must be loaded"
        if not hasattr(self._model, "_model"):
            return None
        if not hasattr(self._model._model, "params"):
            return None

        return self._model._model.params

    def _extract_gpu_layer_recommendation(self, params: Any) -> Optional[int]:
        """
        Extract GPU layer recommendation from model params.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            params: Model parameters object

        Returns:
            Recommended GPU layers or None
        """
        if params is None:
            return None
        if not hasattr(params, "n_gpu_layers"):
            return None

        recommended = params.n_gpu_layers
        if recommended <= 0:
            return None

        return recommended

    def _auto_detect_gpu_layers_direct(self) -> None:
        """
        Directly query llama.cpp for GPU layer recommendation.

        Rule #1: Zero nesting - all logic extracted to helpers
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Raises:
            AssertionError: If called when model is not loaded
        """
        if self._model is None:
            logger.debug("Model not loaded, skipping direct GPU layer detection")
            return

        try:
            params = self._get_model_params()
            if params is None:
                logger.debug("Model params not available")
                return
            recommended = self._extract_gpu_layer_recommendation(params)
            if recommended is None:
                logger.debug("No GPU layer recommendation from model")
                return
            self._n_gpu_layers = recommended
            logger.info(f"Model recommends n_gpu_layers={recommended}")

        except Exception as e:
            logger.debug(f"Could not query model for GPU layers: {e}")

    def unload_model(self) -> None:
        """
        Unload the current model from memory.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        if self._model is not None:
            try:
                # Close the model
                if hasattr(self._model, "close"):
                    self._model.close()
                del self._model
                self._model = None
                logger.info(f"Unloaded model: {self.model_path}")
            except Exception as e:
                logger.warning(f"Error unloading model: {e}")

    def _safe_max_tokens(self, max_tokens: Optional[int]) -> int:
        """
        Apply safety ceiling to max_tokens to prevent OOM.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            max_tokens: Requested max tokens

        Returns:
            Safe max tokens value
        """
        if max_tokens is None:
            # Default to 2048 tokens for generation
            max_tokens = 2048

        # Get context length from instance attribute
        ctx_len = getattr(self, "n_ctx", 8192)

        # Apply safety margin (75% of context length)
        safe_ceiling = int(ctx_len * 0.75)

        # Return minimum of requested and safe ceiling
        return min(max_tokens, safe_ceiling)

    def _handle_oom_and_retry(
        self, error: Exception, messages: list, config: Any
    ) -> None:
        """
        Handle OOM error and retry on CPU.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines (extracted from stream_generate_with_context)
        Rule #9: Full type hints

        Args:
            error: The OOM exception
            messages: Chat messages
            config: Generation config

        Yields:
            Generated text chunks
        """
        error_str = str(error).lower()
        if not (
            "cuda" in error_str or "memory" in error_str or "out of memory" in error_str
        ):
            raise

        logger.warning(
            f"GPU OOM detected during streaming, falling back to CPU: {error}"
        )
        if self._n_gpu_layers == 0:
            raise RuntimeError(f"Out of memory even on CPU: {error}") from error

        # Retry on CPU
        self.unload_model()
        self._n_gpu_layers = 0
        self.n_gpu_layers = 0

        # Stream again on CPU
        yield from self._stream_chat_completion(messages, config)

    def _handle_oom_retry(
        self, error: Exception, messages: list, config: Any
    ) -> dict[str, Any]:
        """
        Handle OOM error and retry generation on CPU.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            error: The OOM exception
            messages: Chat messages
            config: Generation config

        Returns:
            Generation result dictionary

        Raises:
            RuntimeError: If OOM persists on CPU
        """
        error_str = str(error).lower()
        if (
            "cuda" not in error_str
            and "memory" not in error_str
            and "out of memory" not in error_str
        ):
            raise

        logger.warning(f"GPU OOM detected, falling back to CPU: {error}")
        if self._n_gpu_layers == 0:
            raise RuntimeError(f"Out of memory even on CPU: {error}") from error

        # Retry on CPU
        self.unload_model()
        self._n_gpu_layers = 0
        self.n_gpu_layers = 0

        # Regenerate on CPU
        return self.generate_chat(messages, config)

    @property
    def context_length(self) -> int:
        """
        Get the model's context length.

        Rule #1: Reduced nesting with helper method
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Returns:
            Context length in tokens
        """
        # Try to get from loaded model metadata
        if self._model is not None:
            try:
                metadata = self._model.metadata
                ctx = self._extract_context_from_metadata(metadata)
                if ctx:
                    return ctx
            except Exception as e:
                logger.debug(f"Could not extract context from model metadata: {e}")

        # Try to read from GGUF file
        gguf_ctx = self.get_model_context_from_gguf()
        if gguf_ctx:
            return gguf_ctx

        # Fall back to instance n_ctx attribute
        return getattr(self, "n_ctx", 8192)

    def _extract_context_from_metadata(self, metadata: dict) -> Optional[int]:
        """
        Extract context length from model metadata.

        Rule #1: Simple extraction logic
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            metadata: Model metadata dictionary

        Returns:
            Context length or None
        """
        # Try common metadata keys
        for key in [
            "llama.context_length",
            "context_length",
            "max_position_embeddings",
        ]:
            if key in metadata:
                return int(metadata[key])
        return None

    def get_model_context_from_gguf(self) -> Optional[int]:
        """
        Read context length directly from GGUF file.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Returns:
            Context length or None if not found
        """
        if not self.model_path:
            return None

        try:
            # Try to read GGUF metadata (requires gguf package)
            from gguf import GGUFReader

            reader = GGUFReader(str(self.model_path))

            # Look for context length in metadata
            for field in reader.fields.values():
                if "context" in field.name.lower():
                    return int(field.parts[0])

        except ImportError:
            logger.debug("gguf package not available, cannot read GGUF metadata")
        except Exception as e:
            logger.debug(f"Could not read context from GGUF: {e}")

        return None
