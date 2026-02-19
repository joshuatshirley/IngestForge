"""
llama.cpp local LLM provider.

Runs GGUF models locally without a separate server using llama-cpp-python.
Supports Gemma, Llama, Mistral, Phi, Qwen, and other GGUF-format models.

Features:
- Auto VRAM detection for optimal GPU layer offloading
- Chat completion API for proper template handling
- No external server required (unlike Ollama)
"""

import time
import threading
from pathlib import Path
from typing import Any, Generator, Optional, Type
from types import TracebackType

from ingestforge.core.logging import get_logger
from ingestforge.core.retry import llm_retry
from ingestforge.llm.base import (
    ConfigurationError,
    GenerationConfig,
    GenerationResult,
    LLMClient,
    LLMError,
)
from ingestforge.llm.llamacpp_memory import LlamaCppMemoryMixin

logger = get_logger(__name__)

# Model specifications for VRAM auto-detection
# Each entry contains layer count and approximate VRAM per layer for Q4_K_M quantization
# MoE models (A3B suffix) use sparse activation and are more memory-efficient
MODEL_SPECS = {
    # Standard dense models
    "qwen2.5-14b": {"layers": 40, "gb_per_layer": 0.19, "overhead_gb": 1.5},
    "qwen2.5-7b": {"layers": 32, "gb_per_layer": 0.12, "overhead_gb": 1.2},
    "llama3-8b": {"layers": 32, "gb_per_layer": 0.14, "overhead_gb": 1.2},
    "llama2-7b": {"layers": 32, "gb_per_layer": 0.12, "overhead_gb": 1.2},
    "mistral-7b": {"layers": 32, "gb_per_layer": 0.12, "overhead_gb": 1.2},
    "gemma-2b": {"layers": 18, "gb_per_layer": 0.08, "overhead_gb": 0.8},
    "gemma-7b": {"layers": 28, "gb_per_layer": 0.12, "overhead_gb": 1.0},
    "phi-2": {"layers": 32, "gb_per_layer": 0.08, "overhead_gb": 0.8},
    # Mixture of Experts (MoE) models - sparse activation, very efficient
    "qwen3-coder-30b-a3b": {"layers": 48, "gb_per_layer": 0.35, "overhead_gb": 2.0},
    "glm-4.7-flash-23b-a3b": {"layers": 40, "gb_per_layer": 0.30, "overhead_gb": 1.8},
}

# Default specs for unknown models
DEFAULT_MODEL_SPEC = {"layers": 32, "gb_per_layer": 0.15, "overhead_gb": 1.2}


def _detect_model_spec(model_name: str) -> dict[str, Any]:
    """
    Detect model specs from model name.

    Args:
        model_name: Model filename or name (e.g., "qwen2.5-14b-instruct-q4_k_m")

    Returns:
        Dict with layers, gb_per_layer, overhead_gb
    """
    name_lower = model_name.lower()
    name_normalized = name_lower.replace("-", "").replace(".", "")

    # Try to match known models
    for model_key, specs in MODEL_SPECS.items():
        if model_key.replace("-", "").replace(".", "") in name_normalized:
            return specs

    # Size-based fallback specs
    size_specs = {
        "14b": {"layers": 40, "gb_per_layer": 0.19, "overhead_gb": 1.5},
        "13b": {"layers": 40, "gb_per_layer": 0.17, "overhead_gb": 1.4},
        "7b": {"layers": 32, "gb_per_layer": 0.12, "overhead_gb": 1.2},
        "8b": {"layers": 32, "gb_per_layer": 0.12, "overhead_gb": 1.2},
        "3b": {"layers": 26, "gb_per_layer": 0.10, "overhead_gb": 1.0},
        "2b": {"layers": 18, "gb_per_layer": 0.08, "overhead_gb": 0.8},
    }

    for size_key, specs in size_specs.items():
        if size_key in name_lower:
            return specs

    return DEFAULT_MODEL_SPEC


def calculate_optimal_gpu_layers(
    free_vram_gb: float,
    model_name: str = "",
    total_layers: Optional[int] = None,
    gb_per_layer: Optional[float] = None,
    overhead_gb: Optional[float] = None,
) -> int:
    """
    Calculate optimal n_gpu_layers based on available VRAM.

    For qwen2.5-14b Q4_K_M:
    - 40 layers, ~0.19 GB per layer
    - Full offload needs ~9 GB VRAM

    Args:
        free_vram_gb: Available VRAM in gigabytes
        model_name: Model name for auto-detecting specs
        total_layers: Override total layer count
        gb_per_layer: Override GB per layer estimate
        overhead_gb: Override VRAM overhead estimate

    Returns:
        Recommended number of GPU layers (0 = CPU only)
    """
    # Get model specs
    specs = _detect_model_spec(model_name) if model_name else DEFAULT_MODEL_SPEC

    layers = total_layers if total_layers is not None else specs["layers"]
    per_layer = gb_per_layer if gb_per_layer is not None else specs["gb_per_layer"]
    overhead = overhead_gb if overhead_gb is not None else specs["overhead_gb"]

    # Calculate available VRAM after overhead
    available = free_vram_gb - overhead
    if available <= 0:
        return 0

    # Calculate max layers that fit
    max_layers = int(available / per_layer)
    return min(max_layers, layers)


class LlamaCppClient(
    LlamaCppMemoryMixin,
    LLMClient,
):
    """
    Local LLM client using llama-cpp-python.

    Runs GGUF format models directly in Python without requiring
    an external server like Ollama.

    Supported models (GGUF format):
    - Qwen 2.5 (7B, 14B)
    - Gemma 2B/7B
    - Llama 2/3 (7B, 8B, 13B)
    - Mistral 7B
    - Phi-2/3
    - Many others from HuggingFace

    Features:
    - Auto VRAM detection for optimal GPU layer offloading
    - Proper chat templates via create_chat_completion()
    - No external server required

    Example:
        # With auto GPU detection (recommended)
        client = LlamaCppClient(
            model_path="models/qwen2.5-14b-instruct-q4_k_m.gguf",
            n_ctx=8192,
            auto_gpu_layers=True,  # Detects VRAM automatically
        )

        # With manual GPU layers
        client = LlamaCppClient(
            model_path="models/qwen2.5-14b-instruct-q4_k_m.gguf",
            n_ctx=8192,
            n_gpu_layers=40,  # All layers on GPU
            auto_gpu_layers=False,
        )

        response = client.generate("What is machine learning?")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        auto_gpu_layers: bool = True,
    ):
        """
        Initialize llama.cpp client.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (tokens). Default 8192 for modern models.
            n_gpu_layers: Number of layers to offload to GPU.
                         0 = CPU only (or auto-detect if auto_gpu_layers=True)
                         -1 = All layers on GPU
                         N = Specific layer count
            n_threads: Number of CPU threads (None = auto)
            verbose: Enable verbose logging from llama.cpp
            auto_gpu_layers: If True and n_gpu_layers=0, auto-detect from VRAM
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self.auto_gpu_layers = auto_gpu_layers
        self._model = None
        self._model_name = Path(model_path).stem if model_path else "unknown"
        self._vram_info = None
        self._model_lock = threading.Lock()
        self._n_gpu_layers = 0  # Track original n_gpu_layers for OOM fallback

        # Handle GPU layer configuration
        if n_gpu_layers == -1:
            # -1 means all layers (will be resolved during model load)
            self.n_gpu_layers = -1
            self._n_gpu_layers = -1
        elif n_gpu_layers > 0:
            # Explicit layer count
            self.n_gpu_layers = n_gpu_layers
            self._n_gpu_layers = n_gpu_layers
        elif auto_gpu_layers:
            # Auto-detect from VRAM
            self._auto_detect_gpu_layers()
            self._n_gpu_layers = self.n_gpu_layers
        else:
            # CPU only
            self.n_gpu_layers = 0
            self._n_gpu_layers = 0

    # Sprint 3 (Rule #4): Memory management methods moved to llamacpp_memory.py

    def __enter__(self):
        """Support context manager for automatic cleanup."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Cleanup resources on context manager exit."""
        self.unload_model()
        return False

    @property
    def model(self) -> Any:
        """Lazy-load the model (thread-safe)."""
        with self._model_lock:
            if self._model is None:
                if not self.model_path:
                    raise ConfigurationError(
                        "model_path is required. Provide path to a GGUF model file."
                    )

                model_file = Path(self.model_path)
                if not model_file.exists():
                    raise ConfigurationError(
                        f"Model file not found: {self.model_path}\n"
                        "Download a GGUF model from HuggingFace, e.g.:\n"
                        "  huggingface-cli download google/gemma-2b-it-GGUF gemma-2b-it-q4_k_m.gguf"
                    )

                try:
                    from llama_cpp import Llama

                    logger.info(
                        f"Loading model: {self._model_name}",
                        n_ctx=self.n_ctx,
                        n_gpu_layers=self.n_gpu_layers,
                    )

                    self._model = Llama(
                        model_path=str(model_file),
                        n_ctx=self.n_ctx,
                        n_gpu_layers=self.n_gpu_layers,
                        n_threads=self.n_threads,
                        verbose=self.verbose,
                    )

                    logger.info(f"Model loaded successfully: {self._model_name}")

                except ImportError:
                    raise ImportError(
                        "llama-cpp-python is required for local LLM inference.\n"
                        "Install with: pip install llama-cpp-python\n"
                        "For GPU support: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python"
                    )

            return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        """Check if model is available."""
        if not self.model_path:
            return False
        return Path(self.model_path).exists()

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from prompt.

        Rule #4: Reduced from 62 → 33 lines via helper extraction

        Supports GBNF grammar for constrained output via config.grammar.
        """
        config = config or GenerationConfig()
        safe_max_tokens = self._safe_max_tokens(config.max_tokens)

        # Build generation kwargs
        gen_kwargs = {
            "max_tokens": safe_max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stop": config.stop_sequences,
            "echo": False,
        }

        # Add grammar if provided (for constrained generation)
        if config.grammar:
            gen_kwargs["grammar"] = self._get_llama_grammar(config.grammar)

        try:
            response = self.model(prompt, **gen_kwargs)
            return self._extract_llamacpp_response(response)

        except (RuntimeError, MemoryError) as e:
            return self._handle_oom_error(e, prompt, config, **kwargs)
        except Exception as e:
            if "llama" in str(type(e).__module__).lower():
                raise LLMError(f"llama.cpp error: {e}")
            raise

    def _get_llama_grammar(self, grammar_str: str) -> Any:
        """Convert GBNF grammar string to LlamaGrammar object.

        Args:
            grammar_str: GBNF grammar string

        Returns:
            LlamaGrammar object for constrained generation
        """
        try:
            from llama_cpp import LlamaGrammar

            return LlamaGrammar.from_string(grammar_str)
        except ImportError:
            logger.warning("LlamaGrammar not available, grammar constraint ignored")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse grammar: {e}")
            return None

    def _extract_llamacpp_response(self, response: Any) -> str:
        """
        Extract text from llama.cpp response and record usage.

        Rule #4: Extracted to reduce generate() size
        """
        if response and "choices" in response and response["choices"]:
            text = response["choices"][0].get("text", "")
            usage = response.get("usage", {})
            if usage:
                self._record_usage(
                    prompt_tokens=usage.get("prompt_tokens", 0) or 0,
                    completion_tokens=usage.get("completion_tokens", 0) or 0,
                )
            return text.strip()
        else:
            raise LLMError("Empty response from llama.cpp")

    def _handle_oom_error(
        self, error: Exception, prompt: str, config: GenerationConfig, **kwargs: Any
    ) -> str:
        """
        Handle OOM errors with CPU fallback.

        Rule #4: Extracted to reduce generate() size
        """
        error_str = str(error).lower()
        if "cuda" in error_str or "memory" in error_str or "out of memory" in error_str:
            logger.warning(f"GPU OOM detected, falling back to CPU: {error}")
            # Only attempt fallback once
            if self._n_gpu_layers > 0:
                self.unload_model()
                self._n_gpu_layers = 0
                self.n_gpu_layers = 0
                # Retry on CPU
                return self.generate(prompt, config, **kwargs)
            else:
                raise LLMError(f"Out of memory even on CPU: {error}") from error
        raise LLMError(f"llama.cpp generation failed: {error}") from error

    @llm_retry
    def generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """
        Generate with system prompt and context.

        Uses create_chat_completion() API which automatically handles
        chat templates for different model architectures (Qwen, Llama, Gemma, etc.)
        based on metadata embedded in the GGUF file.
        """
        # Build messages for chat completion API
        messages = []

        # System prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # User message with optional context
        if context:
            user_content = f"Context:\n{context}\n\n{user_prompt}"
        else:
            user_content = user_prompt

        messages.append({"role": "user", "content": user_content})

        return self.generate_chat(messages, config, **kwargs)

    @property
    def supports_streaming(self) -> bool:
        return True

    def stream_generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Stream text generation with llama.cpp.

        Rule #4: Reduced from 63 lines to <60 lines via helper extraction

        Uses create_chat_completion() with streaming for proper template handling.
        """
        config = config or GenerationConfig()

        # Apply safety ceiling to max_tokens
        safe_max_tokens = self._safe_max_tokens(config.max_tokens)

        # Build messages for chat completion API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context:
            user_content = f"Context:\n{context}\n\n{user_prompt}"
        else:
            user_content = user_prompt
        messages.append({"role": "user", "content": user_content})

        try:
            yield from self._stream_chat_completion(messages, safe_max_tokens, config)
        except (RuntimeError, MemoryError) as e:
            yield from self._handle_oom_and_retry(
                e, system_prompt, user_prompt, context, config, **kwargs
            )
        except Exception as e:
            raise LLMError(f"llama.cpp streaming failed: {e}")

    def _stream_chat_completion(
        self, messages: list, max_tokens: int, config: GenerationConfig
    ) -> Generator[str, None, None]:
        """
        Stream chat completion tokens.

        Rule #4: Extracted to reduce function size
        """
        stream = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop_sequences,
            stream=True,
        )
        for output in stream:
            if output and "choices" in output and output["choices"]:
                delta = output["choices"][0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield text

    def generate_chat(
        self,
        messages: list,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """
        Generate response for chat messages.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            config: Generation configuration

        Returns:
            Generated response
        """
        config = config or GenerationConfig()

        # Apply safety ceiling to max_tokens
        safe_max_tokens = self._safe_max_tokens(config.max_tokens)

        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=safe_max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop_sequences,
            )

            if response and "choices" in response and response["choices"]:
                message = response["choices"][0].get("message", {})
                return message.get("content", "").strip()
            else:
                raise LLMError("Empty chat response from llama.cpp")

        except (RuntimeError, MemoryError) as e:
            # Handle OOM errors
            error_str = str(e).lower()
            if (
                "cuda" in error_str
                or "memory" in error_str
                or "out of memory" in error_str
            ):
                logger.warning(f"GPU OOM detected in chat generation: {e}")
                if self._n_gpu_layers > 0:
                    self.unload_model()
                    self._n_gpu_layers = 0
                    self.n_gpu_layers = 0
                    # Retry on CPU
                    return self.generate_chat(messages, config, **kwargs)
                else:
                    raise LLMError(f"Out of memory even on CPU: {e}") from e
            raise LLMError(f"llama.cpp chat generation failed: {e}") from e
        except Exception as e:
            if "llama" in str(type(e).__module__).lower():
                raise LLMError(f"llama.cpp chat error: {e}")
            raise

    def _build_chat_messages(
        self, system_prompt: str, user_prompt: str, context: Optional[str]
    ) -> list[Any]:
        """
        Build messages array for llama.cpp chat API.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            system_prompt: System instruction
            user_prompt: User query
            context: Optional context to prepend

        Returns:
            List of message dictionaries
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context:
            user_content = f"Context:\n{context}\n\n{user_prompt}"
        else:
            user_content = user_prompt
        messages.append({"role": "user", "content": user_content})

        return messages

    def _parse_llamacpp_response(
        self, response: dict, latency_ms: float
    ) -> GenerationResult:
        """
        Parse llama.cpp response into GenerationResult.

        Rule #1: Early return pattern
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            response: Response dictionary from llama.cpp
            latency_ms: Request latency in milliseconds

        Returns:
            GenerationResult with parsed data
        """
        text = ""
        finish_reason = "stop"
        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            message = choice.get("message", {})
            text = message.get("content", "").strip()
            finish_reason = choice.get("finish_reason", "stop")

        # Calculate usage statistics
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if "usage" in response:
            usage["prompt_tokens"] = response["usage"].get("prompt_tokens", 0) or 0
            usage["completion_tokens"] = (
                response["usage"].get("completion_tokens", 0) or 0
            )
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            self._record_usage(usage["prompt_tokens"], usage["completion_tokens"])

        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            model=self._model_name,
            latency_ms=latency_ms,
            usage=usage,
            raw_response=response,
        )

    def generate_with_result(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate text and return rich result with metadata.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            system_prompt: System instruction
            user_prompt: User query
            context: Optional context to prepend
            config: Generation configuration
            **kwargs: Additional arguments

        Returns:
            GenerationResult with text, finish_reason, latency, and usage

        Raises:
            LLMError: On generation failure or OOM
        """
        config = config or GenerationConfig()
        start_time = time.time()

        # Build request components using helpers
        safe_max_tokens = self._safe_max_tokens(config.max_tokens)
        messages = self._build_chat_messages(system_prompt, user_prompt, context)

        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=safe_max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop_sequences,
            )

            latency_ms = (time.time() - start_time) * 1000
            return self._parse_llamacpp_response(response, latency_ms)

        except (RuntimeError, MemoryError) as e:
            return self._handle_oom_retry(
                e, system_prompt, user_prompt, context, config, **kwargs
            )
        except Exception as e:
            raise LLMError(f"llama.cpp generation failed: {e}")

    def tokenize(self, text: str) -> list[Any]:
        """Tokenize text and return token IDs."""
        return self.model.tokenize(text.encode())

    def detokenize(self, tokens: list) -> str:
        """Convert token IDs back to text."""
        return self.model.detokenize(tokens).decode()


def download_model(
    repo_id: str,
    filename: str,
    output_dir: str = "models",
) -> Path:
    """
    Download a GGUF model from HuggingFace.

    Args:
        repo_id: HuggingFace repo (e.g., "google/gemma-2b-it-GGUF")
        filename: Model filename (e.g., "gemma-2b-it-q4_k_m.gguf")
        output_dir: Directory to save model

    Returns:
        Path to downloaded model
    """
    try:
        from huggingface_hub import hf_hub_download

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {filename} from {repo_id}...")

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(output_path),
        )

        logger.info(f"Model downloaded to: {model_path}")
        return Path(model_path)

    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models.\n"
            "Install with: pip install huggingface_hub"
        )


# Pre-configured model suggestions
# Each entry includes HuggingFace download info and VRAM specs for auto-detection
RECOMMENDED_MODELS = {
    # Qwen 2.5 - Excellent for research and instruction following (via bartowski)
    "qwen2.5-14b": {
        "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "filename": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "description": "Qwen 2.5 14B Instruct, 4-bit quantized (~8.9GB). Best quality.",
        "layers": 40,
        "gb_per_layer": 0.19,
    },
    "qwen2.5-7b": {
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "description": "Qwen 2.5 7B Instruct, 4-bit quantized (~4.7GB). Good balance.",
        "layers": 32,
        "gb_per_layer": 0.12,
    },
    "qwen2.5-3b": {
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "description": "Qwen 2.5 3B Instruct, 4-bit quantized (~2.0GB). Fast & lightweight.",
        "layers": 26,
        "gb_per_layer": 0.06,
    },
    # Llama 3
    "llama3-8b": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "description": "Meta Llama 3 8B Instruct, 4-bit quantized (~4.9GB)",
        "layers": 32,
        "gb_per_layer": 0.14,
    },
    # Mistral
    "mistral-7b": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "description": "Mistral 7B Instruct, 4-bit quantized (~4.4GB)",
        "layers": 32,
        "gb_per_layer": 0.12,
    },
    # Gemma
    "gemma-2b": {
        "repo_id": "google/gemma-2b-it-GGUF",
        "filename": "gemma-2b-it-q4_k_m.gguf",
        "description": "Google Gemma 2B instruction-tuned, 4-bit quantized (~1.5GB)",
        "layers": 18,
        "gb_per_layer": 0.08,
    },
    # Phi
    "phi-2": {
        "repo_id": "TheBloke/phi-2-GGUF",
        "filename": "phi-2.Q4_K_M.gguf",
        "description": "Microsoft Phi-2 2.7B, 4-bit quantized (~1.6GB)",
        "layers": 32,
        "gb_per_layer": 0.05,
    },
    # Mixture of Experts (MoE) - Larger models with sparse activation
    "qwen3-coder-30b-a3b": {
        "repo_id": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "filename": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        "description": "Qwen3 Coder 30B MoE (3B active), 4-bit quantized (~17.3GB). Best for code.",
        "layers": 48,
        "gb_per_layer": 0.35,
    },
    "glm-4.7-flash-23b-a3b": {
        "repo_id": "unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF",
        "filename": "GLM-4.7-Flash-REAP-23B-A3B-Q4_K_M.gguf",
        "description": "GLM 4.7 Flash 23B MoE (3B active), 4-bit quantized (~13.1GB). Fast general use.",
        "layers": 40,
        "gb_per_layer": 0.30,
    },
}
