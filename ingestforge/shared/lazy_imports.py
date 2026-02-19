"""
Lazy Import Utilities for Optional Dependencies.

This module provides standardized patterns for lazy-loading optional dependencies.
IngestForge has many optional features (LLM providers, OCR, etc.) that shouldn't
require users to install everything just to use basic functionality.

Architecture Context
--------------------
Lazy imports are used throughout the codebase:

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   LLM Clients   │     │    Enrichers    │     │   Processors    │
    │  ClaudeClient   │     │ EntityExtractor │     │  OCRProcessor   │
    │  OllamaClient   │     │ EmbeddingGen    │     │  HTMLProcessor  │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                          @lazy_property
                          LazyImport()
                          check_optional_dependency()

Problem Solved
--------------
Without lazy imports:

    # This fails if anthropic isn't installed, even if user wants Ollama
    from anthropic import Anthropic  # ImportError!

With lazy imports:

    class ClaudeClient:
        @lazy_property
        def client(self):
            from anthropic import Anthropic  # Only fails if actually used
            return Anthropic()

Components
----------
**@lazy_property decorator**
    Converts a method into a cached lazy-loading property:

        class MyService:
            @lazy_property
    def model(self) -> Any:
                import expensive_library
                return expensive_library.Model()

        service = MyService()  # No import yet
        model = service.model  # Now it imports
        same = service.model   # Returns cached instance

**LazyImport context manager**
    Try importing a module with graceful failure:

        with LazyImport("google.generativeai", "pip install google-generativeai") as genai:
            if genai:
                model = genai.GenerativeModel("gemini-pro")
            else:
                print("Gemini not available")

**check_optional_dependency()**
    Check if a dependency is available:

        if check_optional_dependency("pytesseract", "pip install pytesseract"):
            import pytesseract
            # Use OCR
        else:
            # Fall back to non-OCR processing

Design Decisions
----------------
1. **Import at use-time**: Only import when the feature is actually used.
2. **Caching**: Lazy properties cache the result to avoid repeated imports.
3. **Clear error messages**: LazyImport includes install instructions in errors.
4. **Optional by default**: Code paths should handle missing dependencies gracefully.
"""

from functools import wraps
from typing import Callable, Any, Literal, Optional, Type, cast
from types import ModuleType, TracebackType


def lazy_property(import_func: Callable[..., Any]) -> Any:
    """Decorator for lazy-loaded properties.

    This decorator caches the result of the import function and returns it
    on subsequent accesses, avoiding repeated imports and expensive initialization.

    The decorated function should import and initialize the dependency.

    Args:
        import_func: Function that imports and returns the dependency

    Returns:
        A property that lazy-loads the dependency

    Examples:
        >>> class MyService:
        ...     @lazy_property
        ...     def client(self):
        ...         from expensive_library import ExpensiveClient
        ...         return ExpensiveClient()
        ...
        >>> service = MyService()
        >>> # Client not loaded yet
        >>> client = service.client  # Now it loads
        >>> same_client = service.client  # Returns cached instance
        >>> assert client is same_client
    """
    attr_name = f"_{import_func.__name__}_cached"

    @wraps(import_func)
    def wrapper(self: Any) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, import_func(self))
        return getattr(self, attr_name)

    return cast(Any, property(wrapper))


class LazyImport:
    """Context manager for lazy imports with error handling.

    This class provides a way to attempt importing optional dependencies
    with clear error messages if they're not available.

    Examples:
        >>> with LazyImport("google.generativeai", "pip install google-generativeai") as genai:
        ...     if genai:
        ...         model = genai.GenerativeModel("gemini-pro")
    """

    def __init__(self, module_name: str, install_command: str) -> None:
        """Initialize lazy import context.

        Args:
            module_name: Name of the module to import
            install_command: Command to install the module if missing
        """
        self.module_name = module_name
        self.install_command = install_command
        self.module: Optional[ModuleType] = None

    def __enter__(self) -> Any:
        """Attempt to import the module."""
        try:
            # Dynamic import
            parts = self.module_name.split(".")
            module = __import__(self.module_name)
            for part in parts[1:]:
                module = getattr(module, part)
            self.module = module
            return module
        except ImportError:
            return None

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        """Handle import errors."""
        if exc_type is AttributeError and self.module is None:
            raise ImportError(
                f"Module '{self.module_name}' is not installed. "
                f"Install it with: {self.install_command}"
            ) from exc_val
        return False


def check_optional_dependency(
    module_name: str, install_command: str, error_message: str | None = None
) -> bool:
    """Check if an optional dependency is available.

    Args:
        module_name: Name of the module to check
        install_command: Command to install the module
        error_message: Custom error message (optional)

    Returns:
        True if the module is available, False otherwise

    Raises:
        ImportError: If the module is not available and this is called
                     in a context where it's required

    Examples:
        >>> if check_optional_dependency("pytesseract", "pip install pytesseract"):
        ...     import pytesseract
        ...     # Use pytesseract
        ... else:
        ...     print("OCR not available")
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        if error_message:
            raise ImportError(error_message) from None
        return False
