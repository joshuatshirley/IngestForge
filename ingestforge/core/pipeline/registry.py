"""
Processor Registry for IngestForge (IF).

Provides dynamic discovery and registration of IFProcessors.
Process-safe registry with global locks for resource-heavy operations.
Follows NASA JPL Power of Ten rules.
"""

import logging
import multiprocessing
import os
from typing import Callable, Dict, List, Type, Any, Optional

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.errors import SafeErrorMessage
from ingestforge.core.pipeline.artifacts import IFFileArtifact

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_PROCESSORS = 256
MAX_ENRICHER_FACTORIES = 128
MAX_LOCK_WAIT_SEC = 30  # Maximum wait time for resource lock

# =============================================================================
# GLOBAL PROCESS LOCK FOR RESOURCE-HEAVY INITIALIZATION
# =============================================================================

# Global lock for serializing resource-heavy operations (e.g., loading models)
# This prevents RAM spikes when multiple workers try to load the same model
_GLOBAL_RESOURCE_LOCK: Optional[multiprocessing.Lock] = None


def _get_resource_lock() -> multiprocessing.Lock:
    """
    Get or create the global resource lock.

    Shared lock for model initialization.
    JPL Rule #7: Check return values.

    Returns:
        multiprocessing.Lock for resource synchronization.
    """
    global _GLOBAL_RESOURCE_LOCK
    if _GLOBAL_RESOURCE_LOCK is None:
        _GLOBAL_RESOURCE_LOCK = multiprocessing.Lock()
    return _GLOBAL_RESOURCE_LOCK


def acquire_resource_lock(timeout: float = MAX_LOCK_WAIT_SEC) -> bool:
    """
    Acquire the global resource lock with timeout.

    Lock for model initialization to prevent RAM spikes.
    JPL Rule #2: Bounded wait time.

    Args:
        timeout: Maximum seconds to wait (default: MAX_LOCK_WAIT_SEC).

    Returns:
        True if lock acquired, False if timeout.
    """
    lock = _get_resource_lock()
    return lock.acquire(timeout=timeout)


def release_resource_lock() -> None:
    """
    Release the global resource lock.

    Release after resource initialization complete.
    """
    lock = _get_resource_lock()
    try:
        lock.release()
    except RuntimeError:
        # Lock was not held - ignore
        pass


class EnricherEntry:
    """
    Entry for a registered enricher factory.

    Enricher Registration Decorator.
    Stores factory function, capabilities, and priority for lazy instantiation.
    """

    __slots__ = ("factory", "capabilities", "priority", "cls_name")

    def __init__(
        self,
        factory: Callable[..., IFProcessor],
        capabilities: List[str],
        priority: int,
        cls_name: str,
    ):
        self.factory = factory
        self.capabilities = capabilities
        self.priority = priority
        self.cls_name = cls_name


class IFRegistry:
    """
    Process-aware singleton registry for IFProcessors.

    Registry is process-aware to prevent duplicate instances per worker.
    Supports routing by:
    - MIME type (traditional)
    - Functional capabilities ()
    - Enricher factory ()

    JPL Rule #5: Assert registry health before processing.
    """

    _instance: Optional["IFRegistry"] = None
    _instance_pid: Optional[int] = None  # Track owning process
    _processors: Dict[str, List[IFProcessor]] = {}
    _id_map: Dict[str, IFProcessor] = {}
    _capability_index: Dict[str, List[IFProcessor]] = {}
    # Enricher factory registration
    _enricher_factories: Dict[str, EnricherEntry] = {}
    _enricher_capability_index: Dict[str, List[str]] = {}
    # Health tracking
    _initialized: bool = False
    # Auto-discovery tracking
    _auto_discovered: bool = False

    def __new__(cls) -> "IFRegistry":
        """
        Process-aware singleton creation.

        Detects process boundary crossing and creates fresh instance.
        Triggers auto-discovery on first creation.
        JPL Rule #7: Check process ID to prevent stale singleton reuse.
        """
        current_pid = os.getpid()

        # Detect if we're in a new process (forked worker)
        if cls._instance is not None and cls._instance_pid != current_pid:
            logger.debug(
                f"Registry: Detected process change ({cls._instance_pid} -> {current_pid}), "
                "creating fresh instance for worker"
            )
            # Reset for new process
            cls._instance = None
            cls._reset_class_state()

        if cls._instance is None:
            cls._instance = super(IFRegistry, cls).__new__(cls)
            cls._instance_pid = current_pid
            cls._initialized = True
            logger.debug(f"Registry: Created new instance for process {current_pid}")

            # Auto-discover processors on first creation
            if not cls._auto_discovered:
                cls._auto_discovered = True
                _auto_discover_processors()

        return cls._instance

    @classmethod
    def _reset_class_state(cls) -> None:
        """
        Reset all class-level state for a fresh process.

        Called when process boundary is detected.
        Also resets auto-discovery flag.
        """
        cls._processors = {}
        cls._id_map = {}
        cls._capability_index = {}
        cls._enricher_factories = {}
        cls._enricher_capability_index = {}
        cls._initialized = False
        cls._auto_discovered = False

    def is_healthy(self) -> bool:
        """
        Check if the registry is in a healthy state.

        AC / JPL Rule #5: Health check before processing.

        Returns:
            True if registry is healthy and ready for use.
        """
        # Check basic initialization
        if not self._initialized:
            return False

        # Check we're in the correct process
        if self._instance_pid != os.getpid():
            return False

        return True

    def assert_healthy(self) -> None:
        """
        Assert that the registry is healthy for worker startup.

        AC / JPL Rule #5: Assert registry health before worker starts.

        Raises:
            AssertionError: If registry is not in a healthy state.
        """
        assert self._initialized, "Registry not initialized"
        assert self._instance_pid == os.getpid(), (
            f"Registry belongs to process {self._instance_pid}, "
            f"current process is {os.getpid()}"
        )

    def get_process_id(self) -> Optional[int]:
        """
        Get the process ID that owns this registry instance.

        Process awareness for debugging.

        Returns:
            Process ID or None if not set.
        """
        return self._instance_pid

    def register(
        self, processor: IFProcessor, mime_types: List[str], priority: int = 100
    ) -> None:
        """
        Register a processor for specific MIME types and capabilities.

        Rule #2: Fixed upper bound check.
        Rule #4: Function < 60 lines.
        """
        if len(self._id_map) >= MAX_PROCESSORS:
            raise RuntimeError(f"Registry limit reached: {MAX_PROCESSORS} processors")

        # Attach priority to processor instance for sorting later
        object.__setattr__(processor, "_priority", priority)

        self._id_map[processor.processor_id] = processor

        # Index by MIME type
        for mime in mime_types:
            if mime not in self._processors:
                self._processors[mime] = []
            self._processors[mime].append(processor)
            self._processors[mime].sort(
                key=lambda p: (-getattr(p, "_priority", 100), p.processor_id)
            )

        # Index by capability ()
        for cap in processor.capabilities:
            if cap not in self._capability_index:
                self._capability_index[cap] = []
            self._capability_index[cap].append(processor)
            self._capability_index[cap].sort(
                key=lambda p: (-getattr(p, "_priority", 100), p.processor_id)
            )

        caps = processor.capabilities or ["none"]
        logger.debug(
            f"Registered IFProcessor: {processor.processor_id} for {mime_types}, capabilities={caps}"
        )

    def get_processors(self, mime_type: str) -> List[IFProcessor]:
        """Return list of processors for a MIME type, sorted by priority."""
        return self._processors.get(mime_type, [])

    def dispatch(self, artifact: IFArtifact) -> IFProcessor:
        """
        Dispatch the best available processor for an artifact.

        Rule #7: Check return values (Implicitly by raising if None).
        """
        mime_type = "application/octet-stream"
        if isinstance(artifact, IFFileArtifact):
            mime_type = artifact.mime_type

        processors = self.get_processors(mime_type)

        for proc in processors:
            if proc.is_available():
                return proc

        # SEC-002: Sanitize error message to prevent info disclosure
        error_msg = SafeErrorMessage.sanitize(
            RuntimeError(
                f"No available IFProcessor for artifact: {artifact.artifact_id} (MIME: {mime_type})"
            ),
            "processor discovery",
            logger,
        )
        raise RuntimeError(error_msg)

    def get_by_capability(self, capability: str) -> List[IFProcessor]:
        """
        Get processors by functional capability.

        Capability - Functional Routing.
        Rule #9: Complete type hints.

        Args:
            capability: Capability string (e.g., "ocr", "embedding").

        Returns:
            List of matching processors sorted by priority.
        """
        return self._capability_index.get(capability, [])

    def get_by_capabilities(
        self, capabilities: List[str], match: str = "all"
    ) -> List[IFProcessor]:
        """
        Get processors matching multiple capabilities.

        Capability - Functional Routing.
        Rule #9: Complete type hints.

        Args:
            capabilities: List of required capability strings.
            match: "all" requires ALL capabilities, "any" requires ANY.

        Returns:
            List of matching processors sorted by priority.
        """
        if not capabilities:
            return []

        if match == "any":
            # Union of all capabilities
            seen_ids: set[str] = set()
            result: List[IFProcessor] = []
            for cap in capabilities:
                for proc in self._capability_index.get(cap, []):
                    if proc.processor_id not in seen_ids:
                        seen_ids.add(proc.processor_id)
                        result.append(proc)
            # Re-sort by priority
            result.sort(key=lambda p: (-getattr(p, "_priority", 100), p.processor_id))
            return result

        # match == "all": intersection
        cap_set = set(capabilities)
        result = []
        # Check each registered processor for all capabilities
        for proc in self._id_map.values():
            proc_caps = set(proc.capabilities)
            if cap_set.issubset(proc_caps):
                result.append(proc)
        result.sort(key=lambda p: (-getattr(p, "_priority", 100), p.processor_id))
        return result

    def dispatch_by_capability(
        self, capability: str, artifact: IFArtifact
    ) -> IFProcessor:
        """
        Dispatch the best available processor with a specific capability.

        Capability - Functional Routing.
        Rule #7: Check return values.

        Args:
            capability: Required capability string.
            artifact: The artifact to process.

        Returns:
            First available processor with the capability.

        Raises:
            RuntimeError: If no available processor found.
        """
        processors = self.get_by_capability(capability)

        for proc in processors:
            if proc.is_available():
                return proc

        raise RuntimeError(
            f"No available IFProcessor with capability '{capability}' "
            f"for artifact: {artifact.artifact_id}"
        )

    def get_processors_by_memory(self, max_mb: int) -> List[IFProcessor]:
        """
        Get processors whose memory requirements fit within a limit.

        Resources - Memory-Aware Selection.
        Rule #7: Check return values.
        Rule #9: Complete type hints.

        Args:
            max_mb: Maximum memory in megabytes.

        Returns:
            List of processors requiring <= max_mb, sorted by priority.
        """
        result = [proc for proc in self._id_map.values() if proc.memory_mb <= max_mb]
        result.sort(key=lambda p: (-getattr(p, "_priority", 100), p.processor_id))
        return result

    def dispatch_memory_safe(
        self, artifact: IFArtifact, max_mb: Optional[int] = None
    ) -> IFProcessor:
        """
        Dispatch processor considering memory constraints.

        Resources - Memory-Aware Selection.
        Rule #7: Check return values.

        Args:
            artifact: The artifact to process.
            max_mb: Maximum memory limit in MB. If None, uses available system memory.

        Returns:
            First available processor that fits within memory constraints.

        Raises:
            RuntimeError: If no suitable processor found.
        """
        # Determine memory limit
        if max_mb is None:
            max_mb = get_available_memory_mb()

        mime_type = "application/octet-stream"
        if isinstance(artifact, IFFileArtifact):
            mime_type = artifact.mime_type

        processors = self.get_processors(mime_type)

        for proc in processors:
            # Check both availability and memory requirements
            if proc.is_available() and proc.memory_mb <= max_mb:
                return proc

        raise RuntimeError(
            f"No available IFProcessor for artifact: {artifact.artifact_id} "
            f"(MIME: {mime_type}) within memory limit: {max_mb}MB"
        )

    def teardown_all(self) -> Dict[str, Any]:
        """
        Teardown all registered processors safely.

        Teardown - Safe Resource Finalization.
        Rule #1: Linear control flow.
        Rule #7: Check return values, isolate errors.

        Returns:
            Summary dict with 'success_count', 'failure_count', 'failed_ids'.
        """
        success_count = 0
        failure_count = 0
        failed_ids: List[str] = []

        for proc_id, proc in self._id_map.items():
            try:
                result = proc.teardown()
                if result:
                    success_count += 1
                    logger.debug(f"Teardown succeeded: {proc_id}")
                else:
                    failure_count += 1
                    failed_ids.append(proc_id)
                    logger.warning(f"Teardown returned False: {proc_id}")
            except Exception as e:
                failure_count += 1
                failed_ids.append(proc_id)
                logger.error(f"Teardown failed for {proc_id}: {e}")

        logger.info(
            f"Registry teardown complete: {success_count} succeeded, "
            f"{failure_count} failed"
        )

        return {
            "success_count": success_count,
            "failure_count": failure_count,
            "failed_ids": failed_ids,
        }

    def __enter__(self) -> "IFRegistry":
        """
        Context manager entry.

        Teardown - Safe Resource Finalization.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """
        Context manager exit - ensures teardown on scope exit.

        Teardown - Safe Resource Finalization.
        Rule #7: Check return values.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Traceback if exception raised.

        Returns:
            False to propagate any exceptions.
        """
        self.teardown_all()
        return False  # Don't suppress exceptions

    # -------------------------------------------------------------------------
    # Enricher Factory Registration
    # -------------------------------------------------------------------------

    def register_enricher(
        self,
        cls: Type[IFProcessor],
        capabilities: List[str],
        priority: int = 100,
        factory: Optional[Callable[..., IFProcessor]] = None,
    ) -> None:
        """
        Register an enricher processor by capability with factory support.

        Enricher Registration Decorator.
        Rule #2: Fixed upper bound check.
        Rule #9: Complete type hints.

        Args:
            cls: The IFProcessor class being registered.
            capabilities: List of capabilities this enricher provides.
            priority: Higher priority = selected first (default 100).
            factory: Optional factory function. If None, cls() is used.

        Raises:
            RuntimeError: If registry limit reached.
            ValueError: If capabilities list is empty.
        """
        if len(self._enricher_factories) >= MAX_ENRICHER_FACTORIES:
            raise RuntimeError(
                f"Enricher factory limit reached: {MAX_ENRICHER_FACTORIES}"
            )

        if not capabilities:
            raise ValueError("Enricher must have at least one capability")

        cls_name = cls.__name__

        # Use cls as factory if none provided
        effective_factory = factory if factory is not None else cls

        entry = EnricherEntry(
            factory=effective_factory,
            capabilities=list(capabilities),
            priority=priority,
            cls_name=cls_name,
        )

        self._enricher_factories[cls_name] = entry

        # Index by capability
        for cap in capabilities:
            if cap not in self._enricher_capability_index:
                self._enricher_capability_index[cap] = []
            if cls_name not in self._enricher_capability_index[cap]:
                self._enricher_capability_index[cap].append(cls_name)
                # Sort by priority (descending)
                self._enricher_capability_index[cap].sort(
                    key=lambda name: (
                        -self._enricher_factories[name].priority,
                        name,
                    )
                )

        logger.debug(
            f"Registered enricher: {cls_name} for capabilities={capabilities}, "
            f"priority={priority}"
        )

    def get_enricher_factory(
        self,
        capability: str,
    ) -> Optional[Callable[..., IFProcessor]]:
        """
        Get the highest-priority enricher factory for a capability.

        Enricher Registration Decorator.
        Rule #7: Check return values.
        Rule #9: Complete type hints.

        Args:
            capability: Required capability string.

        Returns:
            Factory function if found, None otherwise.
        """
        cls_names = self._enricher_capability_index.get(capability, [])

        if not cls_names:
            return None

        # First entry is highest priority
        entry = self._enricher_factories.get(cls_names[0])
        return entry.factory if entry else None

    def get_enricher(
        self,
        capability: str,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[IFProcessor]:
        """
        Get an enricher instance by capability.

        Enricher Registration Decorator.
        Instantiates via factory, passing through any arguments.
        Rule #7: Check return values.
        Rule #9: Complete type hints.

        Args:
            capability: Required capability string.
            *args: Positional arguments for factory.
            **kwargs: Keyword arguments for factory.

        Returns:
            Instantiated IFProcessor if factory found, None otherwise.
        """
        factory = self.get_enricher_factory(capability)
        if factory is None:
            return None

        try:
            return factory(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"Failed to instantiate enricher for capability '{capability}': {e}"
            )
            return None

    def get_enricher_factories_by_capability(
        self,
        capability: str,
    ) -> List[Callable[..., IFProcessor]]:
        """
        Get all enricher factories for a capability, sorted by priority.

        Enricher Registration Decorator.
        Rule #7: Check return values.
        Rule #9: Complete type hints.

        Args:
            capability: Required capability string.

        Returns:
            List of factory functions sorted by priority (highest first).
        """
        cls_names = self._enricher_capability_index.get(capability, [])

        factories = []
        for name in cls_names:
            entry = self._enricher_factories.get(name)
            if entry:
                factories.append(entry.factory)

        return factories

    def get_all_enricher_capabilities(self) -> List[str]:
        """
        Get list of all registered enricher capabilities.

        Enricher Registration Decorator.
        Rule #9: Complete type hints.

        Returns:
            List of capability strings.
        """
        return list(self._enricher_capability_index.keys())

    def clear(self) -> None:
        """
        Clear the registry (mostly for testing).

        Also resets health tracking state.
        """
        self._processors.clear()
        self._id_map.clear()
        self._capability_index.clear()
        self._enricher_factories.clear()
        self._enricher_capability_index.clear()
        # Maintain initialized state after clear
        IFRegistry._initialized = True

    @staticmethod
    def reset_for_worker() -> "IFRegistry":
        """
        Reset registry state for a new worker process.

        Called by orchestrator workers to ensure fresh state.
        JPL Rule #5: Returns fresh registry and asserts health.

        Returns:
            Fresh IFRegistry instance for this worker.
        """
        current_pid = os.getpid()

        # Force reset of singleton state
        IFRegistry._instance = None
        IFRegistry._reset_class_state()

        # Create new instance
        registry = IFRegistry()

        # JPL Rule #5: Assert health before returning
        registry.assert_healthy()

        logger.debug(f"Registry: Reset for worker process {current_pid}")
        return registry


def get_available_memory_mb() -> int:
    """
    Get available system memory in megabytes.

    Resources - Memory-Aware Selection.
    Rule #7: Check return values (returns sensible default on failure).

    Returns:
        Available memory in MB, or 1024 (1GB) as fallback.
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        # Return available memory in MB
        return int(mem.available / (1024 * 1024))
    except ImportError:
        # psutil not installed, return conservative default
        logger.warning("psutil not available, using default memory limit of 1024MB")
        return 1024
    except Exception as e:
        logger.warning(f"Failed to get system memory: {e}, using default 1024MB")
        return 1024


def register_if_processor(mime_types: List[str], priority: int = 100):
    """
    Decorator to register an IFProcessor.

    Validates that the class implements IFProcessor.
    """

    def decorator(cls: Type[IFProcessor]):
        assert issubclass(
            cls, IFProcessor
        ), f"{cls.__name__} must inherit from IFProcessor"

        # Instantiate the processor
        processor = cls()
        IFRegistry().register(processor, mime_types, priority)

        return cls

    return decorator


def register_processor(
    processor_id: str,
    capabilities: List[str],
    mime_types: List[str],
    priority: int = 100,
):
    """
    Decorator to register an IFProcessor with full metadata.

    Registry-Driven Discovery.
    Registers processor by MIME type and indexes by capability.

    Args:
        processor_id: Unique identifier for this processor.
        capabilities: List of capabilities this processor provides.
        mime_types: List of MIME types this processor handles.
        priority: Higher priority = selected first (default 100).

    Example::

        @register_processor(
            processor_id="pdf-processor",
            capabilities=["pdf-extraction", "ocr"],
            mime_types=["application/pdf"],
        )
        class PDFProcessor(IFProcessor): ...

    Rule #4: Function < 60 lines. Rule #9: Complete type hints.
    """

    def decorator(cls: Type[IFProcessor]) -> Type[IFProcessor]:
        assert issubclass(
            cls, IFProcessor
        ), f"{cls.__name__} must inherit from IFProcessor"

        # Instantiate the processor
        processor = cls()

        # Register with MIME types
        IFRegistry().register(processor, mime_types, priority)

        logger.debug(
            f"Registered processor: {processor_id} "
            f"capabilities={capabilities}, mime_types={mime_types}"
        )

        return cls

    return decorator


def register_enricher(
    capabilities: List[str],
    priority: int = 100,
    factory: Optional[Callable[..., IFProcessor]] = None,
):
    """
    Decorator to register an enricher processor by capability.

    Enricher Registration Decorator.
    Registers a factory for lazy instantiation, unlike @register_if_processor.

    Args:
        capabilities: List of capabilities this enricher provides.
        priority: Higher priority = selected first (default 100).
        factory: Optional factory function. If None, cls() is used.

    Example::

        @register_enricher(capabilities=["embedding"], factory=lambda c: Embed(c))
        class Embed(IFProcessor): ...

        # Retrieve: registry.get_enricher("embedding", config)

    Rule #4: Function < 60 lines. Rule #9: Complete type hints.
    """

    def decorator(cls: Type[IFProcessor]) -> Type[IFProcessor]:
        assert issubclass(
            cls, IFProcessor
        ), f"{cls.__name__} must inherit from IFProcessor"
        IFRegistry().register_enricher(
            cls=cls,
            capabilities=capabilities,
            priority=priority,
            factory=factory,
        )
        return cls

    return decorator


# =============================================================================
# AUTO-DISCOVERY OF PROCESSORS
# =============================================================================


def _auto_discover_processors() -> None:
    """
    Auto-discover and register processors from the processors package.

    Registry-Driven Discovery.
    Called lazily on first registry access to avoid circular imports.
    Rule #7: Check for import errors gracefully.
    """
    try:
        # Import processors package to trigger @register_processor decorators
        import ingestforge.processors  # noqa: F401

        logger.debug("Auto-discovered processors from ingestforge.processors")
    except ImportError as e:
        logger.debug(f"Processor auto-discovery skipped: {e}")

    # Also discover enrichers
    try:
        import ingestforge.enrichment.embeddings  # noqa: F401
        import ingestforge.enrichment.entities  # noqa: F401
        import ingestforge.enrichment.questions  # noqa: F401

        logger.debug("Auto-discovered enrichers from ingestforge.enrichment")
    except ImportError as e:
        logger.debug(f"Enricher auto-discovery skipped: {e}")


# JPL Rule #2: Fixed upper bounds for plugin discovery
MAX_PLUGIN_FILES = 100
MAX_PLUGIN_DEPTH = 3


def discover_plugins(plugin_dir: str) -> int:
    """
    Discover and load processor plugins from an external directory.

    AC#3: Plugin directory scanning for external processor modules.
    Rule #1: Iterative scanning (no recursion).
    Rule #2: Fixed upper bounds.
    Rule #7: Check return values.

    Args:
        plugin_dir: Path to directory containing plugin modules.

    Returns:
        Number of processors discovered and registered.
    """
    import importlib.util
    import sys

    plugin_path = Path(plugin_dir)
    if not plugin_path.exists():
        logger.warning(f"Plugin directory does not exist: {plugin_dir}")
        return 0

    if not plugin_path.is_dir():
        logger.warning(f"Plugin path is not a directory: {plugin_dir}")
        return 0

    registry = IFRegistry()
    count_before = len(registry._id_map)

    # Iterative directory scanning (JPL Rule #1: No recursion)
    dirs_to_scan = [(plugin_path, 0)]  # (path, depth)
    py_files: List[Path] = []

    while dirs_to_scan and len(py_files) < MAX_PLUGIN_FILES:
        current_dir, depth = dirs_to_scan.pop(0)

        if depth > MAX_PLUGIN_DEPTH:
            continue

        try:
            for item in current_dir.iterdir():
                if (
                    item.is_file()
                    and item.suffix == ".py"
                    and not item.name.startswith("_")
                ):
                    py_files.append(item)
                elif item.is_dir() and not item.name.startswith(("_", ".")):
                    dirs_to_scan.append((item, depth + 1))
        except PermissionError:
            logger.warning(f"Permission denied: {current_dir}")
            continue

    # Load each plugin module
    for py_file in py_files[:MAX_PLUGIN_FILES]:
        try:
            module_name = f"ingestforge_plugin_{py_file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            logger.debug(f"Loaded plugin module: {py_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load plugin {py_file}: {e}")

    count_after = len(registry._id_map)
    discovered = count_after - count_before

    logger.info(f"Plugin discovery complete: {discovered} processors from {plugin_dir}")
    return discovered
