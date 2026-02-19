"""
Main Pipeline class for document processing.

Uses IFPipelineRunner for stage orchestration (composition over inheritance).
Migrated from ChunkRecord to IFChunkArtifact for chunk representation.

Migration Notes ():
    - _create_chunk_record() → _create_chunk_artifact()
    - ChunkRecord import removed, now uses IFChunkArtifact from artifacts module
    - All chunk processing now uses IFChunkArtifact instances
    - Backward compatibility maintained through IFChunkArtifact.to_chunk_record()
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.core.config import Config

logger = get_logger(__name__)
from ingestforge.core.config_loaders import load_config
from ingestforge.core.provenance import SourceLocation
from ingestforge.core.security import SafeFileOperations
from ingestforge.core.state import (
    DocumentState,
    ProcessingStatus,
    StateManager,
)
from ingestforge.core.pipeline.result import PipelineResult
from ingestforge.core.pipeline.streaming import PipelineStreamingMixin
from ingestforge.core.pipeline.utils import PipelineUtilsMixin


def _process_file_worker(
    file_path: Path,
    config_dict: dict,
    base_path: Path,
) -> PipelineResult:
    """
    Worker function for parallel file processing.

    This is a module-level function to avoid pickle issues with ProcessPoolExecutor.
    Workers process files but do NOT update shared state - they return results
    to the main process which handles state updates.

    Args:
        file_path: Path to the document to process
        config_dict: Configuration as dict (for pickling)
        base_path: Base path for project

    Returns:
        PipelineResult with processing outcome
    """
    try:
        # Reconstruct config in worker process
        from ingestforge.core.config_loaders import load_config

        config = load_config(base_path=base_path)

        # Create pipeline with state updates disabled
        pipeline = Pipeline(config=config, base_path=base_path)
        pipeline._skip_state_updates = True

        return pipeline.process_file(file_path)
    except Exception as e:
        return PipelineResult(
            document_id=str(file_path),
            source_file=str(file_path),
            success=False,
            chunks_created=0,
            chunks_indexed=0,
            error_message=str(e),
        )


class Pipeline(
    PipelineStreamingMixin,
    PipelineUtilsMixin,
):
    """
    Main document processing pipeline.

    Orchestrates the full ingestion workflow:
    1. Split documents (PDF → chapters)
    2. Extract text (PDF/EPUB → markdown)
    3. Chunk content (semantic splitting)
    4. Enrich chunks (embeddings, entities, questions)
    5. Index chunks (storage backend)

    Uses IFPipelineRunner for stage orchestration (composition over inheritance).
    No longer inherits from PipelineSplittersMixin - uses registry dispatch.
    Supports context manager for automatic resource cleanup.
    Rule #4: Uses composition with PipelineStagesHelper instead of mixin inheritance.
    """

    # Flag to skip state updates (used by parallel workers)
    _skip_state_updates: bool = False

    def __init__(
        self,
        config: Optional[Config] = None,
        base_path: Optional[Path] = None,
    ):
        """
        Initialize pipeline.

        Initializes IFPipelineRunner and PipelineStagesHelper.

        Args:
            config: Configuration object. Loaded from config.yaml if not provided.
            base_path: Base path for project. Defaults to current directory.
        """
        from ingestforge.core.pipeline.runner import IFPipelineRunner
        from ingestforge.core.pipeline.stages_helper import PipelineStagesHelper

        self.base_path = base_path or Path.cwd()
        self.config = config or load_config(base_path=self.base_path)
        # Apply performance preset to adjust settings based on performance_mode
        self.config = apply_performance_preset(self.config)
        self.config.ensure_directories()

        # Safe file operations (prevents path traversal attacks)
        self._safe_ops = SafeFileOperations(self.base_path)

        # State management
        state_file = self.config.data_path / "pipeline_state.json"
        self.state_manager = StateManager(
            state_file, project_name=self.config.project.name
        )

        # Pipeline runner for stage orchestration
        self._runner = IFPipelineRunner(auto_teardown=True)

        # Stages helper for composition-based stage methods
        self._stages_helper = PipelineStagesHelper(self)

        # Lazy-loaded components
        self._splitter = None
        self._extractor = None
        self._refiner = None
        self._chunker = None
        self._enricher = None
        self._storage = None
        self._retriever = None

        # Progress callbacks
        self._progress_callback: Optional[Callable[[str, float, str], None]] = None

        # Internal state for stage processing
        self._current_chapter_markers: Dict[str, Any] = {}
        self._current_pdf_structure: Optional[Any] = None

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    def __enter__(self) -> "Pipeline":
        """
        Enter context manager.

        Processor Teardown in Pipeline.
        Rule #7: Return self for use in with statement.

        Returns:
            Self for use in with block.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """
        Exit context manager, performing resource cleanup.

        Processor Teardown in Pipeline.
        Calls teardown on all initialized components.
        Rule #7: Never suppresses exceptions.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Traceback if exception raised.

        Returns:
            False to not suppress exceptions.
        """
        self._teardown_components()
        return False  # Don't suppress exceptions

    def _teardown_components(self) -> bool:
        """
        Teardown all initialized pipeline components.

        Processor Teardown in Pipeline.
        Rule #1: Linear control flow.
        Rule #7: Exceptions are caught and logged.

        Returns:
            True if all teardowns successful, False if any failed.
        """
        all_success = True

        # Teardown enricher (IFEnrichmentStage)
        if self._enricher is not None and hasattr(self._enricher, "teardown"):
            try:
                if not self._enricher.teardown():
                    logger.warning("Enricher teardown returned False")
                    all_success = False
            except Exception as e:
                logger.warning(f"Enricher teardown failed: {e}")
                all_success = False

        # Teardown storage backend
        if self._storage is not None and hasattr(self._storage, "close"):
            try:
                self._storage.close()
            except Exception as e:
                logger.warning(f"Storage close failed: {e}")
                all_success = False

        # Teardown retriever
        if self._retriever is not None and hasattr(self._retriever, "close"):
            try:
                self._retriever.close()
            except Exception as e:
                logger.warning(f"Retriever close failed: {e}")
                all_success = False

        return all_success

    def teardown(self) -> bool:
        """
        Explicit teardown method for manual resource cleanup.

        Processor Teardown in Pipeline.
        Call this when not using context manager.
        Rule #7: Returns success status.

        Returns:
            True if all teardowns successful, False if any failed.
        """
        return self._teardown_components()

    # -------------------------------------------------------------------------
    # Registry-Driven Discovery
    # -------------------------------------------------------------------------

    @property
    def registry(self) -> "IFRegistry":
        """
        Get the processor registry for dispatch.

        Registry-Driven Discovery.
        Rule #7: Returns singleton registry instance.

        Returns:
            IFRegistry singleton instance.
        """
        from ingestforge.core.pipeline.registry import IFRegistry

        return IFRegistry()

    def dispatch(self, artifact: Any) -> Any:
        """
        Dispatch artifact to appropriate processor via registry.

        Registry-Driven Discovery.
        Rule #7: Check return values.

        Args:
            artifact: IFArtifact to process.

        Returns:
            IFProcessor instance for handling this artifact.

        Raises:
            RuntimeError: If no processor available.
        """
        return self.registry.dispatch(artifact)

    def dispatch_by_capability(self, capability: str, artifact: Any) -> Any:
        """
        Dispatch to processor with specific capability.

        Registry-Driven Discovery.
        Rule #7: Check return values.

        Args:
            capability: Required capability string.
            artifact: IFArtifact to process.

        Returns:
            IFProcessor instance with the capability.

        Raises:
            RuntimeError: If no processor available.
        """
        return self.registry.dispatch_by_capability(capability, artifact)

    # -------------------------------------------------------------------------
    # Legacy Component Properties (deprecated, use registry.dispatch)
    # -------------------------------------------------------------------------

    @property
    def splitter(self) -> Any:
        """Lazy-load PDF splitter.

        DEPRECATED: Use self.dispatch() or self.registry.dispatch() instead.
        JPL #5: Asserts config is valid before initializing component.
        """
        if self._splitter is None:
            assert (
                self.config is not None
            ), "Config must be set before accessing splitter"
            from ingestforge.ingest.pdf_splitter import PDFSplitter

            self._splitter = PDFSplitter(self.config)
        return self._splitter

    @property
    def extractor(self) -> Any:
        """Lazy-load text extractor.

        DEPRECATED: Use self.dispatch() or self.registry.dispatch() instead.
        JPL #5: Asserts config is valid before initializing component.
        """
        if self._extractor is None:
            assert (
                self.config is not None
            ), "Config must be set before accessing extractor"
            from ingestforge.ingest.text_extractor import TextExtractor

            self._extractor = TextExtractor(self.config)
        return self._extractor

    @property
    def audio_processor(self) -> Any:
        """Lazy-load audio processor for Whisper transcription.

        JPL #5: Asserts config is valid before initializing component.
        """
        if not hasattr(self, "_audio_processor") or self._audio_processor is None:
            assert (
                self.config is not None
            ), "Config must be set before accessing audio processor"
            from ingestforge.ingest.audio.processor import AudioProcessor

            # Get whisper settings from config
            model = getattr(self.config.ingest, "whisper_model", "base")
            lang = getattr(self.config.ingest, "whisper_language", "en")

            self._audio_processor = AudioProcessor(whisper_model=model, language=lang)
        return self._audio_processor

    @property
    def refiner(self) -> Any:
        """Lazy-load text refinement pipeline.

        JPL #5: Asserts config is valid before initializing component.
        """
        if self._refiner is None:
            assert (
                self.config is not None
            ), "Config must be set before accessing refiner"
            assert self.config.refinement is not None, "Refinement config must be set"
            from ingestforge.ingest.refinement import TextRefinementPipeline
            from ingestforge.ingest.refiners import (
                OCRCleanupRefiner,
                FormatNormalizer,
                ChapterDetector,
                TextCleanerRefiner,
            )

            refiners = []
            cfg = self.config.refinement

            if cfg.cleanup_ocr:
                refiners.append(OCRCleanupRefiner())

            if cfg.normalize_formatting:
                refiners.append(FormatNormalizer())

            # Text cleaning functions
            text_cleaner_enabled = (
                getattr(cfg, "group_paragraphs", True)
                or getattr(cfg, "clean_bullets", True)
                or getattr(cfg, "clean_prefix_postfix", True)
            )
            if text_cleaner_enabled:
                refiners.append(
                    TextCleanerRefiner(
                        group_paragraphs=getattr(cfg, "group_paragraphs", True),
                        clean_bullets=getattr(cfg, "clean_bullets", True),
                        clean_prefix_postfix=getattr(cfg, "clean_prefix_postfix", True),
                    )
                )

            if cfg.detect_chapters:
                refiners.append(ChapterDetector())

            self._refiner = TextRefinementPipeline(refiners, skip_unavailable=True)

        return self._refiner

    def _create_chunker(self) -> Any:
        """Create chunker instance based on strategy. Rule #1: Extracted helper."""
        strategy = self.config.chunking.strategy.lower()

        if strategy == "legal":
            from ingestforge.chunking.legal_chunker import LegalChunker

            return LegalChunker(self.config)

        if strategy == "code":
            from ingestforge.chunking.code_chunker import CodeChunker

            return CodeChunker(self.config)

        if strategy == "header":
            from ingestforge.chunking.header_chunker import HeaderChunker

            return HeaderChunker(
                min_chunk_size=self.config.chunking.min_size,
                max_chunk_size=self.config.chunking.max_size * 8,
                split_level=1,
            )

        # semantic, fixed, paragraph all use SemanticChunker
        from ingestforge.chunking.semantic_chunker import SemanticChunker

        return SemanticChunker(self.config)

    @property
    def chunker(self) -> Any:
        """Lazy-load chunker based on config.chunking.strategy.

        JPL #5: Asserts config is valid before initializing component.
        """
        if self._chunker is None:
            assert (
                self.config is not None
            ), "Config must be set before accessing chunker"
            assert self.config.chunking is not None, "Chunking config must be set"
            self._chunker = self._create_chunker()

        return self._chunker

    def _select_enricher_by_capability(
        self, capability: str, fallback_factory: Callable[[], Any]
    ) -> Optional[Any]:
        """
        Select a single enricher by capability with fallback.

        Prefers registry factory (get_enricher) over pre-instantiated.
        Rule #4: Function < 60 lines.

        Args:
            capability: Required capability string
            fallback_factory: Factory function to create fallback enricher

        Returns:
            Selected enricher instance or None
        """
        from ingestforge.core.pipeline.registry import IFRegistry

        registry = IFRegistry()

        # Try registry factory first (supports constructor args)
        enricher = registry.get_enricher(capability, self.config)
        if enricher is not None:
            logger.info(
                f"Selected enricher via registry factory: "
                f"{enricher.processor_id} (capability={capability})"
            )
            return enricher

        # Fallback: Try pre-instantiated processors (legacy path)
        candidates = registry.get_by_capability(capability)
        for proc in candidates:
            if proc.is_available():
                logger.info(
                    f"Selected processor via capability routing: "
                    f"{proc.processor_id} (capability={capability})"
                )
                return proc

        # No registered processor found - use fallback factory
        logger.debug(f"No registered processor for '{capability}' - using fallback")
        return fallback_factory()

    def _select_enrichers_by_capability(self) -> List[Any]:
        """
        Select enrichers using capability-based routing.

        Uses IFRegistry to dispatch processors by capability when available,
        falling back to hardcoded enrichers for backward compatibility.

        Returns:
            List of enricher instances to use in pipeline
        """
        enrichers = []

        # Embedding generation
        if self.config.enrichment.generate_embeddings:
            enricher = self._select_enricher_by_capability(
                "embedding", lambda: self._create_embedding_enricher()
            )
            if enricher:
                enrichers.append(enricher)

        # Entity extraction
        if self.config.enrichment.extract_entities:
            enricher = self._select_enricher_by_capability(
                "entity-extraction", lambda: self._create_entity_enricher()
            )
            if enricher:
                enrichers.append(enricher)

        # Question generation
        if self.config.enrichment.generate_questions:
            enricher = self._select_enricher_by_capability(
                "question-generation", lambda: self._create_question_enricher()
            )
            if enricher:
                enrichers.append(enricher)

        # Summary generation
        if self.config.enrichment.generate_summaries:
            enricher = self._select_enricher_by_capability(
                "summarization", lambda: self._create_summary_enricher()
            )
            if enricher:
                enrichers.append(enricher)

        # Instructor citation
        if self.config.enrichment.use_instructor_citation:
            enricher = self._select_enricher_by_capability(
                "citation", lambda: self._create_citation_enricher()
            )
            if enricher:
                enrichers.append(enricher)

        # Quality scoring
        if self.config.enrichment.compute_quality:
            enricher = self._select_enricher_by_capability(
                "quality-scoring", lambda: _QualityScorerAdapter()
            )
            if enricher:
                enrichers.append(enricher)

        # Week 8: Autonomous Domain Routing
        # Always enable dynamic enrichment to handle specialized verticals
        from ingestforge.enrichment.dynamic_enricher import DynamicDomainEnricher

        enrichers.append(DynamicDomainEnricher())

        return enrichers

    def _create_embedding_enricher(self) -> Any:
        """Create embedding enricher (fallback factory)."""
        from ingestforge.enrichment.embeddings import EmbeddingGenerator

        return EmbeddingGenerator(self.config)

    def _create_entity_enricher(self) -> Any:
        """Create entity enricher (fallback factory)."""
        from ingestforge.enrichment.entities import EntityExtractor

        return EntityExtractor()

    def _create_question_enricher(self) -> Any:
        """Create question enricher (fallback factory)."""
        from ingestforge.enrichment.questions import QuestionGenerator

        return QuestionGenerator(self.config)

    def _create_summary_enricher(self) -> Any:
        """Create summary enricher (fallback factory)."""
        from ingestforge.enrichment.summary import SummaryGenerator

        return SummaryGenerator(self.config)

    def _create_citation_enricher(self) -> Any:
        """Create citation enricher (fallback factory)."""
        from ingestforge.enrichment.instructor_citation import (
            InstructorCitationEnricher,
        )

        return InstructorCitationEnricher(self.config)

    @property
    def enricher(self) -> Any:
        """Lazy-load enrichment stage.

        Uses IFEnrichmentStage instead of deprecated EnrichmentPipeline.

        Builds an IFEnrichmentStage that chains all enabled IFProcessor enrichers
        based on config flags:
        - generate_embeddings → EmbeddingGenerator
        - extract_entities    → EntityExtractor
        - generate_questions  → QuestionGenerator
        - compute_quality     → QualityScorer (via adapter)

        Uses capability-based routing via IFRegistry when available,
        falling back to hardcoded enrichers for backward compatibility.

        JPL #5: Asserts config is valid before initializing component.
        """
        if self._enricher is None:
            assert (
                self.config is not None
            ), "Config must be set before accessing enricher"
            assert self.config.enrichment is not None, "Enrichment config must be set"
            from ingestforge.core.pipeline.enrichment_stage import IFEnrichmentStage

            processors = self._select_enrichers_by_capability()

            if processors:
                self._enricher = IFEnrichmentStage(
                    processors=processors,
                    skip_failures=True,
                    stage_name="enrichment",
                )
            else:
                # No enrichers enabled - use a pass-through
                self._enricher = _NoOpEnricher()

        return self._enricher

    @property
    def storage(self) -> Any:
        """Lazy-load storage backend.

        JPL #5: Asserts config is valid before initializing component.
        """
        if self._storage is None:
            assert (
                self.config is not None
            ), "Config must be set before accessing storage"
            assert self.config.storage is not None, "Storage config must be set"
            from ingestforge.storage.factory import get_storage_backend

            self._storage = get_storage_backend(self.config)
        return self._storage

    @property
    def retriever(self) -> Any:
        """Lazy-load retrieval engine.

        JPL #5: Asserts config and storage are valid before initializing.
        """
        if self._retriever is None:
            assert (
                self.config is not None
            ), "Config must be set before accessing retriever"
            assert self.config.retrieval is not None, "Retrieval config must be set"
            from ingestforge.retrieval.hybrid import HybridRetriever

            self._retriever = HybridRetriever(self.config, self.storage)
        return self._retriever

    # Map component names to the lazy-loaded attributes they affect
    COMPONENT_MAP = {
        "storage": ["_storage", "_retriever"],  # retriever depends on storage
        "retriever": ["_retriever"],
        "refiner": ["_refiner"],
        "chunker": ["_chunker"],
        "enricher": ["_enricher"],
        "splitter": ["_splitter"],
        "extractor": ["_extractor"],
    }

    def invalidate_component(self, name: str) -> None:
        """
        Reset lazy-loaded component(s) so they are re-created on next access.

        Used by the config API when Tier 2 settings change (e.g. storage.backend).
        Cascades: invalidating 'storage' also invalidates 'retriever'.
        """
        for attr in self.COMPONENT_MAP.get(name, []):
            setattr(self, attr, None)
        logger.info(f"Invalidated component: {name}")

    def set_progress_callback(
        self, callback: Callable[[str, float, str], None]
    ) -> None:
        """
        Set progress callback for UI updates.

        Implements Rule #4 (decouple progress from processing).

        The callback receives:
        - stage: Current processing stage (e.g., "splitting", "extracting", "chunking")
        - progress: Progress fraction (0.0 to 1.0)
        - message: Human-readable status message

        Args:
            callback: Function(stage, progress, message) called during processing

        Example:
            def progress_handler(stage: str, progress: float, message: str) -> None:
                print(f"[{stage}] {progress*100:.0f}%: {message}")

            pipeline.set_progress_callback(progress_handler)
            pipeline.process_file(file_path)
        """
        self._progress_callback = callback

    def clear_progress_callback(self) -> None:
        """Remove progress callback.

        Useful for disabling progress reporting temporarily.
        """
        self._progress_callback = None

    # -------------------------------------------------------------------------
    # Stage Method Shims (Backward Compatibility)
    # These delegate to PipelineStagesHelper for composition-based access.
    # -------------------------------------------------------------------------

    def _stage_split_document(
        self,
        file_path: Path,
        document_id: str,
        doc_state: DocumentState,
        plog: Any,
    ) -> tuple:
        """Stage 1: Split document. Delegates to stages helper."""
        return self._stages_helper.stage_split_document(
            file_path, document_id, doc_state, plog
        )

    def _stage_extract_text(
        self,
        chapters: List[Path],
        file_path: Path,
        context: Dict[str, Any],
        plog: Any,
    ) -> List[Dict[str, Any]]:
        """Stage 2: Extract text. Delegates to stages helper."""
        return self._stages_helper.stage_extract_text(
            chapters, file_path, context, plog
        )

    def _stage_refine_text(
        self,
        extracted_texts: List[Dict[str, Any]],
        file_path: Path,
        plog: Any,
    ) -> List[Dict[str, Any]]:
        """Stage 2.5: Refine text. Delegates to stages helper."""
        return self._stages_helper.stage_refine_text(extracted_texts, file_path, plog)

    def _stage_chunk_text(
        self,
        extracted_texts: List[Dict[str, Any]],
        document_id: str,
        file_path: Path,
        library: Optional[str],
        source_location: Optional[SourceLocation],
        doc_state: DocumentState,
        context: Dict[str, Any],
        plog: Any,
    ) -> List[Any]:
        """Stage 3: Chunk text. Delegates to stages helper."""
        return self._stages_helper.stage_chunk_text(
            extracted_texts,
            document_id,
            file_path,
            library,
            source_location,
            doc_state,
            context,
            plog,
        )

    def _stage_enrich_chunks(
        self,
        chunks: List[Any],
        context: Dict[str, Any],
        plog: Any,
    ) -> List[Any]:
        """Stage 4: Enrich chunks. Delegates to stages helper."""
        return self._stages_helper.stage_enrich_chunks(chunks, context, plog)

    def _stage_index_chunks(
        self,
        chunks: List[Any],
        file_path: Path,
        document_id: str,
        doc_state: DocumentState,
        plog: Any,
    ) -> int:
        """Stage 5: Index chunks. Delegates to stages helper."""
        return self._stages_helper.stage_index_chunks(
            chunks, file_path, document_id, doc_state, plog
        )

    # Sprint 3 (Rule #4): Utility methods moved to pipeline_utils.py
    # - _report_progress, _compute_file_hash, _get_hash_store
    # - _is_duplicate_content, _store_content_hash
    # - _generate_document_id, _extract_library_from_path
    # - _initialize_document_processing, _finalize_document_processing
    # - _move_to_completed, reset, _clear_data_directories, _clear_directory

    def process_file(
        self,
        file_path: Path,
        library: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process a single document through the full pipeline.

        Rule #4: Reduced from 62 → 40 lines via helper extraction

        Args:
            file_path: Path to the document to process.
            library: Target library name. If None, auto-detected from folder structure.

        Returns:
            PipelineResult with processing outcome.
        """
        import time

        start_time = time.time()
        if not file_path.exists():
            return self._create_file_not_found_result(file_path)

        document_id = self._generate_document_id(file_path)
        if self._is_duplicate_content(file_path, document_id):
            logger.info(f"Skipping duplicate content: {file_path.name}")
            return self._create_duplicate_content_result(document_id, file_path)

        # When running as parallel worker, skip state management
        # Main process will update state from results
        if self._skip_state_updates:
            return self._process_file_without_state(
                file_path, document_id, start_time, library
            )
        with self.state_manager.document(document_id, str(file_path)) as doc_state:
            try:
                result = self._process_document(
                    file_path, document_id, doc_state, start_time, library
                )
                self._store_content_hash(file_path)
                return result
            except Exception as e:
                logger.exception(f"Pipeline failed for {file_path}")
                doc_state.fail(str(e))
                return self._create_error_result(document_id, file_path, e, start_time)

    def _create_file_not_found_result(self, file_path: Path) -> PipelineResult:
        """
        Create result for non-existent file.

        Rule #4: Extracted to reduce process_file() size
        """
        return PipelineResult(
            document_id="",
            source_file=str(file_path),
            success=False,
            chunks_created=0,
            chunks_indexed=0,
            error_message=f"File not found: {file_path}",
        )

    def _create_duplicate_content_result(
        self, document_id: str, file_path: Path
    ) -> PipelineResult:
        """
        Create result for duplicate content.

        Rule #4: Extracted to reduce process_file() size
        """
        return PipelineResult(
            document_id=document_id,
            source_file=str(file_path),
            success=True,
            chunks_created=0,
            chunks_indexed=0,
            error_message="Skipped: duplicate content already indexed",
        )

    def _create_error_result(
        self, document_id: str, file_path: Path, error: Exception, start_time: float
    ) -> PipelineResult:
        """
        Create result for processing error.

        Rule #4: Extracted to reduce process_file() size
        """
        import time

        return PipelineResult(
            document_id=document_id,
            source_file=str(file_path),
            success=False,
            chunks_created=0,
            chunks_indexed=0,
            error_message=str(error),
            processing_time_sec=time.time() - start_time,
        )

    def _process_file_without_state(
        self,
        file_path: Path,
        document_id: str,
        start_time: float,
        library: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process file without state management.

        Used by parallel workers to avoid state file race conditions.
        Main process updates state from returned results.
        """
        import time
        from ingestforge.core.logging import PipelineLogger

        try:
            # Create a minimal doc_state for pipeline stages
            doc_state = DocumentState(
                document_id=document_id,
                source_file=str(file_path),
            )
            doc_state.start_processing()

            # Create pipeline logger
            plog = PipelineLogger(document_id, str(file_path))

            # Run the 5-stage pipeline
            # Stage 1: Split
            chapters, source_location, context = self._stage_split_document(
                file_path, document_id, doc_state, plog
            )

            # Stage 2: Extract
            doc_state.status = ProcessingStatus.EXTRACTING
            extracted_texts = self._stage_extract_text(
                chapters, file_path, context, plog
            )

            # Stage 2.5: Refine (optional)
            extracted_texts = self._stage_refine_text(extracted_texts, file_path, plog)

            # Stage 3: Chunk
            all_chunks = self._stage_chunk_text(
                extracted_texts,
                document_id,
                file_path,
                library,
                source_location,
                doc_state,
                context,
                plog,
            )

            # Stage 4: Enrich
            doc_state.status = ProcessingStatus.ENRICHING
            enriched_chunks = self._stage_enrich_chunks(all_chunks, context, plog)

            # Stage 5: Index
            indexed_count = self._stage_index_chunks(
                enriched_chunks, file_path, document_id, doc_state, plog
            )

            self._store_content_hash(file_path)

            return PipelineResult(
                document_id=document_id,
                source_file=str(file_path),
                success=True,
                chunks_created=len(all_chunks),
                chunks_indexed=indexed_count,
                processing_time_sec=time.time() - start_time,
            )

        except Exception as e:
            logger.exception(f"Pipeline failed for {file_path}")
            return self._create_error_result(document_id, file_path, e, start_time)

    # =========================================================================
    # IFPipelineRunner-Based Processing
    # =========================================================================

    def _process_document(
        self,
        file_path: Path,
        document_id: str,
        doc_state: DocumentState,
        start_time: float,
        library: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute the 5-stage pipeline using IFPipelineRunner.

        Uses runner.run() for stage orchestration.
        Rule #4: Function <60 lines.
        Rule #7: Check return values.
        """
        import time
        from ingestforge.core.pipeline.artifacts import IFFailureArtifact
        from ingestforge.core.pipeline.ingestion_stages import (
            IFPipelineContextArtifact,
            IFSplitStage,
            IFExtractStage,
            IFRefineStage,
            IFChunkStage,
            IFEnrichStageAdapter,
            IFIndexStage,
            create_initial_artifact,
        )

        # Initialize processing
        plog = self._initialize_document_processing(file_path, document_id, doc_state)

        # Create initial artifact
        initial_artifact = create_initial_artifact(file_path, document_id, library)

        # Assemble IFStage pipeline
        stages = [
            IFSplitStage(self, doc_state, plog),
            IFExtractStage(self, plog),
            IFRefineStage(self, plog),
            IFChunkStage(self, doc_state, plog),
            IFEnrichStageAdapter(self, plog),
            IFIndexStage(self, doc_state, plog),
        ]

        # Execute via runner.run()
        final_artifact = self._runner.run(initial_artifact, stages, document_id)

        # Map artifact to PipelineResult
        if isinstance(final_artifact, IFFailureArtifact):
            return PipelineResult(
                document_id=document_id,
                source_file=str(file_path),
                success=False,
                chunks_created=0,
                chunks_indexed=0,
                error_message=final_artifact.error_message,
                processing_time_sec=time.time() - start_time,
            )

        # Extract result fields from final context artifact
        if isinstance(final_artifact, IFPipelineContextArtifact):
            return PipelineResult(
                document_id=document_id,
                source_file=str(file_path),
                success=final_artifact.success,
                chunks_created=len(final_artifact.enriched_chunks),
                chunks_indexed=final_artifact.indexed_count,
                processing_time_sec=time.time() - start_time,
            )

        # Fallback for unexpected artifact type
        return self._finalize_document_processing(
            document_id, file_path, [], 0, doc_state, plog, start_time
        )

    def process_pending(
        self, parallel: bool = True, max_workers: Optional[int] = None
    ) -> List[PipelineResult]:
        """
        Process all pending documents in the ingest directory.

        Scans both the root pending directory and library subfolders
        (e.g., .ingest/pending/MyLibrary/).

        Args:
            parallel: Enable parallel processing (default: True)
            max_workers: Number of parallel workers (default: cpu_count - 1)

        Returns:
            List of PipelineResult for each processed document.
        """
        files = self._collect_pending_files()
        if not files:
            return []

        if parallel and len(files) > 1:
            return self._process_files_parallel(files, max_workers)

        return [self.process_file(f) for f in files]

    def _collect_pending_files(self) -> List[Path]:
        """Collect all supported files from pending directory."""
        pending_dir = self.config.pending_path
        supported = set(self.config.ingest.supported_formats)

        return [
            file_path
            for file_path in pending_dir.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in supported
        ]

    def _process_files_parallel(
        self, files: List[Path], max_workers: Optional[int]
    ) -> List[PipelineResult]:
        """
        Process files in parallel using ProcessPoolExecutor.

        Workers process files but do NOT update state. Results are returned to
        the main process which handles all state updates, preventing race conditions.
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)

        logger.info(
            f"Processing {len(files)} files in parallel with {max_workers} workers"
        )

        results = []
        config_dict = {}  # Placeholder for config serialization

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit work to workers using module-level function
            futures = {
                executor.submit(_process_file_worker, f, config_dict, self.base_path): f
                for f in files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Centralized state update in main process
                    self._update_state_from_result(result)

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    error_result = PipelineResult(
                        document_id=str(file_path),
                        source_file=str(file_path),
                        success=False,
                        chunks_created=0,
                        chunks_indexed=0,
                        error_message=str(e),
                    )
                    results.append(error_result)
                    self._update_state_from_result(error_result)

        # Final save to ensure all state is persisted
        self.state_manager.save()

        return results

    def _update_state_from_result(self, result: PipelineResult) -> None:
        """
        Update state from a pipeline result.

        This is called by the main process after receiving results from workers,
        ensuring all state updates happen in a single process with proper locking.

        Args:
            result: Pipeline result from processing a document
        """
        doc_state = self.state_manager.get_or_create_document(
            result.document_id, result.source_file
        )

        if result.success:
            doc_state.status = ProcessingStatus.COMPLETED
            doc_state.total_chunks = result.chunks_created
            doc_state.indexed_chunks = result.chunks_indexed
            doc_state.complete()
        else:
            doc_state.fail(result.error_message or "Unknown error")

        self.state_manager.update_document(doc_state)

    def _process_file_safe(self, file_path: Path) -> PipelineResult:
        """Wrapper for process_file that handles exceptions."""
        try:
            return self.process_file(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return PipelineResult(
                document_id=str(file_path),
                success=False,
                error=str(e),
                chunks_created=0,
                metadata={},
            )

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        library_filter: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Query the indexed documents.

        Args:
            query_text: Search query.
            top_k: Number of results to return. Defaults to config.
            library_filter: If provided, only return chunks from this library.

        Returns:
            List of search results with content and metadata.
        """
        if top_k is None:
            top_k = self.config.retrieval.top_k

        results = self.retriever.search(
            query_text, top_k=top_k, library_filter=library_filter, **kwargs
        )
        return [r.to_dict() for r in results]

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        state = self.state_manager.state

        return {
            "project_name": state.project_name,
            "total_documents": state.total_documents,
            "total_chunks": state.total_chunks,
            "total_embeddings": state.total_embeddings,
            "pending_count": len(state.get_pending_documents()),
            "in_progress_count": len(state.get_in_progress_documents()),
            "completed_count": len(state.get_completed_documents()),
            "failed_count": len(state.get_failed_documents()),
            "last_updated": state.last_updated,
        }

    def process_text(
        self,
        text: str,
        source: str = "api",
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Process raw text content directly without file ingestion.

        Rule #4: Function <60 lines (extracted helpers for validation and processing)
        Rule #7: Input validation before processing

        Args:
            text: Text content to process
            source: Source identifier (e.g., "ide", "api", "clipboard")
            title: Optional title for the content
            metadata: Additional metadata to attach

        Returns:
            PipelineResult with processing outcome

        Raises:
            ValueError: If text is empty or invalid
        """
        import time
        import hashlib

        self._validate_text_input(text)

        start_time = time.time()

        # Generate document ID from content hash
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        document_id = f"text_{source}_{content_hash}"

        try:
            result = self._execute_text_processing(
                text, source, title, metadata, document_id, start_time
            )
            return result

        except Exception as e:
            logger.exception(f"Text processing failed: {e}")
            return PipelineResult(
                document_id=document_id,
                source_file=source,
                success=False,
                chunks_created=0,
                chunks_indexed=0,
                error_message=str(e),
                processing_time_sec=time.time() - start_time,
            )

    def _validate_text_input(self, text: str) -> None:
        """
        Validate text input parameters.

        Rule #4: Extracted from process_text to reduce function size
        Rule #7: Input validation with clear error messages
        """
        if not text or not text.strip():
            raise ValueError("Text content cannot be empty")

        if len(text) > 1_000_000:  # 1MB limit
            raise ValueError(f"Text too large: {len(text)} chars (max 1,000,000)")

    def _execute_text_processing(
        self,
        text: str,
        source: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        document_id: str,
        start_time: float,
    ) -> PipelineResult:
        """
        Execute text processing pipeline.

        Rule #4: Extracted from process_text to reduce function size
        """
        import time
        from ingestforge.core.provenance import SourceLocation
        from ingestforge.core.logging import PipelineLogger
        from ingestforge.core.state import DocumentState, ProcessingStatus

        # Create source location
        source_location = SourceLocation(
            source_type=SourceType.API,
            source=source,
            title=title or f"Text from {source}",
        )

        # Chunk the text
        all_chunks = self._chunk_text_content(
            text, document_id, source, title, source_location, metadata
        )

        # Enrich chunks
        plog = PipelineLogger(document_id, source)
        enriched_chunks = self._stage_enrich_chunks(all_chunks, {}, plog)

        # Index chunks
        doc_state = DocumentState(document_id=document_id, source_file=source)
        doc_state.status = ProcessingStatus.INDEXING
        indexed_count = self._stage_index_chunks(
            enriched_chunks, Path(source), document_id, doc_state, plog
        )

        # Collect chunk IDs
        chunk_ids = [
            getattr(chunk, "artifact_id", getattr(chunk, "chunk_id", None))
            for chunk in enriched_chunks
        ]

        return PipelineResult(
            document_id=document_id,
            source_file=source,
            success=True,
            chunks_created=len(all_chunks),
            chunks_indexed=indexed_count,
            chunk_ids=chunk_ids,
            processing_time_sec=time.time() - start_time,
        )

    def _chunk_text_content(
        self,
        text: str,
        document_id: str,
        source: str,
        title: Optional[str],
        source_location: Any,
        metadata: Optional[Dict[str, Any]],
    ) -> List["IFChunkArtifact"]:
        """
        Chunk raw text content.

        Rule #4: Extracted helper to keep process_text <60 lines
        Returns IFChunkArtifact instances (migrated from ChunkRecord)
        Rule #9: Complete type hints with forward reference
        """

        # Use the chunker to split the text
        chunks = self.chunker.chunk_text(
            text=text,
            document_id=document_id,
            source_file=source,
        )

        # Enrich chunks with metadata
        all_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_artifact = self._create_chunk_artifact(
                i, chunk, document_id, source, title, source_location, metadata
            )
            all_chunks.append(chunk_artifact)

        return all_chunks

    def _create_chunk_artifact(
        self,
        index: int,
        chunk: Any,
        document_id: str,
        source: str,
        title: Optional[str],
        source_location: Any,
        metadata: Optional[Dict[str, Any]],
    ) -> "IFChunkArtifact":
        """Create an IFChunkArtifact from chunk data.

        Migrated from ChunkRecord to IFChunkArtifact.
        Rule #9: Complete type hints with forward reference.
        Rule #7: Returns IFChunkArtifact for type safety.

        Args:
            index: Chunk index
            chunk: Raw chunk
            document_id: Document ID
            source: Source file
            title: Document title
            source_location: Source location
            metadata: Additional metadata

        Returns:
            IFChunkArtifact instance
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        # Build metadata dictionary
        chunk_metadata: Dict[str, Any] = {
            "section_title": title or "Untitled",
            "chunk_type": "text",
            "source_file": source,
            "word_count": len(str(chunk).split()),
            "source_location": source_location,
            "library": metadata.get("library", "default") if metadata else "default",
        }

        # Add custom metadata if provided
        if metadata:
            for key, value in metadata.items():
                if key not in ("library",):
                    chunk_metadata[key] = value

        # Create IFChunkArtifact
        chunk_artifact = IFChunkArtifact(
            artifact_id=f"{document_id}_chunk_{index}",
            document_id=document_id,
            content=chunk.content if hasattr(chunk, "content") else str(chunk),
            chunk_index=index,
            total_chunks=1,  # Will be updated later if needed
            metadata=chunk_metadata,
        )

        return chunk_artifact


class _QualityScorerAdapter:
    """Internal adapter wrapping QualityScorer to match the IEnricher interface.

    This is an internal class used by the Pipeline to integrate the QualityScorer
    into the EnrichmentPipeline. It adapts the QualityScorer's score() method
    to the IEnricher.enrich_chunk() interface expected by EnrichmentPipeline.

    The adapter uses lazy initialization to avoid importing QualityScorer
    until it's actually needed.

    Note:
        This class is internal to the pipeline module and should not be
        imported or used directly by external code.
    """

    __slots__ = ("_scorer",)

    def __init__(self) -> None:
        self._scorer = None

    def _get_scorer(self):
        """Lazy-load the QualityScorer instance."""
        if self._scorer is None:
            from ingestforge.chunking.quality_scorer import QualityScorer

            self._scorer = QualityScorer()
        return self._scorer

    def enrich_chunk(self, chunk: Any) -> Any:
        """Enrich a single chunk by computing its quality score."""
        scorer = self._get_scorer()
        metrics = scorer.score(chunk)
        chunk.quality_score = metrics.overall_score
        return chunk

    def enrich_batch(
        self, chunks: Any, batch_size: Any = None, **kwargs: Any
    ) -> List[Any]:
        """Enrich a batch of chunks by computing quality scores."""
        scorer = self._get_scorer()
        return scorer.score_batch(chunks)

    def is_available(self) -> bool:
        """Quality scoring is always available (no external dependencies)."""
        return True

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata about this enricher."""
        return {"name": "QualityScorer", "available": True}

    def __repr__(self) -> str:
        return "QualityScorerAdapter(available)"


class _NoOpEnricher:
    """Internal pass-through enricher when no enrichment is enabled.

    This is an internal class that provides a no-op implementation of the
    IEnricher interface. Used by Pipeline.enricher when all enrichment
    options are disabled in config.

    The enricher simply returns chunks unchanged, avoiding None checks
    in the pipeline processing stages.

    Note:
        This class is internal to the pipeline module and should not be
        imported or used directly by external code.
    """

    __slots__ = ()

    def enrich_batch(self, chunks: Any, **kwargs: Any) -> List[Any]:
        """Return chunks unchanged."""
        return chunks

    def enrich_chunk(self, chunk: Any) -> Any:
        """Return chunk unchanged."""
        return chunk
