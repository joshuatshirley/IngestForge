"""
Pipeline Stages Helper.

Composition-based stage methods.
Registry-based document dispatch (replaces PipelineSplittersMixin).
Extracts stage logic from PipelineStagesMixin for use with IFPipelineRunner.

NASA JPL Power of Ten compliant.
"""

from datetime import datetime, timezone
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ingestforge.core.logging import PipelineLogger
from ingestforge.core.provenance import SourceLocation, SourceType
from ingestforge.core.state import DocumentState, ProcessingStatus

if TYPE_CHECKING:
    from ingestforge.core.pipeline.artifacts import IFTextArtifact, IFChunkArtifact

# JPL Rule #2: Fixed upper bounds for dispatch
MAX_DISPATCH_RETRIES = 3


class _Logger:
    """Lazy logger holder. Rule #6: Smallest scope."""

    _instance = None

    @classmethod
    def get(cls):
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


class PipelineStagesHelper:
    """
    Helper providing pipeline stage methods.

    Composition-based access to stage logic.
    Replaces PipelineStagesMixin inheritance.

    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(self, pipeline: Any) -> None:
        """
        Initialize stages helper.

        Args:
            pipeline: Pipeline instance providing components.
        """
        self._pipeline = pipeline

    # -------------------------------------------------------------------------
    # Stage 1: Split Document (Registry-based dispatch)
    # -------------------------------------------------------------------------

    def stage_split_document(
        self,
        file_path: Path,
        document_id: str,
        doc_state: DocumentState,
        plog: PipelineLogger,
    ) -> tuple[List[Path], Optional[SourceLocation], Dict[str, Any]]:
        """
        Stage 1: Split document based on file type using IFProcessor registry.

        Uses registry dispatch instead of mixin method.
        Rule #1: Dictionary dispatch eliminates nesting.
        Rule #4: Function <60 lines.
        """
        plog.start_stage("split")
        self._pipeline._report_progress("split", 0.0, f"Processing {file_path.name}")
        doc_state.status = ProcessingStatus.SPLITTING

        context: Dict[str, Any] = {}
        chapters, source_loc = self._dispatch_via_registry(
            file_path, document_id, doc_state, context
        )

        self._pipeline._report_progress(
            "split", 1.0, f"Split into {len(chapters)} parts"
        )
        plog.log_progress(f"Split into {len(chapters)} parts")

        return chapters, source_loc, context

    def _dispatch_via_registry(
        self,
        file_path: Path,
        document_id: str,
        doc_state: DocumentState,
        context: Dict[str, Any],
    ) -> tuple[List[Path], Optional[SourceLocation]]:
        """
        Dispatch to IFProcessor via registry or fallback to mixin.

        Registry-based processor dispatch.
        Rule #4: Function < 60 lines.
        Rule #7: Check return values.
        """
        from ingestforge.core.pipeline.registry import IFRegistry
        from ingestforge.core.pipeline.artifacts import IFFileArtifact

        suffix = file_path.suffix.lower()
        mime_type = self._get_mime_type(file_path, suffix)

        # Try registry dispatch first
        registry = IFRegistry()
        processors = registry.get_processors(mime_type)

        for processor in processors:
            if not processor.is_available():
                continue

            try:
                # Create IFFileArtifact from file path
                artifact = IFFileArtifact(
                    artifact_id=document_id,
                    file_path=str(file_path.absolute()),
                    mime_type=mime_type,
                    metadata={"document_id": document_id},
                )

                # Process via IFProcessor
                result = processor.process(artifact)

                # Convert result to legacy format
                return self._convert_processor_result(
                    result, file_path, doc_state, context, processor.processor_id
                )

            except Exception as e:
                _Logger.get().warning(
                    f"Processor {processor.processor_id} failed: {e}, trying next"
                )
                continue

        # Fallback: Use legacy mixin if available and no processor found
        if hasattr(self._pipeline, "_dispatch_document_splitter"):
            _Logger.get().debug(
                f"No registry processor for {mime_type}, using fallback"
            )
            return self._pipeline._dispatch_document_splitter(
                file_path, suffix, document_id, doc_state, context
            )

        # Default: return file as single chapter
        doc_state.chapters = [str(file_path)]
        return [file_path], None

    def _get_mime_type(self, file_path: Path, suffix: str) -> str:
        """
        Get MIME type for file.

        Rule #4: Function < 60 lines.
        """
        # Override common types
        mime_overrides = {
            ".pdf": "application/pdf",
            ".html": "text/html",
            ".htm": "text/html",
            ".mhtml": "text/html",
            ".md": "text/markdown",
            ".cls": "text/x-apex",
            ".trigger": "text/x-apex",
            ".js": "text/javascript",
        }

        if suffix in mime_overrides:
            return mime_overrides[suffix]

        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"

    def _convert_processor_result(
        self,
        result: Any,
        file_path: Path,
        doc_state: DocumentState,
        context: Dict[str, Any],
        processor_id: str,
    ) -> tuple[List[Path], Optional[SourceLocation]]:
        """
        Convert IFProcessor result to legacy tuple format.

        Adapter for backward compatibility.
        Rule #4: Function < 60 lines.
        """
        from ingestforge.core.pipeline.artifacts import (
            IFTextArtifact,
            IFFailureArtifact,
        )

        # Handle failure
        if isinstance(result, IFFailureArtifact):
            _Logger.get().warning(
                f"Processor {processor_id} returned failure: {result.error_message}"
            )
            doc_state.chapters = [str(file_path)]
            return [file_path], None

        # Handle text artifact (OCR/HTML/Code result)
        if isinstance(result, IFTextArtifact):
            doc_state.chapters = [str(file_path)]

            # Extract context from metadata
            metadata = result.metadata or {}

            # Store result in context for extraction stage
            if "ocr_result" in metadata or processor_id == "standard-pdf-processor":
                if metadata.get("is_scanned"):
                    context["scanned_pdf_ocr_result"] = result
            elif "html" in processor_id:
                self._pipeline._html_result = result
            elif "code" in processor_id:
                context["code_result"] = result
            elif "markdown" in processor_id and metadata.get("is_ado"):
                context["ado_result"] = result

            # Extract source location
            source_loc = self._extract_source_location(result, file_path)

            return [file_path], source_loc

        # Handle list of artifacts (chapters)
        if isinstance(result, list):
            chapter_paths = []
            for item in result:
                if isinstance(item, IFTextArtifact):
                    chapter_paths.append(
                        Path(item.metadata.get("source_path", str(file_path)))
                    )
                elif isinstance(item, Path):
                    chapter_paths.append(item)
            doc_state.chapters = [str(p) for p in chapter_paths]
            return chapter_paths, None

        # Unknown result type - use file as-is
        _Logger.get().warning(
            f"Unknown result type from {processor_id}: {type(result)}"
        )
        doc_state.chapters = [str(file_path)]
        return [file_path], None

    def _extract_source_location(
        self,
        artifact: Any,
        file_path: Path,
    ) -> Optional[SourceLocation]:
        """
        Extract SourceLocation from artifact metadata.

        Rule #4: Function < 60 lines.
        """
        metadata = getattr(artifact, "metadata", {}) or {}

        source_type_str = metadata.get("source_type", "FILE")
        try:
            source_type = SourceType[source_type_str.upper()]
        except (KeyError, AttributeError):
            source_type = SourceType.FILE

        return SourceLocation(
            source_type=source_type,
            title=metadata.get("title", file_path.stem),
            file_path=str(file_path),
            url=metadata.get("url"),
            authors=metadata.get("authors", []),
        )

    # -------------------------------------------------------------------------
    # Stage 2: Extract Text
    # -------------------------------------------------------------------------

    def stage_extract_text(
        self,
        chapters: List[Path],
        file_path: Path,
        context: Dict[str, Any],
        plog: PipelineLogger,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Extract text from chapters.

        d: Produces IFTextArtifact for each extraction.
        Rule #4: Function <60 lines.
        """
        plog.start_stage("extract")
        self._pipeline._report_progress("extract", 0.0, "Extracting text")

        suffix = file_path.suffix.lower()

        # Handle special cases
        if "scanned_pdf_ocr_result" in context:
            ocr_result = context["scanned_pdf_ocr_result"]
            self._pipeline._report_progress(
                "extract", 1.0, f"Extracted via OCR ({ocr_result.engine})"
            )
            artifact = self._create_text_artifact(ocr_result.text, file_path, "ocr")
            return [
                {"path": str(file_path), "text": ocr_result.text, "_artifact": artifact}
            ]

        if suffix in (".html", ".htm", ".mhtml") and hasattr(
            self._pipeline, "_html_result"
        ):
            self._pipeline._report_progress("extract", 1.0, "Extracted HTML content")
            artifact = self._create_text_artifact(
                self._pipeline._html_result.text, file_path, "html"
            )
            return [
                {
                    "path": str(file_path),
                    "text": self._pipeline._html_result.text,
                    "_artifact": artifact,
                }
            ]

        if "code_result" in context:
            code_result = context["code_result"]
            self._pipeline._report_progress("extract", 1.0, f"Extracted {suffix} code")
            artifact = self._create_text_artifact(code_result.text, file_path, "code")
            return [
                {
                    "path": str(file_path),
                    "text": code_result.text,
                    "metadata": code_result.metadata,
                    "_artifact": artifact,
                }
            ]

        if "ado_result" in context:
            ado_result = context["ado_result"]
            self._pipeline._report_progress("extract", 1.0, "Extracted ADO work item")
            artifact = self._create_text_artifact(ado_result.text, file_path, "ado")
            return [
                {
                    "path": str(file_path),
                    "text": ado_result.text,
                    "metadata": ado_result.metadata,
                    "_artifact": artifact,
                }
            ]

        if suffix in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"):
            return self._extract_audio(file_path, context)

        # Standard extraction
        return self._extract_standard(chapters, plog)

    def _extract_audio(
        self,
        file_path: Path,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract audio content. Rule #4: Helper to reduce size."""
        self._pipeline._report_progress("extract", 0.1, "Starting transcription")
        audio_result = self._pipeline.audio_processor.process(file_path)
        if not audio_result.success:
            raise ValueError(f"Transcription failed: {audio_result.error}")

        self._pipeline._report_progress(
            "extract", 1.0, f"Transcribed {audio_result.word_count} words"
        )
        context["audio_result"] = audio_result
        artifact = self._create_text_artifact(audio_result.text, file_path, "audio")
        return [
            {"path": str(file_path), "text": audio_result.text, "_artifact": artifact}
        ]

    def _extract_standard(
        self,
        chapters: List[Path],
        plog: PipelineLogger,
    ) -> List[Dict[str, Any]]:
        """Standard extraction loop. Rule #4: Helper to reduce size."""
        extracted_texts = []
        for i, chapter_path in enumerate(chapters):
            artifact = self._pipeline.extractor.extract_to_artifact(Path(chapter_path))
            extracted_texts.append(
                {
                    "path": str(chapter_path),
                    "text": artifact.content,
                    "_artifact": artifact,
                }
            )
            progress = (i + 1) / len(chapters)
            self._pipeline._report_progress(
                "extract", progress, f"Extracted {i + 1}/{len(chapters)}"
            )

        plog.log_progress(f"Extracted {len(extracted_texts)} text sections")
        return extracted_texts

    def _create_text_artifact(
        self,
        text: str,
        file_path: Path,
        extraction_method: str,
    ) -> "IFTextArtifact":
        """Create IFTextArtifact from extracted text."""
        from ingestforge.core.pipeline.artifact_factory import ArtifactFactory

        return ArtifactFactory.text_from_string(
            content=text,
            source_path=str(file_path.absolute()),
            metadata={
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "extraction_method": extraction_method,
                "word_count": len(text.split()),
                "char_count": len(text),
            },
        )

    # -------------------------------------------------------------------------
    # Stage 2.5: Refine Text
    # -------------------------------------------------------------------------

    def stage_refine_text(
        self,
        extracted_texts: List[Dict[str, Any]],
        file_path: Path,
        plog: PipelineLogger,
    ) -> List[Dict[str, Any]]:
        """Stage 2.5: Refine extracted text."""
        if not self._pipeline.config.refinement.enabled:
            return extracted_texts

        plog.start_stage("refine")
        self._pipeline._report_progress("refine", 0.0, "Refining text")

        suffix = file_path.suffix.lower()
        chapter_markers_by_path = {}

        for i, extracted in enumerate(extracted_texts):
            result = self._pipeline.refiner.refine(
                extracted["text"],
                metadata={"source": suffix, "path": extracted["path"]},
            )
            extracted["text"] = result.refined

            if result.chapter_markers:
                chapter_markers_by_path[extracted["path"]] = result.chapter_markers

            if result.changes:
                plog.log_progress(
                    f"Refined {extracted['path']}: {', '.join(result.changes[:3])}"
                )

            progress = (i + 1) / len(extracted_texts)
            self._pipeline._report_progress(
                "refine", progress, f"Refined {i + 1}/{len(extracted_texts)}"
            )

        self._pipeline._current_chapter_markers = chapter_markers_by_path
        plog.log_progress("Text refinement complete")

        return extracted_texts

    # -------------------------------------------------------------------------
    # Stage 3: Chunk Text
    # -------------------------------------------------------------------------

    def stage_chunk_text(
        self,
        extracted_texts: List[Dict[str, Any]],
        document_id: str,
        file_path: Path,
        library: Optional[str],
        source_location: Optional[SourceLocation],
        doc_state: DocumentState,
        context: Dict[str, Any],
        plog: PipelineLogger,
    ) -> List[Any]:
        """Stage 3: Chunk text into semantic units."""
        plog.start_stage("chunk")
        self._pipeline._report_progress("chunk", 0.0, "Chunking content")
        doc_state.status = ProcessingStatus.CHUNKING

        if library is None:
            library = self._pipeline._extract_library_from_path(file_path)
        if library != "default":
            _Logger.get().info(f"Document assigned to library: {library}")

        ingestion_time = datetime.now(timezone.utc).isoformat()

        if "audio_result" in context:
            return self._chunk_audio(
                context,
                document_id,
                library,
                ingestion_time,
                extracted_texts,
                doc_state,
                plog,
            )

        return self._chunk_standard(
            extracted_texts,
            document_id,
            file_path,
            library,
            source_location,
            ingestion_time,
            context,
            doc_state,
            plog,
        )

    def _chunk_audio(
        self,
        context: Dict[str, Any],
        document_id: str,
        library: str,
        ingestion_time: str,
        extracted_texts: List[Dict[str, Any]],
        doc_state: DocumentState,
        plog: PipelineLogger,
    ) -> List[Any]:
        """Chunk audio content. Rule #4: Helper."""
        audio_result = context["audio_result"]
        chunks = self._pipeline.audio_processor.to_chunk_records(
            audio_result, document_id
        )
        for chunk in chunks:
            chunk.library = library
            chunk.ingested_at = ingestion_time

        context["_chunk_artifacts"] = self._convert_chunks_to_artifacts(
            chunks, extracted_texts, context
        )

        doc_state.total_chunks = len(chunks)
        doc_state.chunk_ids = [c.chunk_id for c in chunks]
        plog.log_progress(f"Created {len(chunks)} timestamped audio chunks")
        return chunks

    def _chunk_standard(
        self,
        extracted_texts: List[Dict[str, Any]],
        document_id: str,
        file_path: Path,
        library: str,
        source_location: Optional[SourceLocation],
        ingestion_time: str,
        context: Dict[str, Any],
        doc_state: DocumentState,
        plog: PipelineLogger,
    ) -> List[Any]:
        """Standard chunking. Rule #4: Helper."""
        all_chunks = []
        all_artifacts: List["IFChunkArtifact"] = []

        for i, extracted in enumerate(extracted_texts):
            parent_artifact = extracted.get("_artifact")

            chunks = self._chunk_with_layout_awareness(
                extracted["text"],
                extracted["path"],
                document_id,
            )

            for chunk in chunks:
                chunk.library = library
                chunk.ingested_at = ingestion_time

            if source_location:
                chunks = self._attach_source_locations(chunks, source_location)

            all_chunks.extend(chunks)
            section_artifacts = self._chunks_to_artifacts(chunks, parent_artifact)
            all_artifacts.extend(section_artifacts)

            progress = (i + 1) / len(extracted_texts)
            self._pipeline._report_progress(
                "chunk", progress, f"Chunked {i + 1}/{len(extracted_texts)}"
            )

        all_chunks = self._optimize_chunks(all_chunks, plog)
        context["_chunk_artifacts"] = all_artifacts

        doc_state.total_chunks = len(all_chunks)
        doc_state.chunk_ids = [c.chunk_id for c in all_chunks]
        plog.log_progress(f"Created {len(all_chunks)} chunks")

        return all_chunks

    def _chunk_with_layout_awareness(
        self,
        text: str,
        source_path: str,
        document_id: str,
    ) -> List[Any]:
        """Chunk with layout awareness if markers available."""
        use_layout = getattr(
            getattr(self._pipeline.config, "chunking", None),
            "respect_section_boundaries",
            False,
        )

        chapter_markers = getattr(self._pipeline, "_current_chapter_markers", {})
        path_markers = chapter_markers.get(source_path, [])

        if use_layout and path_markers:
            try:
                from ingestforge.chunking.layout_chunker import LayoutChunker

                chunking_config = getattr(self._pipeline.config, "chunking", None)
                layout_chunker = LayoutChunker(
                    max_chunk_size=getattr(chunking_config, "max_size", 2000) * 4,
                    min_chunk_size=getattr(chunking_config, "min_size", 50) * 4,
                    combine_text_under_n_chars=getattr(
                        chunking_config, "combine_text_under_n_chars", 200
                    ),
                    respect_section_boundaries=True,
                    chunk_by_title=getattr(chunking_config, "chunk_by_title", False),
                )

                return layout_chunker.chunk_with_markers(
                    text,
                    path_markers,
                    document_id=document_id,
                    source_file=source_path,
                )
            except ImportError:
                _Logger.get().debug("LayoutChunker not available")
            except Exception as e:
                _Logger.get().warning(f"Layout chunking failed: {e}")

        return self._pipeline.chunker.chunk(
            text,
            document_id=document_id,
            source_file=source_path,
        )

    def _chunks_to_artifacts(
        self,
        chunks: List[Any],
        parent: Optional["IFTextArtifact"] = None,
    ) -> List["IFChunkArtifact"]:
        """Convert ChunkRecords to IFChunkArtifacts."""
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        artifacts: List["IFChunkArtifact"] = []
        for chunk in chunks:
            artifact = IFChunkArtifact.from_chunk_record(chunk, parent)
            artifacts.append(artifact)
        return artifacts

    def _convert_chunks_to_artifacts(
        self,
        chunks: List[Any],
        extracted_texts: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List["IFChunkArtifact"]:
        """Convert chunks to artifacts for audio."""
        parent = None
        if extracted_texts and "_artifact" in extracted_texts[0]:
            parent = extracted_texts[0]["_artifact"]
        return self._chunks_to_artifacts(chunks, parent)

    def _attach_source_locations(
        self,
        chunks: List[Any],
        source_location: SourceLocation,
    ) -> List[Any]:
        """Attach source location metadata to chunks."""
        for chunk in chunks:
            chapter = None
            section = chunk.section_title or None

            if (
                hasattr(self._pipeline, "_current_pdf_structure")
                and self._pipeline._current_pdf_structure
            ):
                if chunk.page_start:
                    location = (
                        self._pipeline._current_pdf_structure.get_location_for_page(
                            chunk.page_start
                        )
                    )
                    chapter = location.get("chapter")
                    section = location.get("section") or section

            chunk.source_location = SourceLocation(
                source_id=source_location.source_id,
                source_type=source_location.source_type,
                title=source_location.title,
                authors=source_location.authors,
                publication_date=source_location.publication_date,
                url=source_location.url,
                file_path=source_location.file_path,
                accessed_date=source_location.accessed_date,
                chapter=chapter,
                section=section,
                page_start=chunk.page_start if chunk.page_start else None,
                page_end=chunk.page_end if chunk.page_end else None,
            )

        return chunks

    def _optimize_chunks(
        self,
        chunks: List[Any],
        plog: PipelineLogger,
    ) -> List[Any]:
        """Optimize chunks with size optimizer and deduplicator."""
        if not chunks:
            return chunks

        try:
            from ingestforge.chunking.size_optimizer import SizeOptimizer

            optimizer = SizeOptimizer(self._pipeline.config)
            chunks, opt_report = optimizer.optimize(chunks)
            if opt_report.chunks_split or opt_report.chunks_merged:
                plog.log_progress(
                    f"Size optimizer: {opt_report.chunks_split} splits, "
                    f"{opt_report.chunks_merged} merges"
                )
        except Exception as e:
            _Logger.get().warning(f"Size optimization skipped: {e}")

        try:
            from ingestforge.chunking.deduplicator import Deduplicator

            deduplicator = Deduplicator()
            chunks, dedup_report = deduplicator.deduplicate(chunks)
            if dedup_report.duplicates_removed > 0:
                plog.log_progress(
                    f"Deduplicator: removed {dedup_report.duplicates_removed} duplicates"
                )
        except Exception as e:
            _Logger.get().warning(f"Deduplication skipped: {e}")

        return chunks

    # -------------------------------------------------------------------------
    # Stage 4: Enrich Chunks
    # -------------------------------------------------------------------------

    def stage_enrich_chunks(
        self,
        chunks: List[Any],
        context: Dict[str, Any],
        plog: PipelineLogger,
    ) -> List[Any]:
        """Stage 4: Enrich chunks with embeddings."""
        from ingestforge.core.pipeline.interfaces import IFStage

        plog.start_stage("enrich")
        self._pipeline._report_progress("enrich", 0.0, "Generating embeddings")

        if not chunks:
            return chunks

        max_batch = getattr(
            self._pipeline.config.enrichment, "enrichment_max_batch_size", 500
        )

        if isinstance(self._pipeline.enricher, IFStage):
            enriched_chunks = self._enrich_via_stage(chunks)
        elif len(chunks) <= max_batch:
            enriched_chunks = self._pipeline.enricher.enrich_batch(chunks)
        else:
            enriched_chunks = self._enrich_in_batches(chunks, max_batch, plog)

        artifacts = context.get("_chunk_artifacts", [])
        if artifacts:
            enriched_artifacts = self._sync_enrichment_to_artifacts(
                enriched_chunks, artifacts
            )
            context["_chunk_artifacts"] = enriched_artifacts
            plog.log_progress(
                f"Synced enrichment to {len(enriched_artifacts)} artifacts"
            )

        self._pipeline._report_progress(
            "enrich", 1.0, f"Enriched {len(enriched_chunks)} chunks"
        )
        return enriched_chunks

    def _enrich_in_batches(
        self,
        chunks: List[Any],
        max_batch: int,
        plog: PipelineLogger,
    ) -> List[Any]:
        """Enrich in sub-batches for large documents."""
        _Logger.get().info(
            f"Large document: {len(chunks)} chunks, processing in batches of {max_batch}"
        )
        enriched_all: List[Any] = []
        total_batches = (len(chunks) + max_batch - 1) // max_batch

        for batch_idx in range(0, len(chunks), max_batch):
            batch = chunks[batch_idx : batch_idx + max_batch]
            batch_num = batch_idx // max_batch + 1

            enriched_batch = self._pipeline.enricher.enrich_batch(batch)
            enriched_all.extend(enriched_batch)

            progress = batch_num / total_batches
            self._pipeline._report_progress(
                "enrich", progress, f"Enriched batch {batch_num}/{total_batches}"
            )
            plog.log_progress(f"Enriched batch {batch_num}/{total_batches}")

        return enriched_all

    def _enrich_via_stage(self, chunks: List[Any]) -> List[Any]:
        """Enrich via IFStage.execute() interface."""
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        enriched: List[Any] = []
        for chunk in chunks:
            artifact = IFChunkArtifact.from_chunk_record(chunk)
            result = self._pipeline.enricher.execute(artifact)

            if isinstance(result, IFChunkArtifact):
                enriched.append(result.to_chunk_record())
            else:
                _Logger.get().warning(
                    f"Enrichment returned {type(result).__name__}, keeping original"
                )
                enriched.append(chunk)

        return enriched

    def _sync_enrichment_to_artifacts(
        self,
        enriched_chunks: List[Any],
        artifacts: List["IFChunkArtifact"],
    ) -> List["IFChunkArtifact"]:
        """Sync enrichment data to artifacts."""
        if not enriched_chunks or not artifacts:
            return artifacts

        chunk_map: Dict[str, Any] = {}
        for chunk in enriched_chunks:
            chunk_id = getattr(chunk, "chunk_id", None)
            if chunk_id:
                chunk_map[chunk_id] = chunk

        enriched_artifacts: List["IFChunkArtifact"] = []

        for artifact in artifacts:
            chunk = chunk_map.get(artifact.artifact_id)
            if chunk is None:
                enriched_artifacts.append(artifact)
                continue

            updated_metadata = dict(artifact.metadata)

            if hasattr(chunk, "embedding") and chunk.embedding:
                updated_metadata["embedding"] = chunk.embedding
            if hasattr(chunk, "entities") and chunk.entities:
                updated_metadata["entities"] = chunk.entities
            if hasattr(chunk, "concepts") and chunk.concepts:
                updated_metadata["concepts"] = chunk.concepts
            if hasattr(chunk, "quality_score") and chunk.quality_score:
                updated_metadata["quality_score"] = chunk.quality_score

            enriched = artifact.derive(
                "enricher",
                metadata=updated_metadata,
            )
            enriched_artifacts.append(enriched)

        return enriched_artifacts

    # -------------------------------------------------------------------------
    # Stage 5: Index Chunks
    # -------------------------------------------------------------------------

    def stage_index_chunks(
        self,
        chunks: List[Any],
        file_path: Path,
        document_id: str,
        doc_state: DocumentState,
        plog: PipelineLogger,
    ) -> int:
        """Stage 5: Index chunks into storage."""
        plog.start_stage("index")
        self._pipeline._report_progress("index", 0.0, "Indexing chunks")
        doc_state.status = ProcessingStatus.INDEXING

        indexed_count = self._pipeline.storage.add_chunks(chunks)
        doc_state.indexed_chunks = indexed_count

        self._pipeline._report_progress("index", 1.0, f"Indexed {indexed_count} chunks")

        if self._pipeline.config.ingest.move_completed:
            self._pipeline._move_to_completed(file_path, document_id)

        return indexed_count
