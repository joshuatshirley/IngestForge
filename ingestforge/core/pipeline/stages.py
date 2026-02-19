"""
Processing Stages Mixin for Pipeline.

Handles the main processing stages:
- Stage 1: Split Document
- Stage 2: Extract Text (d: produces IFTextArtifact)
- Stage 2.5: Refine Text
- Stage 3: Chunk Text (e: creates IFChunkArtifact)
- Stage 4: Enrich Chunks (f: syncs to IFChunkArtifact)
- Stage 5: Index Chunks

Migration Status (TASK-011):
    - No direct ChunkRecord imports (migrated to IFChunkArtifact)
    - Uses List[Any] type hints for backward compatibility during transition
    - Internally converts between ChunkRecord and IFChunkArtifact as needed
    - Full IFChunkArtifact support with lineage tracking

This module is part of the Pipeline refactoring (Sprint 3, Rule #4)
to reduce pipeline.py from 1680 lines to <400 lines.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ingestforge.core.logging import PipelineLogger
from ingestforge.core.provenance import SourceLocation
from ingestforge.core.state import DocumentState, ProcessingStatus

if TYPE_CHECKING:
    from ingestforge.core.pipeline.artifacts import IFTextArtifact, IFChunkArtifact

from ingestforge.core.pipeline.interfaces import IFStage


class _Logger:
    """Lazy logger holder.

    Rule #6: Encapsulates logger state in smallest scope.
    Avoids slow startup from rich library import.
    """

    _instance = None

    @classmethod
    def get(cls):
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


class PipelineStagesMixin:
    """
    Mixin providing processing stage methods for Pipeline.

    Rule #4: Extracted from pipeline.py to reduce file size
    """

    def _stage_split_document(
        self,
        file_path: Path,
        document_id: str,
        doc_state: DocumentState,
        plog: PipelineLogger,
    ) -> tuple[List[Path], Optional[SourceLocation], Dict[str, Any]]:
        """
        Stage 1: Split document based on file type.

        Rule #1: Dictionary dispatch eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        plog.start_stage("split")
        self._report_progress("split", 0.0, f"Processing {file_path.name}")
        doc_state.status = ProcessingStatus.SPLITTING

        suffix = file_path.suffix.lower()
        context: Dict[str, Any] = {}
        chapters, source_loc = self._dispatch_document_splitter(
            file_path, suffix, document_id, doc_state, context
        )

        self._report_progress("split", 1.0, f"Split into {len(chapters)} parts")
        plog.log_progress(f"Split into {len(chapters)} parts")

        return chapters, source_loc, context

    def _stage_extract_text(
        self,
        chapters: List[Path],
        file_path: Path,
        context: Dict[str, Any],
        plog: PipelineLogger,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Extract text from chapters.

        d: Now produces IFTextArtifact for each extraction.
        Output dicts include '_artifact' key for artifact access.
        Backward compatible: 'text' key still provides raw string.

        Rule #1: Early return reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        plog.start_stage("extract")
        self._report_progress("extract", 0.0, "Extracting text")

        suffix = file_path.suffix.lower()
        if "scanned_pdf_ocr_result" in context:
            ocr_result = context["scanned_pdf_ocr_result"]
            self._report_progress(
                "extract", 1.0, f"Extracted via OCR ({ocr_result.engine})"
            )
            artifact = self._create_text_artifact(ocr_result.text, file_path, "ocr")
            return [
                {"path": str(file_path), "text": ocr_result.text, "_artifact": artifact}
            ]

        if suffix in (".html", ".htm", ".mhtml") and hasattr(self, "_html_result"):
            self._report_progress("extract", 1.0, "Extracted HTML content")
            artifact = self._create_text_artifact(
                self._html_result.text, file_path, "html"
            )
            return [
                {
                    "path": str(file_path),
                    "text": self._html_result.text,
                    "_artifact": artifact,
                }
            ]

        if "code_result" in context:
            code_result = context["code_result"]
            self._report_progress("extract", 1.0, f"Extracted {suffix} code")
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
            self._report_progress("extract", 1.0, "Extracted ADO work item")
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
            self._report_progress("extract", 0.1, "Starting transcription")
            audio_result = self.audio_processor.process(file_path)
            if not audio_result.success:
                raise ValueError(f"Transcription failed: {audio_result.error}")

            self._report_progress(
                "extract", 1.0, f"Transcribed {audio_result.word_count} words"
            )
            context["audio_result"] = audio_result
            artifact = self._create_text_artifact(audio_result.text, file_path, "audio")
            return [
                {
                    "path": str(file_path),
                    "text": audio_result.text,
                    "_artifact": artifact,
                }
            ]

        # Standard extraction loop with artifacts
        extracted_texts = []
        for i, chapter_path in enumerate(chapters):
            artifact = self.extractor.extract_to_artifact(Path(chapter_path))
            extracted_texts.append(
                {
                    "path": str(chapter_path),
                    "text": artifact.content,
                    "_artifact": artifact,
                }
            )
            progress = (i + 1) / len(chapters)
            self._report_progress(
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
        """
        Create IFTextArtifact from extracted text.

        d: Helper for special extraction cases.
        Rule #4: Function <60 lines.
        Rule #7: Explicit return type.

        Args:
            text: Extracted text content.
            file_path: Source file path.
            extraction_method: Method used for extraction.

        Returns:
            IFTextArtifact with metadata.
        """
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

    def _stage_refine_text(
        self,
        extracted_texts: List[Dict[str, Any]],
        file_path: Path,
        plog: PipelineLogger,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2.5: Refine extracted text.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        if not self.config.refinement.enabled:
            return extracted_texts

        plog.start_stage("refine")
        self._report_progress("refine", 0.0, "Refining text")

        suffix = file_path.suffix.lower()
        chapter_markers_by_path = {}

        for i, extracted in enumerate(extracted_texts):
            result = self.refiner.refine(
                extracted["text"],
                metadata={"source": suffix, "path": extracted["path"]},
            )
            extracted["text"] = result.refined

            # Store chapter markers
            if result.chapter_markers:
                chapter_markers_by_path[extracted["path"]] = result.chapter_markers

            if result.changes:
                plog.log_progress(
                    f"Refined {extracted['path']}: {', '.join(result.changes[:3])}"
                )

            progress = (i + 1) / len(extracted_texts)
            self._report_progress(
                "refine", progress, f"Refined {i + 1}/{len(extracted_texts)}"
            )

        self._current_chapter_markers = chapter_markers_by_path
        plog.log_progress("Text refinement complete")

        return extracted_texts

    def _stage_chunk_text(
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
        """
        Stage 3: Chunk text into semantic units.

        e: Creates IFChunkArtifact internally and stores in context.
        Returns legacy ChunkRecord objects for Stage 4/5 backward compatibility.

        Note: List[Any] type hint is intentional - supports both ChunkRecord
        and IFChunkArtifact during migration period.

        Rule #4: Function <60 lines
        Rule #9: Full type hints (Any used for migration compatibility)
        """
        plog.start_stage("chunk")
        self._report_progress("chunk", 0.0, "Chunking content")
        doc_state.status = ProcessingStatus.CHUNKING

        # Determine library
        if library is None:
            library = self._extract_library_from_path(file_path)
        if library != "default":
            _Logger.get().info(f"Document assigned to library: {library}")

        ingestion_time = datetime.now(timezone.utc).isoformat()
        if "audio_result" in context:
            audio_result = context["audio_result"]
            chunks = self.audio_processor.to_chunk_records(audio_result, document_id)
            for chunk in chunks:
                chunk.library = library
                chunk.ingested_at = ingestion_time

            # e: Create artifacts for audio chunks
            context["_chunk_artifacts"] = self._convert_chunks_to_artifacts(
                chunks, extracted_texts, context
            )

            doc_state.total_chunks = len(chunks)
            doc_state.chunk_ids = [c.chunk_id for c in chunks]
            plog.log_progress(f"Created {len(chunks)} timestamped audio chunks")
            return chunks

        # Chunk all sections with artifact tracking
        all_chunks = []
        all_artifacts: List["IFChunkArtifact"] = []

        for i, extracted in enumerate(extracted_texts):
            # e: Get parent artifact from Stage 2 output
            parent_artifact = extracted.get("_artifact")

            # Use layout-aware chunking if chapter markers available
            chunks = self._chunk_with_layout_awareness(
                extracted["text"],
                extracted["path"],
                document_id,
            )

            # Set metadata on chunks
            for chunk in chunks:
                chunk.library = library
                chunk.ingested_at = ingestion_time

            # Attach source locations
            if source_location:
                chunks = self._attach_source_locations(chunks, source_location)

            all_chunks.extend(chunks)

            # e: Create artifacts from chunks with lineage
            section_artifacts = self._chunks_to_artifacts(chunks, parent_artifact)
            all_artifacts.extend(section_artifacts)

            progress = (i + 1) / len(extracted_texts)
            self._report_progress(
                "chunk", progress, f"Chunked {i + 1}/{len(extracted_texts)}"
            )

        # Optimize and deduplicate
        all_chunks = self._optimize_chunks(all_chunks, plog)

        # e: Store artifacts in context for Stage 4 access
        context["_chunk_artifacts"] = all_artifacts

        doc_state.total_chunks = len(all_chunks)
        doc_state.chunk_ids = [c.chunk_id for c in all_chunks]
        plog.log_progress(f"Created {len(all_chunks)} chunks")

        return all_chunks

    def _chunks_to_artifacts(
        self,
        chunks: List[Any],
        parent: Optional["IFTextArtifact"] = None,
    ) -> List["IFChunkArtifact"]:
        """
        Convert legacy ChunkRecord objects to IFChunkArtifacts with lineage.

        e: Helper for artifact conversion during migration.
        Rule #4: Function <60 lines.
        Rule #7: Explicit return type.

        Args:
            chunks: List[Any] - Legacy ChunkRecord objects (type hint is Any for migration).
            parent: Optional parent text artifact for lineage tracking.

        Returns:
            List of IFChunkArtifact with lineage preserved.
        """
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
        """
        Convert legacy chunks to artifacts for special cases (audio, etc.).

        e: Helper for non-standard chunking paths during migration.
        Rule #4: Function <60 lines.
        Rule #7: Explicit return type.

        Args:
            chunks: List[Any] - Legacy ChunkRecord objects (type hint is Any for migration).
            extracted_texts: Stage 2 output (may contain parent artifacts).
            context: Pipeline context for artifact storage.

        Returns:
            List of IFChunkArtifact with proper lineage.
        """
        # Get parent artifact if available
        parent = None
        if extracted_texts and "_artifact" in extracted_texts[0]:
            parent = extracted_texts[0]["_artifact"]

        return self._chunks_to_artifacts(chunks, parent)

    def _attach_source_locations(
        self,
        chunks: List[Any],
        source_location: SourceLocation,
    ) -> List[Any]:
        """
        Attach source location metadata to chunks.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        for chunk in chunks:
            chapter = None
            section = chunk.section_title or None

            # Get location from PDF structure
            if hasattr(self, "_current_pdf_structure") and self._current_pdf_structure:
                if chunk.page_start:
                    location = self._current_pdf_structure.get_location_for_page(
                        chunk.page_start
                    )
                    chapter = location.get("chapter")
                    section = location.get("section") or section

            # Create chunk-specific source location
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

    def _chunk_with_layout_awareness(
        self,
        text: str,
        source_path: str,
        document_id: str,
    ) -> List[Any]:
        """
        Chunk text using layout-aware chunking if chapter markers available.

        Falls back to standard chunker if no markers or layout chunking disabled.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        # Check if we should use layout-aware chunking
        use_layout = getattr(
            getattr(self.config, "chunking", None), "respect_section_boundaries", False
        )

        chapter_markers = getattr(self, "_current_chapter_markers", {})
        path_markers = chapter_markers.get(source_path, [])

        # Use layout chunker if enabled and markers available
        if use_layout and path_markers:
            try:
                from ingestforge.chunking.layout_chunker import LayoutChunker

                chunking_config = getattr(self.config, "chunking", None)
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
                _Logger.get().debug(
                    "LayoutChunker not available, using standard chunker"
                )
            except Exception as e:
                _Logger.get().warning(f"Layout chunking failed, falling back: {e}")

        # Fall back to standard chunker
        return self.chunker.chunk(
            text,
            document_id=document_id,
            source_file=source_path,
        )

    def _optimize_chunks(
        self,
        chunks: List[Any],
        plog: PipelineLogger,
    ) -> List[Any]:
        """
        Optimize chunks with size optimizer and deduplicator.

        Rule #1: Early return for empty chunks
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        if not chunks:
            return chunks

        # Size optimization
        try:
            from ingestforge.chunking.size_optimizer import SizeOptimizer

            optimizer = SizeOptimizer(self.config)
            chunks, opt_report = optimizer.optimize(chunks)
            if opt_report.chunks_split or opt_report.chunks_merged:
                plog.log_progress(
                    f"Size optimizer: {opt_report.chunks_split} splits, "
                    f"{opt_report.chunks_merged} merges"
                )
        except Exception as e:
            _Logger.get().warning(f"Chunk size optimization skipped: {e}")

        # Deduplication
        try:
            from ingestforge.chunking.deduplicator import Deduplicator

            deduplicator = Deduplicator()
            chunks, dedup_report = deduplicator.deduplicate(chunks)
            if dedup_report.duplicates_removed > 0:
                plog.log_progress(
                    f"Deduplicator: removed {dedup_report.duplicates_removed} duplicates"
                )
        except Exception as e:
            _Logger.get().warning(f"Chunk deduplication skipped: {e}")

        return chunks

    def _stage_enrich_chunks(
        self,
        chunks: List[Any],
        context: Dict[str, Any],
        plog: PipelineLogger,
    ) -> List[Any]:
        """
        Stage 4: Enrich chunks with embeddings and metadata.

        f: Syncs enrichment results to IFChunkArtifact in context.
        Uses sub-batching for large documents to prevent memory issues.
        Configurable via enrichment.enrichment_max_batch_size (default: 500).

        Args:
            chunks: List[Any] - Legacy ChunkRecord objects (type hint is Any for migration).
            context: Pipeline context containing _chunk_artifacts.
            plog: Pipeline logger for progress tracking.

        Returns:
            List[Any] - Enriched legacy ChunkRecord objects for Stage 5 compatibility.

        Rule #4: Function <60 lines
        Rule #9: Full type hints (Any used for migration compatibility)
        """
        plog.start_stage("enrich")
        self._report_progress("enrich", 0.0, "Generating embeddings")

        if not chunks:
            return chunks

        # Get max batch size from config (default 500)
        max_batch = getattr(self.config.enrichment, "enrichment_max_batch_size", 500)

        # BUG001: IFEnrichmentStage provides enrich_batch() for backward compatibility.
        # It internally handles the ChunkRecord <-> IFArtifact conversion.
        if hasattr(self.enricher, "enrich_batch"):
            enriched_chunks = self.enricher.enrich_batch(chunks)
        elif isinstance(self.enricher, IFStage):
            enriched_chunks = self._enrich_via_stage(chunks)
        elif len(chunks) <= max_batch:
            enriched_chunks = self.enricher.enrich_batch(chunks)
        else:
            enriched_chunks = self._enrich_in_batches(chunks, max_batch, plog)

        # f: Sync enrichment results to artifacts in context
        artifacts = context.get("_chunk_artifacts", [])
        if artifacts:
            enriched_artifacts = self._sync_enrichment_to_artifacts(
                enriched_chunks, artifacts
            )
            context["_chunk_artifacts"] = enriched_artifacts
            plog.log_progress(
                f"Synced enrichment to {len(enriched_artifacts)} artifacts"
            )

        self._report_progress("enrich", 1.0, f"Enriched {len(enriched_chunks)} chunks")
        return enriched_chunks

    def _enrich_in_batches(
        self,
        chunks: List[Any],
        max_batch: int,
        plog: PipelineLogger,
    ) -> List[Any]:
        """
        Enrich chunks in sub-batches for large documents.

        f: Extracted from _stage_enrich_chunks for Rule #4 compliance.
        Rule #4: Function <60 lines.

        Args:
            chunks: Chunks to enrich.
            max_batch: Maximum batch size.
            plog: Pipeline logger.

        Returns:
            List of enriched chunks.
        """
        _Logger.get().info(
            f"Large document: {len(chunks)} chunks, processing in batches of {max_batch}"
        )
        enriched_all: List[Any] = []
        total_batches = (len(chunks) + max_batch - 1) // max_batch

        for batch_idx in range(0, len(chunks), max_batch):
            batch = chunks[batch_idx : batch_idx + max_batch]
            batch_num = batch_idx // max_batch + 1

            enriched_batch = self.enricher.enrich_batch(batch)
            enriched_all.extend(enriched_batch)

            progress = batch_num / total_batches
            self._report_progress(
                "enrich",
                progress,
                f"Enriched batch {batch_num}/{total_batches} ({len(enriched_all)}/{len(chunks)} chunks)",
            )
            plog.log_progress(f"Enriched batch {batch_num}/{total_batches}")

        return enriched_all

    def _enrich_via_stage(self, chunks: List[Any]) -> List[Any]:
        """
        Enrich chunks via IFStage.execute() interface.

        BUG001: Handles IFEnrichmentStage which uses execute() not enrich_batch().
        Converts legacy ChunkRecord <-> IFChunkArtifact for compatibility.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.

        Args:
            chunks: List[Any] - Legacy ChunkRecord objects to enrich (type hint is Any for migration).

        Returns:
            List[Any] - Enriched legacy ChunkRecord objects for Stage 5 compatibility.
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        enriched: List[Any] = []
        for chunk in chunks:
            # Convert ChunkRecord to IFChunkArtifact
            artifact = IFChunkArtifact.from_chunk_record(chunk)

            # Execute enrichment stage
            result = self.enricher.execute(artifact)

            # Convert back to ChunkRecord for backward compatibility
            if isinstance(result, IFChunkArtifact):
                enriched.append(result.to_chunk_record())
            else:
                # Failure artifact or unexpected type - keep original
                _Logger.get().warning(
                    f"Enrichment returned {type(result).__name__}, keeping original chunk"
                )
                enriched.append(chunk)

        return enriched

    def _sync_enrichment_to_artifacts(
        self,
        enriched_chunks: List[Any],
        artifacts: List["IFChunkArtifact"],
    ) -> List["IFChunkArtifact"]:
        """
        Sync enrichment data from legacy ChunkRecords to IFChunkArtifacts.

        f: Creates new artifacts with enrichment data using derive().
        Artifacts are immutable, so we create derived versions with updated metadata.

        Rule #4: Function <60 lines.
        Rule #7: Explicit return type.

        Args:
            enriched_chunks: List[Any] - Legacy ChunkRecord objects with enrichment data
                            (embeddings, entities, concepts, quality_score).
                            Type hint is Any for migration compatibility.
            artifacts: Original IFChunkArtifacts from Stage 3 before enrichment.

        Returns:
            List of enriched IFChunkArtifacts with enrichment data in metadata.
        """
        if not enriched_chunks or not artifacts:
            return artifacts

        # Build lookup by chunk_id for O(1) matching
        chunk_map: Dict[str, Any] = {}
        for chunk in enriched_chunks:
            chunk_id = getattr(chunk, "chunk_id", None)
            if chunk_id:
                chunk_map[chunk_id] = chunk

        enriched_artifacts: List["IFChunkArtifact"] = []

        for artifact in artifacts:
            chunk = chunk_map.get(artifact.artifact_id)
            if chunk is None:
                # No matching chunk, keep original artifact
                enriched_artifacts.append(artifact)
                continue

            # Build updated metadata with enrichment data
            updated_metadata = dict(artifact.metadata)

            # Sync enrichment fields from legacy ChunkRecord to artifact metadata
            if hasattr(chunk, "embedding") and chunk.embedding:
                updated_metadata["embedding"] = chunk.embedding
            if hasattr(chunk, "entities") and chunk.entities:
                updated_metadata["entities"] = chunk.entities
            if hasattr(chunk, "concepts") and chunk.concepts:
                updated_metadata["concepts"] = chunk.concepts
            if hasattr(chunk, "quality_score") and chunk.quality_score:
                updated_metadata["quality_score"] = chunk.quality_score

            # Create derived artifact with enrichment provenance
            enriched = artifact.derive(
                "enricher",
                metadata=updated_metadata,
            )
            enriched_artifacts.append(enriched)

        return enriched_artifacts

    def _stage_index_chunks(
        self,
        chunks: List[Any],
        file_path: Path,
        document_id: str,
        doc_state: DocumentState,
        plog: PipelineLogger,
    ) -> int:
        """
        Stage 5: Index chunks into storage.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        plog.start_stage("index")
        self._report_progress("index", 0.0, "Indexing chunks")
        doc_state.status = ProcessingStatus.INDEXING

        indexed_count = self.storage.add_chunks(chunks)
        doc_state.indexed_chunks = indexed_count

        self._report_progress("index", 1.0, f"Indexed {indexed_count} chunks")

        # Move file if configured
        if self.config.ingest.move_completed:
            self._move_to_completed(file_path, document_id)

        return indexed_count
