"""
Management Operations Mixin for ChromaDB.

Handles statistics, library management, and collection operations.
"""

from typing import Any, Dict, List

from ingestforge.chunking.semantic_chunker import ChunkRecord


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


class ChromaDBManagementMixin:
    """
    Mixin providing management operations for ChromaDB repository.

    Extracted from chromadb.py to reduce file size (Phase 4, Rule #4).
    """

    def count(self) -> int:
        """Get total chunk count."""
        try:
            return self.collection.count()
        except Exception as e:
            _Logger.get().error(f"Failed to count chunks: {e}")
            return 0

    def get_all_chunks(self, limit: int = 10000) -> List[ChunkRecord]:
        """Get all chunks from the repository.

        Args:
            limit: Maximum number of chunks to return

        Returns:
            List of all chunks
        """
        try:
            # ChromaDB get() without filters returns all documents
            result = self.collection.get(
                limit=limit,
                include=["documents", "metadatas"],
            )

            chunks = []
            if result["ids"]:
                for i, chunk_id in enumerate(result["ids"]):
                    chunk = self._metadata_to_chunk(
                        chunk_id,
                        result["documents"][i],
                        result["metadatas"][i],
                    )
                    chunks.append(chunk)

            return chunks
        except Exception as e:
            _Logger.get().error(f"Failed to get all chunks: {e}")
            return []

    def clear(self) -> None:
        """Clear all data."""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self._collection = None
        except Exception as e:
            _Logger.get().error(f"Failed to clear collection: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_chunks": self.count(),
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
        }

    def get_libraries(self) -> List[str]:
        """
        Get list of unique library names in storage.

        Returns:
            List of library names, always including "default".
        """
        try:
            # Get all chunks and extract unique libraries
            result = self.collection.get(
                include=["metadatas"],
            )

            libraries = set()
            if result["metadatas"]:
                for metadata in result["metadatas"]:
                    lib = metadata.get("library", "default")
                    libraries.add(lib)

            # Always include "default"
            libraries.add("default")
            return sorted(list(libraries))
        except Exception as e:
            _Logger.get().error(f"Failed to get libraries: {e}")
            return ["default"]

    def count_by_library(self, library_name: str) -> int:
        """
        Count chunks in a specific library.

        Args:
            library_name: Library to count

        Returns:
            Number of chunks in the library
        """
        try:
            result = self.collection.get(
                where={"library": library_name},
                include=[],
            )
            return len(result["ids"]) if result["ids"] else 0
        except Exception as e:
            _Logger.get().error(f"Failed to count library {library_name}: {e}")
            return 0

    def delete_by_library(self, library_name: str) -> int:
        """
        Delete all chunks in a library.

        Args:
            library_name: Library to delete

        Returns:
            Number of chunks deleted
        """
        try:
            # First get all chunk IDs in this library
            result = self.collection.get(
                where={"library": library_name},
                include=[],
            )

            if not result["ids"]:
                return 0

            count = len(result["ids"])

            # Delete all chunks
            self.collection.delete(ids=result["ids"])

            _Logger.get().info(f"Deleted {count} chunks from library '{library_name}'")
            return count
        except Exception as e:
            _Logger.get().error(f"Failed to delete library {library_name}: {e}")
            return 0

    def reassign_library(self, old_library: str, new_library: str) -> int:
        """
        Move all chunks from one library to another.

        Args:
            old_library: Source library name
            new_library: Target library name

        Returns:
            Number of chunks moved
        """
        try:
            # Get all chunks in the old library
            result = self.collection.get(
                where={"library": old_library},
                include=["metadatas"],
            )

            if not result["ids"]:
                return 0

            count = len(result["ids"])

            # Update metadata for each chunk
            new_metadatas = []
            for metadata in result["metadatas"]:
                updated = dict(metadata)
                updated["library"] = new_library
                new_metadatas.append(updated)

            # Update all chunks with new library
            self.collection.update(
                ids=result["ids"],
                metadatas=new_metadatas,
            )

            _Logger.get().info(
                f"Reassigned {count} chunks from '{old_library}' to '{new_library}'"
            )
            return count
        except Exception as e:
            _Logger.get().error(
                f"Failed to reassign library {old_library} to {new_library}: {e}"
            )
            return 0

    def move_document_to_library(self, document_id: str, new_library: str) -> int:
        """
        Move all chunks of a specific document to a different library.

        Args:
            document_id: Document to move
            new_library: Target library name

        Returns:
            Number of chunks updated
        """
        try:
            # Get all chunks for this document
            result = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )

            if not result["ids"]:
                return 0

            count = len(result["ids"])

            # Update metadata for each chunk
            new_metadatas = []
            for metadata in result["metadatas"]:
                updated = dict(metadata)
                updated["library"] = new_library
                new_metadatas.append(updated)

            # Update all chunks with new library
            self.collection.update(
                ids=result["ids"],
                metadatas=new_metadatas,
            )

            _Logger.get().info(
                f"Moved document '{document_id}' ({count} chunks) to library '{new_library}'"
            )
            return count
        except Exception as e:
            _Logger.get().error(
                f"Failed to move document {document_id} to library {new_library}: {e}"
            )
            return 0

    def persist(self) -> None:
        """Persist data to disk.

        Note: PersistentClient auto-persists, so this is a no-op.
        Kept for API compatibility.
        """
        # PersistentClient auto-persists - no manual persist needed
        pass
