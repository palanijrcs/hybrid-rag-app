"""The document ingestion pipeline: validate, hash, load, chunk, index, register."""

import logging
from pathlib import Path
from typing import Protocol

from app.ingestion.chunking import chunk_pages
from app.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    UnsupportedFileTypeError,
    load_document,
)
from app.ingestion.store import (
    DocumentRecord,
    DocumentStore,
    DuplicateDocumentError,
    compute_hash,
    new_document_id,
)
from app.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB


class RefreshableIndex(Protocol):
    """Anything that can rebuild itself after documents change."""

    def refresh(self) -> int: ...


class GraphStore(Protocol):
    """Anything that can forget a document's knowledge-graph facts."""

    def delete_document(self, document_id: str) -> None: ...


class FileTooLargeError(Exception):
    """Raised when an uploaded file exceeds the size limit."""


class IngestionPipeline:
    """Turns an uploaded file into stored, indexed, citable content."""

    def __init__(
        self,
        store: DocumentStore,
        vector_store: VectorStore | None,
        bm25_index: RefreshableIndex | None,
        chunk_size: int,
        chunk_overlap: int,
        graph_store: GraphStore | None = None,
    ) -> None:
        self.store = store
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.graph_store = graph_store

    def ingest(self, filename: str, content: bytes) -> DocumentRecord:
        # 1. Validate
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"'{extension or filename}' is not supported. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if len(content) > MAX_FILE_BYTES:
            raise FileTooLargeError(
                f"{filename} is {len(content) // (1024 * 1024)} MB. "
                f"Limit is {MAX_FILE_BYTES // (1024 * 1024)} MB."
            )

        # 2. Hash and check for duplicates
        file_hash = compute_hash(content)
        existing = self.store.find_by_hash(file_hash)
        if existing:
            raise DuplicateDocumentError(
                f"This file was already uploaded as '{existing.filename}' "
                f"(document_id {existing.document_id})."
            )

        # 3. Save, load, chunk
        document_id = new_document_id()
        stored_path = self.store.save_upload(document_id, filename, content)
        logger.info("Ingesting %s as document_id=%s", filename, document_id)

        try:
            pages = load_document(stored_path)
            chunks = chunk_pages(
                pages=pages,
                document_id=document_id,
                document_name=filename,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        except Exception:
            stored_path.unlink(missing_ok=True)  # don't leave orphan files
            raise

        # 4. Index the chunks for semantic search
        if self.vector_store is not None and chunks:
            try:
                self.vector_store.add_chunks([c.to_dict() for c in chunks])
            except Exception:
                logger.exception("Indexing failed for %s; rolling back.", filename)
                stored_path.unlink(missing_ok=True)
                raise

        # 5. Register
        record = self.store.register(
            document_id=document_id,
            filename=filename,
            file_hash=file_hash,
            chunks=chunks,
            page_count=len(pages),
            stored_path=stored_path,
        )

        # 6. Rebuild the keyword index now that the registry has changed
        if self.bm25_index is not None:
            self.bm25_index.refresh()

        logger.info(
            "Ingested %s: %d pages, %d chunks", filename, len(pages), len(chunks)
        )
        return record

    def remove(self, document_id: str) -> bool:
        """Delete a document everywhere it was stored."""
        if self.graph_store is not None:
            try:
                self.graph_store.delete_document(document_id)
            except Exception:
                # The graph is secondary; don't block deleting the document itself
                logger.exception("Graph cleanup failed for document_id=%s", document_id)
        if self.vector_store is not None:
            self.vector_store.delete_document(document_id)
        removed = self.store.delete_document(document_id)
        if removed and self.bm25_index is not None:
            self.bm25_index.refresh()
        return removed