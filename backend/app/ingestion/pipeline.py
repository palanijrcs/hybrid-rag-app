"""The document ingestion pipeline: validate, hash, load, chunk, register."""

import logging
from pathlib import Path

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

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB


class FileTooLargeError(Exception):
    """Raised when an uploaded file exceeds the size limit."""


class IngestionPipeline:
    """Turns an uploaded file into stored, chunked, citable content."""

    def __init__(
        self, store: DocumentStore, chunk_size: int, chunk_overlap: int
    ) -> None:
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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

        # 4. Register
        record = self.store.register(
            document_id=document_id,
            filename=filename,
            file_hash=file_hash,
            chunks=chunks,
            page_count=len(pages),
            stored_path=stored_path,
        )
        logger.info(
            "Ingested %s: %d pages, %d chunks", filename, len(pages), len(chunks)
        )
        return record