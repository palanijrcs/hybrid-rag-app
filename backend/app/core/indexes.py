"""Keeps the in-memory BM25 index in step with the document store."""

import logging

from app.bm25.retriever import BM25Retriever
from app.ingestion.store import DocumentStore

logger = logging.getLogger(__name__)


class BM25Index:
    """A BM25 retriever that rebuilds itself from the document store."""

    def __init__(self, store: DocumentStore) -> None:
        self._store = store
        self._retriever = BM25Retriever()
        self.refresh()

    def refresh(self) -> int:
        """Rebuild the index from every chunk currently stored."""
        chunks = self._store.all_chunks()
        count = self._retriever.build(chunks)
        logger.info("BM25 index refreshed: %d chunks.", count)
        return count

    @property
    def retriever(self) -> BM25Retriever:
        return self._retriever

    def count(self) -> int:
        return self._retriever.count()