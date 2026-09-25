"""Phase 9: extraction (Phase 8) + graph writing, for one document.

Runs outside the request (FastAPI BackgroundTasks or scripts/build_graph.py),
because LLM extraction takes seconds per chunk. Progress and failures are
recorded on the (:Document) node, so nothing is kept in process memory.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from .extraction_pipeline import GraphExtractionService
from .graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class GraphIndexer:
    def __init__(
        self, extraction_factory: Callable[[], GraphExtractionService], builder: GraphBuilder
    ) -> None:
        # A factory, not an instance: each run gets its own event loop (asyncio.run),
        # and an async HTTP client must not be reused across event loops.
        self.extraction_factory = extraction_factory
        self.builder = builder

    def index_document(
        self, document_id: str, document_name: str, chunks: list[Any]
    ) -> dict[str, Any]:
        """Extract and store the graph for one document. Never raises."""
        if not chunks:
            return {"status": "skipped", "reason": "document has no chunks"}
        try:
            self.builder.set_status(document_id, document_name, "processing")
            graph = asyncio.run(self.extraction_factory().extract_document(chunks))

            stats = graph.stats
            if stats["failed_chunks"] == stats["chunks"]:
                first_error = next((c.error for c in graph.chunk_results if c.error), "")
                raise RuntimeError(f"extraction failed for every chunk: {first_error}")

            status = "partial" if stats["failed_chunks"] else "done"
            written = self.builder.write_document(graph, chunks, status=status)
            logger.info("kg.index.%s document_id=%s", status, document_id)
            return {"status": status, **written}
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:500]
            logger.error("kg.index.failed document_id=%s err=%s", document_id, message)
            try:
                self.builder.set_status(document_id, document_name, "failed", message)
            except Exception:
                logger.exception("kg.index.status_update_failed document_id=%s", document_id)
            return {"status": "failed", "error": message}
