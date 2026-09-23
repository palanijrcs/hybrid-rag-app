"""FAISS-backed vector store with persistence and metadata mapping."""

import json
import logging
import threading
from pathlib import Path

import faiss
import numpy as np

from app.vectorstore.base import VectorHit, VectorStore
from app.vectorstore.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"


class FaissVectorStore(VectorStore):
    """Stores embeddings in FAISS and their metadata alongside, on disk."""

    def __init__(self, index_dir: Path, embedding_model: EmbeddingModel) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self._lock = threading.Lock()

        # IndexIDMap lets us attach our own ids to vectors, so we can delete them.
        # IndexFlatIP = exact inner-product search; with normalised vectors this
        # equals cosine similarity.
        self._index = faiss.IndexIDMap2(
            faiss.IndexFlatIP(embedding_model.dimension)
        )
        # vector id (int) -> chunk metadata
        self._metadata: dict[int, dict] = {}
        self._next_id = 0

        self._load()

    # ---------- persistence ----------
    @property
    def _index_path(self) -> Path:
        return self.index_dir / INDEX_FILE

    @property
    def _metadata_path(self) -> Path:
        return self.index_dir / METADATA_FILE

    def _load(self) -> None:
        if not (self._index_path.exists() and self._metadata_path.exists()):
            logger.info("No existing FAISS index found; starting empty.")
            return
        try:
            self._index = faiss.read_index(str(self._index_path))
            raw = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            self._metadata = {int(k): v for k, v in raw["metadata"].items()}
            self._next_id = raw["next_id"]
            logger.info("Loaded FAISS index with %d vectors.", self._index.ntotal)
        except Exception:
            logger.exception("Could not load FAISS index; starting empty.")
            self._index = faiss.IndexIDMap2(
                faiss.IndexFlatIP(self.embedding_model.dimension)
            )
            self._metadata = {}
            self._next_id = 0

    def _save(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        self._metadata_path.write_text(
            json.dumps(
                {
                    "next_id": self._next_id,
                    "embedding_model": self.embedding_model.model_name,
                    "metadata": {str(k): v for k, v in self._metadata.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # ---------- operations ----------
    def add_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0

        vectors = self.embedding_model.embed_texts([c["text"] for c in chunks])

        with self._lock:
            ids = np.arange(
                self._next_id, self._next_id + len(chunks), dtype="int64"
            )
            self._index.add_with_ids(vectors, ids)
            for vector_id, chunk in zip(ids, chunks):
                self._metadata[int(vector_id)] = {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "document_name": chunk["document_name"],
                    "page_number": chunk.get("page_number"),
                    "text": chunk["text"],
                }
            self._next_id += len(chunks)
            self._save()

        logger.info("Indexed %d chunks (total %d).", len(chunks), self._index.ntotal)
        return len(chunks)

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        if self._index.ntotal == 0 or not query.strip():
            return []

        vector = self.embedding_model.embed_query(query).reshape(1, -1)
        scores, ids = self._index.search(vector, min(top_k, self._index.ntotal))

        hits: list[VectorHit] = []
        for score, vector_id in zip(scores[0], ids[0]):
            if vector_id == -1:  # FAISS pads with -1 when fewer results exist
                continue
            meta = self._metadata.get(int(vector_id))
            if meta is None:
                continue
            hits.append(
                VectorHit(
                    chunk_id=meta["chunk_id"],
                    document_id=meta["document_id"],
                    document_name=meta["document_name"],
                    text=meta["text"],
                    page_number=meta["page_number"],
                    score=float(score),
                )
            )
        return hits

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            ids = [
                vector_id
                for vector_id, meta in self._metadata.items()
                if meta["document_id"] == document_id
            ]
            if not ids:
                return 0

            self._index.remove_ids(np.array(ids, dtype="int64"))
            for vector_id in ids:
                del self._metadata[vector_id]
            self._save()

        logger.info("Removed %d vectors for document %s.", len(ids), document_id)
        return len(ids)

    def count(self) -> int:
        return int(self._index.ntotal)