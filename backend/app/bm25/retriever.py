"""BM25 lexical (keyword) retrieval over document chunks."""

import logging
import re
import threading

from rank_bm25 import BM25Okapi

from app.vectorstore.base import VectorHit

logger = logging.getLogger(__name__)

# Keep letters, digits and internal punctuation like PM-KMY or 4395/2026
_TOKEN = re.compile(r"[A-Za-z0-9\u0900-\u097F]+(?:[-/.][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase searchable words."""
    return [token.lower() for token in _TOKEN.findall(text)]


class BM25Retriever:
    """Keyword search over chunks, rebuilt whenever the corpus changes."""

    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._index: BM25Okapi | None = None
        self._lock = threading.Lock()

    def build(self, chunks: list[dict]) -> int:
        """Replace the whole corpus with these chunks."""
        with self._lock:
            self._chunks = list(chunks)
            corpus = [tokenize(chunk["text"]) for chunk in self._chunks]
            # BM25Okapi cannot be built from an empty corpus
            self._index = BM25Okapi(corpus) if corpus else None
        logger.info("BM25 index built over %d chunks.", len(self._chunks))
        return len(self._chunks)

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        """Find chunks whose words best match the query."""
        tokens = tokenize(query)
        with self._lock:
            index, chunks = self._index, self._chunks

        if index is None or not tokens or not chunks:
            return []

        scores = index.get_scores(tokens)
        token_set = set(tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

        hits: list[VectorHit] = []
        for position, score in ranked[:top_k]:
            chunk = chunks[position]
            # BM25 gives a zero or negative score to terms present in every
            # document, so fall back to checking for a real term overlap.
            if score <= 0 and not (token_set & set(tokenize(chunk["text"]))):
                continue
            hits.append(
                VectorHit(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    document_name=chunk["document_name"],
                    text=chunk["text"],
                    page_number=chunk.get("page_number"),
                    score=float(score),
                    retriever="bm25",
                )
            )
        return hits

    def count(self) -> int:
        return len(self._chunks)