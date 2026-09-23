"""The interface every vector store must provide."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorHit:
    """One search result, with everything needed to cite it."""

    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_number: int | None
    score: float
    retriever: str = "vector"


class VectorStore(ABC):
    """Stores chunk vectors and searches them by meaning."""

    @abstractmethod
    def add_chunks(self, chunks: list[dict]) -> int:
        """Embed and store chunks. Returns how many were added."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[VectorHit]:
        """Find the chunks closest in meaning to the query."""

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Remove every vector belonging to a document. Returns how many."""

    @abstractmethod
    def count(self) -> int:
        """How many vectors are stored."""