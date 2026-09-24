"""Request and response shapes for the API."""

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_type: str
    uploaded_at: str
    chunk_count: int
    page_count: int


class ChunkPreview(BaseModel):
    chunk_id: str
    page_number: int | None = None
    text: str


class DocumentDetail(DocumentSummary):
    chunks: list[ChunkPreview] = Field(default_factory=list)


class UploadResponse(BaseModel):
    document: DocumentSummary
    message: str


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool


class ErrorResponse(BaseModel):
    detail: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    text: str
    score: float
    retriever: str


class SearchResponse(BaseModel):
    query: str
    method: str
    results: list[RetrievedChunk] = Field(default_factory=list)


class ComparisonResponse(BaseModel):
    query: str
    vector: list[RetrievedChunk] = Field(default_factory=list)
    bm25: list[RetrievedChunk] = Field(default_factory=list)


class IndexStats(BaseModel):
    documents: int
    chunks: int
    vectors: int
    bm25_chunks: int


class FusedResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    text: str
    score: float
    retrievers: list[str]
    original_scores: dict[str, float]
    ranks: dict[str, int]


class HybridSearchResponse(BaseModel):
    query: str
    fusion_method: str
    retrievers_used: list[str]
    counts: dict[str, int]
    results: list[FusedResult] = Field(default_factory=list)