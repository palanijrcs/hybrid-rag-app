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
    rerank_score: float | None = None


class HybridSearchResponse(BaseModel):
    query: str
    fusion_method: str
    retrievers_used: list[str]
    counts: dict[str, int]
    results: list[FusedResult] = Field(default_factory=list)


class ContextSourceResult(BaseModel):
    ref: int
    label: str
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    retrievers: list[str]
    rerank_score: float | None = None
    text: str


class ContextFactResult(BaseModel):
    source_ref: int
    text: str
    evidence: str


class ContextResponse(BaseModel):
    query: str
    is_empty: bool
    char_count: int
    dropped_chunks: int
    removed_duplicate_sentences: int
    sources: list[ContextSourceResult] = Field(default_factory=list)
    facts: list[ContextFactResult] = Field(default_factory=list)
    context: str


class GraphEntityMatch(BaseModel):
    entity_id: str
    name: str
    type: str
    match: float


class GraphFactResult(BaseModel):
    source: str
    type: str
    target: str
    evidence: str
    confidence: float


class GraphEvidenceResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    text: str
    score: float
    entities: list[str] = Field(default_factory=list)
    facts: list[GraphFactResult] = Field(default_factory=list)


class GraphSearchResponse(BaseModel):
    query: str
    matched_entities: list[GraphEntityMatch] = Field(default_factory=list)
    evidence: list[GraphEvidenceResult] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceCitation(BaseModel):
    ref: int
    label: str
    document_id: str
    document_name: str
    page_number: int | None = None
    chunk_id: str
    retrievers: list[str]
    rerank_score: float | None = None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    insufficient_evidence: bool
    sources: list[SourceCitation] = Field(default_factory=list)
    retrieval: dict[str, int] = Field(default_factory=dict)
    model: str | None = None
    timings_ms: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    coverage: str | None = None
    question_subject: str | None = None
    sources_subject: str | None = None
