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