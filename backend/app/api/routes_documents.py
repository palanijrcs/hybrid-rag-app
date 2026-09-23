"""API endpoints for uploading and managing documents."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.dependencies import get_document_store, get_ingestion_pipeline
from app.ingestion.loaders import DocumentLoadError, UnsupportedFileTypeError
from app.ingestion.pipeline import FileTooLargeError, IngestionPipeline
from app.ingestion.store import DocumentStore, DuplicateDocumentError
from app.models.schemas import (
    ChunkPreview,
    DeleteResponse,
    DocumentDetail,
    DocumentSummary,
    UploadResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> UploadResponse:
    """Upload a document and ingest it into the system."""
    content = await file.read()
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The file is empty.")

    try:
        record = pipeline.ingest(file.filename or "unnamed", content)
    except UnsupportedFileTypeError as error:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(error))
    except DuplicateDocumentError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error))
    except FileTooLargeError as error:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(error))
    except DocumentLoadError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error))
    except Exception as error:  # unexpected
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Ingestion failed: {error}"
        )

    return UploadResponse(
        document=DocumentSummary(**record.__dict__),
        message=f"Ingested {record.chunk_count} chunks from {record.page_count} page(s).",
    )


@router.get("", response_model=list[DocumentSummary])
def list_documents(
    store: DocumentStore = Depends(get_document_store),
) -> list[DocumentSummary]:
    """List every ingested document."""
    return [DocumentSummary(**record.__dict__) for record in store.list_documents()]


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: str,
    store: DocumentStore = Depends(get_document_store),
) -> DocumentDetail:
    """Show one document and its chunks."""
    record = store.get_document(document_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")

    chunks = [
        ChunkPreview(
            chunk_id=c["chunk_id"], page_number=c["page_number"], text=c["text"]
        )
        for c in store.get_chunks(document_id)
    ]
    return DocumentDetail(**record.__dict__, chunks=chunks)


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(
    document_id: str,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> DeleteResponse:
    """Delete a document, its chunks and its vectors."""
    if not pipeline.remove(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    return DeleteResponse(document_id=document_id, deleted=True)