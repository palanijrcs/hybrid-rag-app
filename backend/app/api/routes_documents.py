"""API endpoints for uploading and managing documents."""

import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.core.dependencies import (
    get_document_store,
    get_graph_builder,
    get_graph_indexer,
    get_ingestion_pipeline,
)
from app.ingestion.loaders import DocumentLoadError, UnsupportedFileTypeError
from app.ingestion.pipeline import FileTooLargeError, IngestionPipeline
from app.ingestion.store import DocumentStore, DuplicateDocumentError
from app.knowledge_graph.graph_builder import GraphBuilder
from app.knowledge_graph.graph_indexer import GraphIndexer
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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    indexer: GraphIndexer | None = Depends(get_graph_indexer),
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

    # The knowledge graph needs one LLM call per chunk, so it is built after the
    # response is sent. Check progress with GET /documents/{id}/graph.
    graph_note = "Knowledge graph: disabled."
    if indexer is not None:
        chunks = pipeline.store.get_chunks(record.document_id)
        background_tasks.add_task(
            indexer.index_document, record.document_id, record.filename, chunks
        )
        graph_note = "Knowledge graph: building in the background."

    return UploadResponse(
        document=DocumentSummary(**record.__dict__),
        message=(
            f"Ingested {record.chunk_count} chunks from {record.page_count} page(s). "
            f"{graph_note}"
        ),
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
    """Delete a document, its chunks, its vectors and its graph facts."""
    if not pipeline.remove(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    return DeleteResponse(document_id=document_id, deleted=True)


@router.get("/{document_id}/graph")
def get_document_graph_status(
    document_id: str,
    store: DocumentStore = Depends(get_document_store),
    builder: GraphBuilder | None = Depends(get_graph_builder),
) -> dict:
    """Show whether this document's knowledge graph is built, and its counts."""
    if store.get_document(document_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    if builder is None:
        return {"document_id": document_id, "kg_status": "unavailable"}
    try:
        graph_status = builder.get_status(document_id)
    except Exception:
        logger.exception("Could not read graph status for %s", document_id)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Neo4j is unreachable.")
    return graph_status or {"document_id": document_id, "kg_status": "not_built"}


@router.post("/{document_id}/graph", status_code=202)
def rebuild_document_graph(
    document_id: str,
    background_tasks: BackgroundTasks,
    store: DocumentStore = Depends(get_document_store),
    indexer: GraphIndexer | None = Depends(get_graph_indexer),
) -> dict:
    """(Re)build the knowledge graph for an existing document."""
    record = store.get_document(document_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    if indexer is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Knowledge graph building is disabled (check Neo4j, OPENAI_API_KEY, "
            "ENABLE_KG_RETRIEVAL and KG_BUILD_ON_UPLOAD).",
        )
    chunks = store.get_chunks(document_id)
    background_tasks.add_task(indexer.index_document, document_id, record.filename, chunks)
    return {"document_id": document_id, "kg_status": "queued"}
