"""Shared objects handed to the API routes."""
from app.knowledge_graph.extraction_config import get_kg_settings
from app.knowledge_graph.extraction_pipeline import build_graph_extraction_service
from app.knowledge_graph.graph_builder import GraphBuilder
from app.knowledge_graph.graph_indexer import GraphIndexer
from app.knowledge_graph.graph_retriever import GraphRetriever
from app.knowledge_graph.neo4j_client import Neo4jClient
from functools import lru_cache
from app.guardrails.input_guardrail import InputGuardrail
from app.guardrails.output_guardrail import OutputGuardrail
from app.llm.client import OpenAIChatClient
from app.llm.grounded_generation import GroundedQA
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker

from app.core.config import get_settings
from app.core.indexes import BM25Index
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.store import DocumentStore
from app.vectorstore.base import VectorStore
from app.vectorstore.embeddings import get_embedding_model
from app.vectorstore.faiss_store import FaissVectorStore


@lru_cache
def get_document_store() -> DocumentStore:
    settings = get_settings()
    return DocumentStore(
        upload_dir=settings.resolve_path(settings.upload_dir),
        registry_path=settings.resolve_path("data/processed") / "registry.json",
    )


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return FaissVectorStore(
        index_dir=settings.resolve_path(settings.faiss_index_path),
        embedding_model=get_embedding_model(),
    )


@lru_cache
def get_bm25_index() -> BM25Index:
    return BM25Index(get_document_store())


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    settings = get_settings()
    return IngestionPipeline(
        store=get_document_store(),
        vector_store=get_vector_store(),
        bm25_index=get_bm25_index(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        graph_store=get_graph_builder(),
        skip_garbled_chunks=settings.skip_garbled_chunks,
    )


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(
        vector_retriever=get_vector_store() if settings.enable_vector_retrieval else None,
        bm25_retriever=(
            get_bm25_index().retriever if settings.enable_bm25 else None
        ),
        graph_retriever=get_graph_retriever() if settings.enable_kg_retrieval else None,
        vector_top_k=settings.vector_top_k,
        bm25_top_k=settings.bm25_top_k,
        kg_top_k=settings.kg_top_k,
        fusion_method=settings.fusion_method,
        reranker=get_reranker() if settings.enable_reranker else None,
        rerank_candidates=settings.rerank_candidates,
        rerank_top_k=settings.rerank_top_k,
    )


@lru_cache
def get_context_builder() -> ContextBuilder:
    settings = get_settings()
    return ContextBuilder(
        max_chars=settings.context_max_chars, max_sources=settings.context_max_sources
    )


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    settings = get_settings()
    return CrossEncoderReranker(
        model_name=settings.reranker_model, min_score=settings.min_relevance_score
    )
@lru_cache
def get_neo4j_client() -> Neo4jClient:
    settings = get_settings()
    return Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )


@lru_cache
def get_graph_builder() -> GraphBuilder | None:
    """Writes to Neo4j; None when Neo4j is not configured."""
    client = get_neo4j_client()
    return GraphBuilder(client) if client.is_configured else None


@lru_cache
def get_graph_indexer() -> GraphIndexer | None:
    """Builds a document's graph after upload; None when the feature is off."""
    settings = get_settings()
    kg_settings = get_kg_settings()
    builder = get_graph_builder()
    has_key = bool(
        kg_settings.openai_api_key and kg_settings.openai_api_key.get_secret_value()
    )
    if builder is None or not has_key:
        return None
    if not (settings.enable_kg_retrieval and kg_settings.kg_build_on_upload):
        return None
    return GraphIndexer(lambda: build_graph_extraction_service(kg_settings), builder)


@lru_cache
def get_graph_retriever() -> GraphRetriever | None:
    """Finds evidence through the knowledge graph; None when Neo4j is not configured."""
    client = get_neo4j_client()
    if not client.is_configured:
        return None
    kg_settings = get_kg_settings()
    return GraphRetriever(
        client,
        max_seed_entities=kg_settings.kg_max_seed_entities,
        min_entity_match=kg_settings.kg_min_entity_match,
        max_facts=kg_settings.kg_max_facts,
    )


@lru_cache
def get_chat_client() -> OpenAIChatClient | None:
    """The answering LLM; None when no API key is configured."""
    settings = get_settings()
    key = settings.openai_api_key.get_secret_value()
    return OpenAIChatClient(api_key=key, model=settings.llm_model) if key else None


def get_grounded_qa() -> GroundedQA:
    settings = get_settings()
    return GroundedQA(
        retriever=get_hybrid_retriever(),
        context_builder=get_context_builder(),
        llm=get_chat_client(),
        graph_retriever=get_graph_retriever() if settings.enable_kg_retrieval else None,
        kg_top_k=settings.kg_top_k,
        output_guardrail=get_output_guardrail(),
        max_regenerations=settings.max_regenerations,
    )


def get_output_guardrail() -> OutputGuardrail | None:
    settings = get_settings()
    if not settings.enable_output_guardrail:
        return None
    verifier = get_chat_client() if settings.enable_llm_verification else None
    return OutputGuardrail(verifier=verifier)


@lru_cache
def get_input_guardrail() -> InputGuardrail:
    settings = get_settings()
    return InputGuardrail(
        max_chars=settings.max_query_chars,
        block_out_of_scope=settings.block_out_of_scope_requests,
    )
