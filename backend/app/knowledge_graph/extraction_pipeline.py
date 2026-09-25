"""Orchestrates extraction over a document's chunks and merges the results.

Input chunks may be your Phase-4 Chunk objects or dicts; they need:
chunk_id, document_id, document_name, text, and optionally page_number, section.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterable

from .entity_extraction import LLMGraphExtractor, validate_entities
from .extraction_config import KGExtractionSettings
from .models import ChunkExtraction, DocumentGraph, Entity, Relationship, SourceRef
from .relationship_extraction import validate_relationships

logger = logging.getLogger(__name__)


def _get(chunk: Any, field: str, default: Any = None) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(field, default)
    return getattr(chunk, field, default)


def _chunk_text(chunk: Any) -> str:
    return _get(chunk, "text") or _get(chunk, "content") or _get(chunk, "page_content") or ""


def source_ref_from_chunk(chunk: Any) -> SourceRef:
    return SourceRef(
        document_id=str(_get(chunk, "document_id")),
        document_name=str(_get(chunk, "document_name", "")),
        chunk_id=str(_get(chunk, "chunk_id")),
        page_number=_get(chunk, "page_number"),
        section=_get(chunk, "section"),
    )


class GraphExtractionService:
    def __init__(self, extractor: LLMGraphExtractor, settings: KGExtractionSettings) -> None:
        self.extractor = extractor
        self.settings = settings

    async def extract_chunk(self, chunk: Any) -> ChunkExtraction:
        source = source_ref_from_chunk(chunk)
        text = _chunk_text(chunk)
        if not text.strip():
            return ChunkExtraction(chunk_id=source.chunk_id)
        try:
            raw = await self.extractor.extract_raw(text, source.document_name, source.section)
        except Exception as exc:  # one bad chunk must not fail the whole document
            logger.error("kg.extract.chunk_failed chunk_id=%s err=%s", source.chunk_id, exc)
            return ChunkExtraction(chunk_id=source.chunk_id, error=str(exc)[:500])

        entities, ent_rej = validate_entities(
            raw.entities,
            text,
            self.settings.entity_types,
            source,
            self.settings.kg_max_entities_per_chunk,
        )
        rels, rel_rej = validate_relationships(
            raw.relationships, entities, text, source, self.settings
        )
        return ChunkExtraction(
            chunk_id=source.chunk_id,
            entities=list(entities.values()),
            relationships=rels,
            rejected=ent_rej + rel_rej,
        )

    async def extract_document(self, chunks: Iterable[Any]) -> DocumentGraph:
        chunks = list(chunks)
        if not chunks:
            raise ValueError("no chunks supplied")
        document_id = str(_get(chunks[0], "document_id"))
        sem = asyncio.Semaphore(self.settings.kg_max_concurrency)
        started = time.perf_counter()

        async def run(c: Any) -> ChunkExtraction:
            async with sem:
                return await self.extract_chunk(c)

        results = await asyncio.gather(*(run(c) for c in chunks))
        graph = self._merge(document_id, results)
        logger.info(
            "kg.extract.document_done document_id=%s stats=%s seconds=%.1f",
            document_id, graph.stats, time.perf_counter() - started,
        )
        return graph

    @staticmethod
    def _merge(document_id: str, results: list[ChunkExtraction]) -> DocumentGraph:
        entities: dict[str, Entity] = {}
        rels: dict[str, Relationship] = {}
        for res in results:
            for e in res.entities:
                if e.entity_id not in entities:
                    entities[e.entity_id] = e.model_copy(deep=True)
                    continue
                cur = entities[e.entity_id]
                cur.sources.extend(s for s in e.sources if s not in cur.sources)
                if len(e.description) > len(cur.description):
                    cur.description = e.description
            for r in res.relationships:
                if r.relationship_id not in rels:
                    rels[r.relationship_id] = r.model_copy(deep=True)
                    continue
                cur_r = rels[r.relationship_id]
                cur_r.confidence = max(cur_r.confidence, r.confidence)
                cur_r.evidence.extend(x for x in r.evidence if x not in cur_r.evidence)
                cur_r.sources.extend(s for s in r.sources if s not in cur_r.sources)
        return DocumentGraph(
            document_id=document_id,
            entities=list(entities.values()),
            relationships=list(rels.values()),
            chunk_results=results,
        )


def build_graph_extraction_service(settings: KGExtractionSettings) -> GraphExtractionService:
    """Factory for FastAPI dependencies / scripts."""
    from .entity_extraction import OpenAIJsonClient

    return GraphExtractionService(LLMGraphExtractor(OpenAIJsonClient(settings), settings), settings)
