"""Phase 9: write extracted entities/relationships into Neo4j AuraDB.

Graph model (every fact stays traceable to its source chunk):

    (:Document {document_id, document_name, kg_status, ...counts})
    (:Chunk {chunk_id, document_id, document_name, page_number, section, text})
    (:Entity {entity_id, name, type, description})

    (Chunk)-[:PART_OF]->(Document)
    (Chunk)-[:MENTIONS]->(Entity)
    (Entity)-[:RELATED_TO {relationship_id, type, confidence,
                           chunk_ids[], evidence[], confidences[], document_ids[]}]->(Entity)

Relationship semantics (WORKS_FOR, PRODUCES, ...) live in the `type` property,
so Cypher never has to build relationship types from strings (no injection risk).
chunk_ids / evidence / confidences are parallel lists: index i = one supporting chunk.

Writes are idempotent: re-writing a document first removes its old facts,
then adds the new ones, all in one transaction.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

from .models import DocumentGraph
from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    # Used by graph retrieval (Phase 10) to find entities named in a question
    "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS "
    "FOR (e:Entity) ON EACH [e.name, e.description]",
]

# ---------------------------------------------------------------- Cypher
Q_SET_STATUS = """
MERGE (d:Document {document_id: $document_id})
SET d.document_name = $document_name, d.kg_status = $status,
    d.kg_error = $error, d.updated_at = $now
"""

Q_FIND_DOCUMENT_FACTS = """
OPTIONAL MATCH (c:Chunk {document_id: $document_id})
OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
RETURN collect(DISTINCT c.chunk_id) AS chunk_ids,
       collect(DISTINCT e.entity_id) AS entity_ids
"""

Q_PRUNE_RELATIONSHIPS = """
MATCH ()-[r:RELATED_TO]->()
WHERE $document_id IN coalesce(r.document_ids, [])
WITH r, [i IN range(0, size(r.chunk_ids) - 1) WHERE NOT r.chunk_ids[i] IN $chunk_ids] AS keep
SET r.evidence = [i IN keep | r.evidence[i]],
    r.confidences = [i IN keep | r.confidences[i]],
    r.chunk_ids = [i IN keep | r.chunk_ids[i]],
    r.document_ids = [d IN r.document_ids WHERE d <> $document_id]
WITH r
SET r.confidence = reduce(m = 0.0, x IN r.confidences | CASE WHEN x > m THEN x ELSE m END)
WITH r WHERE size(r.chunk_ids) = 0
DELETE r
"""

Q_DELETE_CHUNKS = "MATCH (c:Chunk {document_id: $document_id}) DETACH DELETE c"
Q_DELETE_DOCUMENT = "MATCH (d:Document {document_id: $document_id}) DETACH DELETE d"

Q_DELETE_ORPHAN_ENTITIES = """
MATCH (e:Entity) WHERE e.entity_id IN $entity_ids AND NOT EXISTS { (e)--() }
DELETE e
"""

Q_UPSERT_DOCUMENT = """
MERGE (d:Document {document_id: $document_id})
SET d.document_name = $document_name, d.kg_status = $status, d.kg_error = null,
    d.updated_at = $now, d += $stats
"""

Q_WRITE_CHUNKS = """
MATCH (d:Document {document_id: $document_id})
UNWIND $rows AS row
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.document_id = row.document_id, c.document_name = row.document_name,
    c.page_number = row.page_number, c.section = row.section, c.text = row.text
MERGE (c)-[:PART_OF]->(d)
"""

# Entities created before Phase 9 have no entity_id; adopt them instead of clashing
# with the existing (name, type) uniqueness constraint.
Q_ADOPT_LEGACY_ENTITIES = """
UNWIND $rows AS row
MATCH (e:Entity {name: row.name, type: row.type}) WHERE e.entity_id IS NULL
SET e.entity_id = row.entity_id
"""

Q_WRITE_ENTITIES = """
UNWIND $rows AS row
MERGE (e:Entity {entity_id: row.entity_id})
ON CREATE SET e.name = row.name, e.type = row.type, e.description = row.description
SET e.description = CASE
      WHEN size(coalesce(e.description, '')) < size(row.description) THEN row.description
      ELSE e.description END
WITH e, row
UNWIND row.chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})
MERGE (c)-[:MENTIONS]->(e)
"""

Q_WRITE_RELATIONSHIPS = """
UNWIND $rows AS row
MATCH (s:Entity {entity_id: row.source_id})
MATCH (t:Entity {entity_id: row.target_id})
MERGE (s)-[r:RELATED_TO {relationship_id: row.relationship_id}]->(t)
ON CREATE SET r.type = row.type, r.chunk_ids = [], r.document_ids = [],
              r.evidence = [], r.confidences = []
WITH r, row
WHERE NOT row.chunk_id IN r.chunk_ids
SET r.chunk_ids = r.chunk_ids + row.chunk_id,
    r.evidence = r.evidence + row.evidence,
    r.confidences = r.confidences + row.confidence,
    r.document_ids = CASE WHEN row.document_id IN r.document_ids
                          THEN r.document_ids ELSE r.document_ids + row.document_id END
WITH r
SET r.confidence = reduce(m = 0.0, x IN r.confidences | CASE WHEN x > m THEN x ELSE m END)
"""

Q_GET_STATUS = """
MATCH (d:Document {document_id: $document_id})
OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d)
OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
RETURN d {.*} AS document, count(DISTINCT c) AS chunks_in_graph,
       count(DISTINCT e) AS entities_in_graph
"""


# ---------------------------------------------------------------- row builders
def _get(chunk: Any, field: str, default: Any = None) -> Any:
    return chunk.get(field, default) if isinstance(chunk, dict) else getattr(chunk, field, default)


def build_chunk_rows(chunks: Iterable[Any]) -> list[dict]:
    return [
        {
            "chunk_id": str(_get(c, "chunk_id")),
            "document_id": str(_get(c, "document_id")),
            "document_name": _get(c, "document_name", ""),
            "page_number": _get(c, "page_number"),
            "section": _get(c, "section"),
            "text": _get(c, "text", ""),
        }
        for c in chunks
    ]


def build_entity_rows(graph: DocumentGraph) -> list[dict]:
    return [
        {
            "entity_id": e.entity_id,
            "name": e.name,
            "type": e.type,
            "description": e.description or "",
            "chunk_ids": sorted({s.chunk_id for s in e.sources}),
        }
        for e in graph.entities
    ]


def build_relationship_rows(graph: DocumentGraph) -> list[dict]:
    """One row per (relationship, supporting chunk), so evidence stays chunk-aligned."""
    rows: dict[tuple[str, str], dict] = {}
    for chunk_result in graph.chunk_results:
        for r in chunk_result.relationships:
            src = r.sources[0]
            key = (r.relationship_id, src.chunk_id)
            if key in rows:
                continue
            rows[key] = {
                "relationship_id": r.relationship_id,
                "source_id": r.source_id,
                "target_id": r.target_id,
                "type": r.type,
                "chunk_id": src.chunk_id,
                "document_id": src.document_id,
                "evidence": r.evidence[0] if r.evidence else "",
                "confidence": float(r.confidence),
            }
    return list(rows.values())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- builder
class GraphBuilder:
    """Writes and removes one document's knowledge graph."""

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    def ensure_schema(self) -> None:
        for statement in SCHEMA_STATEMENTS:
            self.client.run(statement)
        logger.info("kg.schema.ready")

    def set_status(
        self, document_id: str, document_name: str, status: str, error: str | None = None
    ) -> None:
        self.client.run(
            Q_SET_STATUS,
            document_id=document_id, document_name=document_name,
            status=status, error=error, now=_now(),
        )

    def get_status(self, document_id: str) -> dict | None:
        rows = self.client.run(Q_GET_STATUS, document_id=document_id)
        if not rows or rows[0].get("document") is None:
            return None
        row = rows[0]
        return {
            **row["document"],
            "chunks_in_graph": row["chunks_in_graph"],
            "entities_in_graph": row["entities_in_graph"],
        }

    def write_document(
        self, graph: DocumentGraph, chunks: list[Any], status: str = "done"
    ) -> dict[str, int]:
        """Replace this document's facts in Neo4j with `graph`, atomically."""
        chunk_rows = build_chunk_rows(chunks)
        entity_rows = build_entity_rows(graph)
        rel_rows = build_relationship_rows(graph)
        document_name = chunk_rows[0]["document_name"] if chunk_rows else ""
        stats = {
            "kg_chunks": len(chunk_rows),
            "kg_entities": len(entity_rows),
            "kg_relationships": len(rel_rows),
            "kg_failed_chunks": graph.stats["failed_chunks"],
            "kg_rejected": graph.stats["rejected"],
        }

        def work(tx: Any) -> None:
            _delete_in_tx(tx, graph.document_id)
            tx.run(Q_UPSERT_DOCUMENT, document_id=graph.document_id,
                   document_name=document_name, status=status, now=_now(), stats=stats).consume()
            if chunk_rows:
                tx.run(Q_WRITE_CHUNKS, document_id=graph.document_id, rows=chunk_rows).consume()
            if entity_rows:
                tx.run(Q_ADOPT_LEGACY_ENTITIES, rows=entity_rows).consume()
                tx.run(Q_WRITE_ENTITIES, rows=entity_rows).consume()
            if rel_rows:
                tx.run(Q_WRITE_RELATIONSHIPS, rows=rel_rows).consume()

        with self.client.session() as session:
            session.execute_write(work)
        logger.info("kg.write.done document_id=%s stats=%s", graph.document_id, stats)
        return stats

    def delete_document(self, document_id: str) -> None:
        """Remove a document's chunks, its evidence on shared relationships,
        and any entities no other document still mentions."""
        with self.client.session() as session:
            session.execute_write(lambda tx: _delete_in_tx(tx, document_id))
        logger.info("kg.delete.done document_id=%s", document_id)


def _delete_in_tx(tx: Any, document_id: str) -> None:
    record = tx.run(Q_FIND_DOCUMENT_FACTS, document_id=document_id).single()
    chunk_ids = list(record["chunk_ids"]) if record else []
    entity_ids = list(record["entity_ids"]) if record else []
    tx.run(Q_PRUNE_RELATIONSHIPS, document_id=document_id, chunk_ids=chunk_ids).consume()
    tx.run(Q_DELETE_CHUNKS, document_id=document_id).consume()
    tx.run(Q_DELETE_DOCUMENT, document_id=document_id).consume()
    if entity_ids:
        tx.run(Q_DELETE_ORPHAN_ENTITIES, entity_ids=entity_ids).consume()
