"""Phase 9 tests: writing the knowledge graph to Neo4j.

Unit tests use a fake Neo4j session. Live tests use your AuraDB (skipped when it
is not configured) and only touch uniquely-named test data, which they delete.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager

import pytest

from app.knowledge_graph.graph_builder import (
    GraphBuilder,
    build_chunk_rows,
    build_entity_rows,
    build_relationship_rows,
)
from app.knowledge_graph.graph_indexer import GraphIndexer
from app.knowledge_graph.models import (
    ChunkExtraction,
    DocumentGraph,
    Entity,
    Relationship,
    SourceRef,
    make_entity_id,
    make_relationship_id,
)


# ---------------------------------------------------------------- helpers
def make_graph(doc: str, suffix: str = "") -> tuple[DocumentGraph, list[dict]]:
    """Doc with 2 chunks: John works for ABC (both chunks), ABC produces Widget (chunk 1)."""
    john, abc, widget = f"John Smith{suffix}", f"ABC Corporation{suffix}", f"Widget{suffix}"
    chunks = [
        {"chunk_id": f"{doc}_chunk_0", "document_id": doc, "document_name": f"{doc}.pdf",
         "text": f"{john} works for {abc}. {abc} produces {widget}.", "page_number": 1},
        {"chunk_id": f"{doc}_chunk_1", "document_id": doc, "document_name": f"{doc}.pdf",
         "text": f"As CEO, {john} works for {abc}.", "page_number": 2},
    ]
    src = [SourceRef(document_id=doc, document_name=f"{doc}.pdf", chunk_id=c["chunk_id"],
                     page_number=c["page_number"]) for c in chunks]
    ids = {n: make_entity_id(t, n) for n, t in
           [(john, "Person"), (abc, "Organization"), (widget, "Product")]}
    ent = {n: Entity(entity_id=ids[n], name=n, type=t, sources=[s])
           for (n, t), s in [((john, "Person"), src[0]), ((abc, "Organization"), src[0]),
                             ((widget, "Product"), src[0])]}

    def rel(a, t, b, s, ev, conf):
        return Relationship(relationship_id=make_relationship_id(ids[a], t, ids[b]),
                            source_id=ids[a], source_name=a, target_id=ids[b], target_name=b,
                            type=t, confidence=conf, evidence=[ev], sources=[s])

    c0 = ChunkExtraction(chunk_id=src[0].chunk_id, entities=list(ent.values()), relationships=[
        rel(john, "WORKS_FOR", abc, src[0], f"{john} works for {abc}", 0.9),
        rel(abc, "PRODUCES", widget, src[0], f"{abc} produces {widget}", 0.8)])
    c1 = ChunkExtraction(chunk_id=src[1].chunk_id,
                         entities=[ent[john].model_copy(update={"sources": [src[1]]}),
                                   ent[abc].model_copy(update={"sources": [src[1]]})],
                         relationships=[rel(john, "WORKS_FOR", abc, src[1],
                                            f"{john} works for {abc}", 1.0)])
    merged_entities = [e.model_copy(deep=True) for e in ent.values()]
    merged_entities[0].sources.append(src[1])
    merged_entities[1].sources.append(src[1])
    graph = DocumentGraph(document_id=doc, entities=merged_entities,
                          relationships=[], chunk_results=[c0, c1])
    return graph, chunks


class FakeResult:
    def __init__(self, record=None):
        self.record = record

    def single(self):
        return self.record

    def consume(self):
        return None


class FakeTx:
    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    def run(self, query, **params):
        self.queries.append((query, params))
        if "collect(DISTINCT c.chunk_id)" in query:
            return FakeResult({"chunk_ids": ["d_chunk_0"], "entity_ids": ["e1"]})
        return FakeResult()


class FakeSession:
    def __init__(self, tx):
        self.tx = tx

    def execute_write(self, fn):
        return fn(self.tx)


class FakeClient:
    def __init__(self):
        self.tx = FakeTx()
        self.runs: list[tuple[str, dict]] = []

    @contextmanager
    def session(self):
        yield FakeSession(self.tx)

    def run(self, query, **params):
        self.runs.append((query, params))
        return []


# ---------------------------------------------------------------- unit tests
def test_row_builders_keep_provenance():
    graph, chunks = make_graph("d")
    chunk_rows = build_chunk_rows(chunks)
    assert [r["page_number"] for r in chunk_rows] == [1, 2]

    entities = {r["name"]: r for r in build_entity_rows(graph)}
    assert entities["John Smith"]["chunk_ids"] == ["d_chunk_0", "d_chunk_1"]
    assert entities["Widget"]["chunk_ids"] == ["d_chunk_0"]

    rels = build_relationship_rows(graph)
    works = [r for r in rels if r["type"] == "WORKS_FOR"]
    assert len(works) == 2  # one row per supporting chunk
    assert {r["chunk_id"] for r in works} == {"d_chunk_0", "d_chunk_1"}
    assert all(r["evidence"] and r["document_id"] == "d" for r in rels)


def test_write_is_one_transaction_that_replaces_old_facts():
    client = FakeClient()
    graph, chunks = make_graph("d")
    stats = GraphBuilder(client).write_document(graph, chunks)
    queries = [q for q, _ in client.tx.queries]
    # delete-first, then document, chunks, entities, relationships
    order = ["collect(DISTINCT c.chunk_id)", "RELATED_TO]->()", "DETACH DELETE c",
             "DETACH DELETE d", "NOT EXISTS", "d += $stats", "MERGE (c:Chunk",
             "entity_id IS NULL", "MERGE (e:Entity", "MERGE (s)-[r:RELATED_TO"]
    positions = [next(i for i, q in enumerate(queries) if marker in q) for marker in order]
    assert positions == sorted(positions)
    assert stats == {"kg_chunks": 2, "kg_entities": 3, "kg_relationships": 3,
                     "kg_failed_chunks": 0, "kg_rejected": 0}


def test_no_query_builds_cypher_from_extracted_text():
    """Injection safety: extracted names/types only ever travel as parameters."""
    client = FakeClient()
    graph, chunks = make_graph("d", suffix="'}) DETACH DELETE n //")
    GraphBuilder(client).write_document(graph, chunks)
    for query, _ in client.tx.queries:
        assert "DETACH DELETE n //" not in query


class FakeExtraction:
    def __init__(self, graph=None, error=None):
        self.graph, self.error = graph, error

    async def extract_document(self, chunks):
        if self.error:
            raise self.error
        return self.graph


class RecordingBuilder:
    def __init__(self):
        self.statuses: list[tuple[str, str | None]] = []
        self.written = None

    def set_status(self, document_id, document_name, status, error=None):
        self.statuses.append((status, error))

    def write_document(self, graph, chunks, status="done"):
        self.written = status
        return {"kg_entities": len(graph.entities)}


def test_indexer_success():
    graph, chunks = make_graph("d")
    builder = RecordingBuilder()
    result = GraphIndexer(lambda: FakeExtraction(graph), builder).index_document("d", "d.pdf", chunks)
    assert result == {"status": "done", "kg_entities": 3}
    assert builder.statuses == [("processing", None)] and builder.written == "done"


def test_indexer_records_failure_instead_of_raising():
    builder = RecordingBuilder()
    indexer = GraphIndexer(lambda: FakeExtraction(error=RuntimeError("LLM down")), builder)
    result = indexer.index_document("d", "d.pdf", [{"chunk_id": "x"}])
    assert result["status"] == "failed" and "LLM down" in result["error"]
    assert builder.statuses[-1][0] == "failed" and builder.written is None


def test_indexer_fails_when_every_chunk_failed():
    graph = DocumentGraph(document_id="d", chunk_results=[
        ChunkExtraction(chunk_id="c0", error="401 invalid api key")])
    builder = RecordingBuilder()
    result = GraphIndexer(lambda: FakeExtraction(graph), builder).index_document("d", "d.pdf", [{}])
    assert result["status"] == "failed" and "401" in result["error"]
    assert builder.written is None  # nothing half-written


def test_indexer_marks_partial():
    graph, chunks = make_graph("d")
    graph.chunk_results.append(ChunkExtraction(chunk_id="d_chunk_2", error="timeout"))
    builder = RecordingBuilder()
    result = GraphIndexer(lambda: FakeExtraction(graph), builder).index_document("d", "d.pdf", chunks)
    assert result["status"] == "partial" and builder.written == "partial"


# ---------------------------------------------------------------- live AuraDB tests
@pytest.fixture
def live_builder():
    from app.core.config import get_settings
    from app.knowledge_graph.neo4j_client import Neo4jClient

    settings = get_settings()
    if not settings.neo4j_uri:
        pytest.skip("Neo4j is not configured.")
    client = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
    if not client.verify():
        pytest.skip("Neo4j is unreachable.")
    client.ensure_schema()
    builder = GraphBuilder(client)
    builder.ensure_schema()
    yield builder
    client.close()


def _counts(builder: GraphBuilder, suffix: str) -> dict:
    rows = builder.client.run(
        """
        OPTIONAL MATCH (e:Entity) WHERE e.name ENDS WITH $s
        WITH count(e) AS entities
        OPTIONAL MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) WHERE a.name ENDS WITH $s
        RETURN entities, count(r) AS rels, collect(r {.type, .chunk_ids, .evidence,
               .document_ids, .confidence}) AS details
        """,
        s=suffix,
    )
    return rows[0]


def test_live_write_rewrite_share_and_delete(live_builder):
    suffix = f" T{uuid.uuid4().hex[:8]}"
    doc_a, doc_b = f"testA{uuid.uuid4().hex[:8]}", f"testB{uuid.uuid4().hex[:8]}"
    graph_a, chunks_a = make_graph(doc_a, suffix)
    graph_b, chunks_b = make_graph(doc_b, suffix)
    try:
        live_builder.write_document(graph_a, chunks_a)
        c = _counts(live_builder, suffix)
        assert c["entities"] == 3 and c["rels"] == 2
        works = next(d for d in c["details"] if d["type"] == "WORKS_FOR")
        assert len(works["chunk_ids"]) == len(works["evidence"]) == 2
        assert works["confidence"] == 1.0

        status = live_builder.get_status(doc_a)
        assert status["kg_status"] == "done" and status["chunks_in_graph"] == 2
        assert status["entities_in_graph"] == 3

        # Rewriting the same document is idempotent
        live_builder.write_document(graph_a, chunks_a)
        c = _counts(live_builder, suffix)
        assert c["entities"] == 3 and c["rels"] == 2
        works = next(d for d in c["details"] if d["type"] == "WORKS_FOR")
        assert len(works["chunk_ids"]) == 2

        # A second document mentioning the same entities shares their nodes
        live_builder.write_document(graph_b, chunks_b)
        c = _counts(live_builder, suffix)
        assert c["entities"] == 3 and c["rels"] == 2
        works = next(d for d in c["details"] if d["type"] == "WORKS_FOR")
        assert sorted(works["document_ids"]) == sorted([doc_a, doc_b])
        assert len(works["evidence"]) == 4

        # Deleting A keeps B's evidence and the shared entities
        live_builder.delete_document(doc_a)
        c = _counts(live_builder, suffix)
        assert c["entities"] == 3 and c["rels"] == 2
        works = next(d for d in c["details"] if d["type"] == "WORKS_FOR")
        assert works["document_ids"] == [doc_b]
        assert all(cid.startswith(doc_b) for cid in works["chunk_ids"])
        assert live_builder.get_status(doc_a) is None

        # Deleting B removes everything
        live_builder.delete_document(doc_b)
        c = _counts(live_builder, suffix)
        assert c["entities"] == 0 and c["rels"] == 0
    finally:
        live_builder.delete_document(doc_a)
        live_builder.delete_document(doc_b)


# ---------------------------------------------------------------- API wiring
class RecordingIndexer:
    def __init__(self):
        self.calls: list[tuple[str, str, int]] = []

    def index_document(self, document_id, document_name, chunks):
        self.calls.append((document_id, document_name, len(chunks)))
        return {"status": "done"}


class RecordingGraphStore:
    def __init__(self):
        self.deleted: list[str] = []
        self.status: dict | None = None

    def delete_document(self, document_id):
        self.deleted.append(document_id)

    def get_status(self, document_id):
        return self.status


@pytest.fixture
def graph_api(tmp_path):
    """API client with a real document store but no embeddings, LLM or Neo4j."""
    from fastapi.testclient import TestClient

    from app.core.dependencies import (
        get_document_store,
        get_graph_builder,
        get_graph_indexer,
        get_ingestion_pipeline,
    )
    from app.ingestion.pipeline import IngestionPipeline
    from app.ingestion.store import DocumentStore
    from app.main import app

    store = DocumentStore(upload_dir=tmp_path / "uploads",
                          registry_path=tmp_path / "processed" / "registry.json")
    graph_store = RecordingGraphStore()
    pipeline = IngestionPipeline(store=store, vector_store=None, bm25_index=None,
                                 chunk_size=400, chunk_overlap=60, graph_store=graph_store)
    indexer = RecordingIndexer()
    overrides = {
        get_document_store: lambda: store,
        get_ingestion_pipeline: lambda: pipeline,
        get_graph_indexer: lambda: indexer,
        get_graph_builder: lambda: graph_store,
    }
    app.dependency_overrides.update(overrides)
    try:
        yield TestClient(app), indexer, graph_store
    finally:
        for key in overrides:
            app.dependency_overrides.pop(key, None)


def _upload(client):
    files = {"file": ("company.txt", b"John Smith works for ABC Corporation.", "text/plain")}
    response = client.post("/documents/upload", files=files)
    assert response.status_code == 201, response.text
    return response.json()


def test_upload_queues_graph_build(graph_api):
    client, indexer, _ = graph_api
    body = _upload(client)
    assert "building in the background" in body["message"]
    doc_id = body["document"]["document_id"]
    assert indexer.calls == [(doc_id, "company.txt", 1)]  # TestClient runs background tasks


def test_graph_status_endpoint(graph_api):
    client, _, graph_store = graph_api
    doc_id = _upload(client)["document"]["document_id"]
    assert client.get(f"/documents/{doc_id}/graph").json()["kg_status"] == "not_built"
    graph_store.status = {"document_id": doc_id, "kg_status": "done", "kg_entities": 2}
    assert client.get(f"/documents/{doc_id}/graph").json()["kg_entities"] == 2
    assert client.get("/documents/missing/graph").status_code == 404


def test_rebuild_endpoint(graph_api):
    client, indexer, _ = graph_api
    doc_id = _upload(client)["document"]["document_id"]
    response = client.post(f"/documents/{doc_id}/graph")
    assert response.status_code == 202 and len(indexer.calls) == 2


def test_delete_removes_graph_facts(graph_api):
    client, _, graph_store = graph_api
    doc_id = _upload(client)["document"]["document_id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert graph_store.deleted == [doc_id]
