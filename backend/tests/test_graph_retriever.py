"""Phase 10 tests: knowledge-graph retrieval.

Unit tests use a fake Neo4j client. The live test writes a tiny uniquely-named
graph to your AuraDB, queries it, and deletes it (skipped if Neo4j is not set up).
"""
from __future__ import annotations

import time
import uuid

import pytest

from app.knowledge_graph.graph_retriever import (
    GraphRetriever,
    build_lucene_query,
    entity_match_score,
    tokenize,
)

JOHN, ABC, WIDGET, LIC = "e_john", "e_abc", "e_widget", "e_lic"
ENTITIES = {
    JOHN: ("John Smith", "Person"),
    ABC: ("ABC Corporation", "Organization"),
    WIDGET: ("Widget", "Product"),
    LIC: ("Life Insurance Corporation of India", "Organization"),
}
FACTS = [
    {"relationship_id": "r1", "type": "WORKS_FOR", "source_id": JOHN, "source": "John Smith",
     "target_id": ABC, "target": "ABC Corporation", "chunk_ids": ["c1", "c2"],
     "evidence": ["John Smith works for ABC Corporation", "As CEO, John Smith works for ABC Corporation"],
     "confidences": [0.9, 1.0]},
    {"relationship_id": "r2", "type": "PRODUCES", "source_id": ABC, "source": "ABC Corporation",
     "target_id": WIDGET, "target": "Widget", "chunk_ids": ["c3"],
     "evidence": ["ABC Corporation produces Widget"], "confidences": [0.8]},
]
MENTIONS = {JOHN: ["c1", "c2", "c4"], ABC: ["c1", "c2", "c3"], WIDGET: ["c3"], LIC: []}
CHUNKS = {cid: {"chunk_id": cid, "document_id": "d1", "document_name": "company.pdf",
                "page_number": n, "text": f"text of {cid}"}
          for n, cid in enumerate(["c1", "c2", "c3", "c4"], start=1)}


class FakeClient:
    """Answers the four retriever queries from the in-memory graph above."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def run(self, query, **p):
        self.calls.append((query, p))
        if "queryNodes" in query:
            words = set(p["lucene"].split(" OR "))
            return [{"entity_id": eid, "name": n, "type": t, "score": 1.0}
                    for eid, (n, t) in ENTITIES.items() if words & set(tokenize(n))]
        if "RELATED_TO" in query:
            seeds = set(p["seed_ids"])
            return [f for f in FACTS if {f["source_id"], f["target_id"]} & seeds]
        if "MENTIONS" in query:
            return [{"entity_id": s, "chunk_id": c} for s in p["seed_ids"] for c in MENTIONS[s]]
        if "c.chunk_id IN" in query:
            return [CHUNKS[c] for c in p["chunk_ids"] if c in CHUNKS]
        raise AssertionError(query)


def test_tokenize_and_lucene_query_are_injection_safe():
    assert tokenize("Who manages PM-KISAN?") == ["manages", "pm", "kisan"]
    assert build_lucene_query('name:"x" OR *) AND ~~') == "name OR x"
    assert build_lucene_query("what is the") == ""


def test_entity_match_requires_most_of_the_name():
    q = set(tokenize("Which company does John Smith work for?"))
    assert entity_match_score("John Smith", q) == 1.0
    assert entity_match_score("Life Insurance Corporation of India", q) == 0.0
    # one shared word of a long name is not enough
    q2 = set(tokenize("Tell me about insurance"))
    assert entity_match_score("Life Insurance Corporation of India", q2) < 0.6


def test_relation_verb_matching():
    from app.knowledge_graph.graph_retriever import relation_matches_query as m

    assert m("PRODUCES", set(tokenize("What does ABC produce?")))
    assert m("WORKS_FOR", set(tokenize("Where does John work?")))
    assert m("MANAGED_BY", set(tokenize("Who manages the fund?")))
    assert not m("WORKS_FOR", set(tokenize("What does ABC produce?")))


def test_bridge_fact_chunk_ranks_first_with_provenance():
    retriever = GraphRetriever(FakeClient())
    result = retriever.retrieve("Does John Smith work for ABC Corporation?", top_k=10)
    assert {m.name for m in result.matched_entities} == {"John Smith", "ABC Corporation"}
    top = result.evidence[0]
    assert top.chunk_id == "c2"  # fact linking both question entities, confidence 1.0
    assert top.facts[0].as_text() == "John Smith -[WORKS_FOR]-> ABC Corporation"
    assert "As CEO" in top.facts[0].evidence
    assert (top.document_name, top.page_number) == ("company.pdf", 2)
    # mention-only chunk ranks below chunks that state a fact
    order = [e.chunk_id for e in result.evidence]
    assert order.index("c4") > order.index("c1")


def test_search_returns_hits_for_fusion():
    hits = GraphRetriever(FakeClient()).search("What does ABC Corporation produce?", top_k=2)
    assert len(hits) == 2
    assert all(h.retriever == "knowledge_graph" for h in hits)
    assert hits[0].chunk_id == "c3"  # the PRODUCES fact matches the question's verb


def test_no_entities_means_no_graph_evidence():
    client = FakeClient()
    result = GraphRetriever(client).retrieve("What is the weather today?")
    assert result.evidence == [] and result.matched_entities == []
    assert len(client.calls) == 1  # never traversed the graph


def test_stopword_only_question_skips_neo4j():
    client = FakeClient()
    assert GraphRetriever(client).retrieve("what is the").evidence == []
    assert client.calls == []


def test_hybrid_retriever_uses_graph_and_survives_its_failure():
    from app.retrieval.hybrid_retriever import HybridRetriever

    class Broken:
        def search(self, query, top_k):
            raise RuntimeError("Neo4j down")

    ok = HybridRetriever(graph_retriever=GraphRetriever(FakeClient()), kg_top_k=5)
    fused = ok.retrieve("Who does John Smith work for?")
    assert fused and all("knowledge_graph" in f.retrievers for f in fused)

    broken = HybridRetriever(graph_retriever=Broken())
    assert broken.retrieve("Who does John Smith work for?") == []


# ---------------------------------------------------------------- API
@pytest.fixture
def graph_search_api():
    from fastapi.testclient import TestClient

    from app.core.dependencies import get_graph_retriever
    from app.main import app

    app.dependency_overrides[get_graph_retriever] = lambda: GraphRetriever(FakeClient())
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_graph_retriever, None)


def test_graph_search_endpoint(graph_search_api):
    r = graph_search_api.get("/search/graph", params={"q": "Who does John Smith work for?"})
    assert r.status_code == 200
    body = r.json()
    assert "John Smith" in [m["name"] for m in body["matched_entities"]]
    first = body["evidence"][0]
    assert first["facts"][0]["type"] == "WORKS_FOR"
    assert first["document_name"] == "company.pdf" and first["page_number"]


# ---------------------------------------------------------------- live AuraDB
def test_live_graph_retrieval(tmp_path):
    from app.core.config import get_settings
    from app.knowledge_graph.graph_builder import GraphBuilder
    from app.knowledge_graph.neo4j_client import Neo4jClient
    from test_graph_builder import make_graph

    settings = get_settings()
    if not settings.neo4j_uri:
        pytest.skip("Neo4j is not configured.")
    client = Neo4jClient(settings.neo4j_uri, settings.neo4j_username,
                         settings.neo4j_password.get_secret_value(), settings.neo4j_database)
    if not client.verify():
        pytest.skip("Neo4j is unreachable.")

    tag = uuid.uuid4().hex[:8]
    suffix, doc = f" Q{tag}", f"testQ{tag}"
    builder = GraphBuilder(client)
    builder.ensure_schema()
    graph, chunks = make_graph(doc, suffix)
    try:
        builder.write_document(graph, chunks)
        retriever = GraphRetriever(client)
        question = f"Which company does John Smith{suffix} work for?"
        # the full-text index updates asynchronously; give it a few seconds
        result = None
        for _ in range(20):
            result = retriever.retrieve(question, top_k=5)
            if result.evidence:
                break
            time.sleep(0.5)
        assert result and result.evidence, "no graph evidence found"
        names = {m.name for m in result.matched_entities}
        assert f"John Smith{suffix}" in names
        top = result.evidence[0]
        assert top.document_id == doc
        assert any(f.type == "WORKS_FOR" for f in top.facts)
    finally:
        builder.delete_document(doc)
        client.close()
