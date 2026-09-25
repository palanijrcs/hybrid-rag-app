"""Phase 10: knowledge-graph retrieval.

    question
      -> entity identification  (Neo4j full-text index + token-overlap check, no LLM)
      -> 1-hop graph lookup      (facts touching those entities, and chunks mentioning them)
      -> chunk scoring           (facts > mentions; facts linking two question entities score highest)
      -> evidence                (chunk text + the facts it supports + matched entities)

`search()` returns VectorHit objects so it plugs into HybridRetriever / fusion like
FAISS and BM25. `retrieve_evidence()` returns the richer graph evidence for the
context builder and the /search/graph endpoint.

Everything returned is traceable: every fact carries the chunk it was extracted
from and the verbatim quote that supports it.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

from app.vectorstore.base import VectorHit

from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

FULLTEXT_INDEX = "entity_name_fulltext"

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "by", "with", "from",
    "at", "as", "is", "are", "was", "were", "be", "been", "does", "do", "did", "what",
    "which", "who", "whom", "whose", "when", "where", "why", "how", "this", "that",
    "these", "those", "it", "its", "their", "his", "her", "about", "tell", "me", "under",
    "can", "will", "should", "would", "much", "many", "any", "all", "there", "according",
    "document", "documents", "uploaded",
}
_TOKEN = re.compile(r"[a-z0-9]+")

Q_FIND_ENTITIES = """
CALL db.index.fulltext.queryNodes($index, $lucene, {limit: $limit})
YIELD node, score
RETURN node.entity_id AS entity_id, node.name AS name, node.type AS type, score
"""

Q_FACTS = """
UNWIND $seed_ids AS sid
MATCH (:Entity {entity_id: sid})-[r:RELATED_TO]-(:Entity)
WITH DISTINCT r
RETURN r.relationship_id AS relationship_id, r.type AS type,
       startNode(r).entity_id AS source_id, startNode(r).name AS source,
       endNode(r).entity_id AS target_id, endNode(r).name AS target,
       r.chunk_ids AS chunk_ids, r.evidence AS evidence, r.confidences AS confidences
LIMIT $limit
"""

Q_MENTIONS = """
UNWIND $seed_ids AS sid
MATCH (c:Chunk)-[:MENTIONS]->(:Entity {entity_id: sid})
RETURN sid AS entity_id, c.chunk_id AS chunk_id
LIMIT $limit
"""

Q_CHUNKS = """
MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids
RETURN c.chunk_id AS chunk_id, c.document_id AS document_id,
       c.document_name AS document_name, c.page_number AS page_number, c.text AS text
"""


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.casefold()) if t not in _STOPWORDS]


def build_lucene_query(query: str) -> str:
    """Only alphanumeric tokens reach Lucene, so no query syntax can be injected."""
    tokens = list(dict.fromkeys(tokenize(query)))
    return " OR ".join(tokens)


def entity_match_score(entity_name: str, query_tokens: set[str]) -> float:
    """Fraction of the entity's name words that appear in the question."""
    name_tokens = set(tokenize(entity_name))
    if not name_tokens:
        return 0.0
    return len(name_tokens & query_tokens) / len(name_tokens)


def relation_matches_query(rel_type: str, query_tokens: set[str]) -> bool:
    """True when the question uses the relationship's verb (produce ~ PRODUCES,
    work ~ WORKS_FOR, managed ~ MANAGED_BY), via a shared 4-5 letter word stem."""
    for word in tokenize(rel_type.replace("_", " ")):
        if len(word) < 4:
            continue
        stem = word[:5] if len(word) >= 5 else word
        for token in query_tokens:
            if len(token) >= 4 and (token.startswith(stem) or word.startswith(token[:5])):
                return True
    return False


# ---------------------------------------------------------------- result types
@dataclass
class MatchedEntity:
    entity_id: str
    name: str
    type: str
    match: float


@dataclass
class GraphFact:
    source: str
    type: str
    target: str
    evidence: str
    confidence: float

    def as_text(self) -> str:
        return f"{self.source} -[{self.type}]-> {self.target}"


@dataclass
class GraphEvidence:
    """One source chunk plus the graph facts it supports."""

    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    text: str
    score: float
    entities: list[str] = field(default_factory=list)
    facts: list[GraphFact] = field(default_factory=list)

    def to_hit(self) -> VectorHit:
        return VectorHit(
            chunk_id=self.chunk_id, document_id=self.document_id,
            document_name=self.document_name, text=self.text,
            page_number=self.page_number, score=self.score, retriever="knowledge_graph",
        )


@dataclass
class GraphRetrievalResult:
    query: str
    matched_entities: list[MatchedEntity]
    evidence: list[GraphEvidence]


# ---------------------------------------------------------------- retriever
class GraphRetriever:
    FACT_WEIGHT = 1.0      # chunk states a fact about a question entity
    BRIDGE_BONUS = 1.0     # ...and the fact links two question entities
    RELATION_BONUS = 1.0   # ...and the question uses the fact's verb
    MENTION_WEIGHT = 0.3   # chunk merely mentions a question entity

    def __init__(
        self,
        client: Neo4jClient,
        max_seed_entities: int = 5,
        min_entity_match: float = 0.6,
        candidate_limit: int = 25,
        max_facts: int = 50,
        max_mentions: int = 100,
    ) -> None:
        self.client = client
        self.max_seed_entities = max_seed_entities
        self.min_entity_match = min_entity_match
        self.candidate_limit = candidate_limit
        self.max_facts = max_facts
        self.max_mentions = max_mentions

    # -- Retriever protocol (used by HybridRetriever)
    def search(self, query: str, top_k: int) -> list[VectorHit]:
        return [ev.to_hit() for ev in self.retrieve(query, top_k).evidence]

    # -- full graph evidence
    def retrieve(self, query: str, top_k: int = 10) -> GraphRetrievalResult:
        seeds = self.identify_entities(query)
        if not seeds:
            logger.info("kg.retrieve no entities matched")
            return GraphRetrievalResult(query, [], [])

        seed_ids = [s.entity_id for s in seeds]
        match_by_id = {s.entity_id: s.match for s in seeds}
        name_by_id = {s.entity_id: s.name for s in seeds}
        query_tokens = set(tokenize(query))

        scores: dict[str, float] = defaultdict(float)
        facts_by_chunk: dict[str, list[GraphFact]] = defaultdict(list)
        entities_by_chunk: dict[str, set[str]] = defaultdict(set)

        for row in self.client.run(Q_FACTS, seed_ids=seed_ids, limit=self.max_facts):
            touched = [i for i in (row["source_id"], row["target_id"]) if i in match_by_id]
            weight = max(match_by_id[i] for i in touched)
            bridge = self.BRIDGE_BONUS if len(touched) == 2 else 0.0
            if relation_matches_query(row["type"] or "", query_tokens):
                bridge += self.RELATION_BONUS
            chunk_ids = row["chunk_ids"] or []
            evidence = row["evidence"] or []
            confidences = row["confidences"] or []
            for i, chunk_id in enumerate(chunk_ids):
                confidence = float(confidences[i]) if i < len(confidences) else 0.5
                scores[chunk_id] += (self.FACT_WEIGHT + bridge) * weight * confidence
                facts_by_chunk[chunk_id].append(GraphFact(
                    source=row["source"], type=row["type"], target=row["target"],
                    evidence=evidence[i] if i < len(evidence) else "", confidence=confidence,
                ))
                entities_by_chunk[chunk_id].update(name_by_id[t] for t in touched)

        for row in self.client.run(Q_MENTIONS, seed_ids=seed_ids, limit=self.max_mentions):
            scores[row["chunk_id"]] += self.MENTION_WEIGHT * match_by_id[row["entity_id"]]
            entities_by_chunk[row["chunk_id"]].add(name_by_id[row["entity_id"]])

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        chunks = {
            row["chunk_id"]: row
            for row in self.client.run(Q_CHUNKS, chunk_ids=[cid for cid, _ in ranked])
        }
        evidence = [
            GraphEvidence(
                chunk_id=cid,
                document_id=chunks[cid]["document_id"],
                document_name=chunks[cid]["document_name"],
                page_number=chunks[cid]["page_number"],
                text=chunks[cid]["text"] or "",
                score=round(score, 4),
                entities=sorted(entities_by_chunk[cid]),
                facts=facts_by_chunk[cid],
            )
            for cid, score in ranked
            if cid in chunks  # skip anything deleted since
        ]
        logger.info("kg.retrieve seeds=%d evidence=%d", len(seeds), len(evidence))
        return GraphRetrievalResult(query, seeds, evidence)

    def identify_entities(self, query: str) -> list[MatchedEntity]:
        """Find graph entities named in the question."""
        lucene = build_lucene_query(query)
        if not lucene:
            return []
        rows = self.client.run(
            Q_FIND_ENTITIES, index=FULLTEXT_INDEX, lucene=lucene, limit=self.candidate_limit
        )
        query_tokens = set(tokenize(query))
        matched: dict[str, MatchedEntity] = {}
        for row in rows:
            score = entity_match_score(row["name"] or "", query_tokens)
            if score >= self.min_entity_match and row["entity_id"] not in matched:
                matched[row["entity_id"]] = MatchedEntity(
                    row["entity_id"], row["name"], row["type"], round(score, 3)
                )
        best = sorted(matched.values(), key=lambda m: (m.match, len(m.name)), reverse=True)
        return best[: self.max_seed_entities]
