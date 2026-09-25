"""Pydantic models for knowledge-graph extraction.

Raw* models = untrusted LLM output. Entity / Relationship = validated,
source-attributed objects ready for Neo4j (Phase 9).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------- helpers ----------
def normalize_name(name: str) -> str:
    """Collapse whitespace and strip surrounding quotes/punctuation."""
    return re.sub(r"\s+", " ", name).strip().strip("\"'`.,;:")


def canonical_key(text: str) -> str:
    """Case/punctuation-insensitive key used for matching and dedup."""
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def make_entity_id(entity_type: str, name: str) -> str:
    return hashlib.sha1(f"{entity_type}|{canonical_key(name)}".encode()).hexdigest()[:16]


def make_relationship_id(source_id: str, rel_type: str, target_id: str) -> str:
    return hashlib.sha1(f"{source_id}|{rel_type}|{target_id}".encode()).hexdigest()[:16]


# ---------- provenance ----------
class SourceRef(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    page_number: int | None = None
    section: str | None = None


# ---------- raw LLM output ----------
class RawEntity(BaseModel):
    name: str
    type: str
    description: str = ""


class RawRelationship(BaseModel):
    source: str
    target: str
    type: str
    evidence: str = ""
    confidence: float = 0.5

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class RawExtraction(BaseModel):
    entities: list[RawEntity] = Field(default_factory=list)
    relationships: list[RawRelationship] = Field(default_factory=list)


# ---------- validated output ----------
class Entity(BaseModel):
    entity_id: str
    name: str
    type: str
    description: str = ""
    sources: list[SourceRef] = Field(default_factory=list)


class Relationship(BaseModel):
    relationship_id: str
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    type: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)


class Rejection(BaseModel):
    kind: str  # "entity" | "relationship"
    item: dict[str, Any]
    reason: str


class ChunkExtraction(BaseModel):
    chunk_id: str
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    rejected: list[Rejection] = Field(default_factory=list)
    error: str | None = None


class DocumentGraph(BaseModel):
    document_id: str
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    chunk_results: list[ChunkExtraction] = Field(default_factory=list)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "chunks": len(self.chunk_results),
            "failed_chunks": sum(1 for c in self.chunk_results if c.error),
            "entities": len(self.entities),
            "relationships": len(self.relationships),
            "rejected": sum(len(c.rejected) for c in self.chunk_results),
        }
