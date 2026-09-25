"""Relationship validation: every edge must connect grounded entities,
use an allowed type, meet the confidence floor, and quote real evidence."""
from __future__ import annotations

import re

from .extraction_config import KGExtractionSettings
from .models import (
    Entity,
    RawRelationship,
    Rejection,
    Relationship,
    SourceRef,
    canonical_key,
    make_relationship_id,
    normalize_name,
)


def normalize_relation_type(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", raw.strip().upper()).strip("_")


def evidence_is_grounded(evidence: str, chunk_text: str, min_overlap: float) -> bool:
    """Verbatim (normalized) match, or high token overlap to tolerate minor quote drift."""
    ev = canonical_key(evidence)
    if not ev:
        return False
    chunk = canonical_key(chunk_text)
    if ev in chunk:
        return True
    ev_tokens = ev.split()
    chunk_tokens = set(chunk.split())
    return sum(t in chunk_tokens for t in ev_tokens) / len(ev_tokens) >= min_overlap


def validate_relationships(
    raw_rels: list[RawRelationship],
    entities: dict[str, Entity],
    chunk_text: str,
    source: SourceRef,
    settings: KGExtractionSettings,
) -> tuple[list[Relationship], list[Rejection]]:
    allowed = set(settings.relation_types)
    accepted: dict[str, Relationship] = {}
    rejected: list[Rejection] = []

    for raw in raw_rels[: settings.kg_max_relationships_per_chunk]:
        src = entities.get(canonical_key(normalize_name(raw.source)))
        tgt = entities.get(canonical_key(normalize_name(raw.target)))
        rtype = normalize_relation_type(raw.type)

        reason = None
        if src is None or tgt is None:
            reason = "endpoint_not_a_valid_entity"
        elif src.entity_id == tgt.entity_id:
            reason = "self_loop"
        elif not rtype:
            reason = "empty_type"
        elif rtype not in allowed and not settings.kg_allow_unknown_relation_types:
            reason = f"type_not_allowed:{rtype}"
        elif raw.confidence < settings.kg_min_confidence:
            reason = f"low_confidence:{raw.confidence:.2f}"
        elif not evidence_is_grounded(
            raw.evidence, chunk_text, settings.kg_evidence_min_token_overlap
        ):
            reason = "evidence_not_in_chunk"
        if reason:
            rejected.append(Rejection(kind="relationship", item=raw.model_dump(), reason=reason))
            continue

        assert src is not None and tgt is not None
        rid = make_relationship_id(src.entity_id, rtype, tgt.entity_id)
        if rid in accepted:
            accepted[rid].confidence = max(accepted[rid].confidence, raw.confidence)
            continue
        accepted[rid] = Relationship(
            relationship_id=rid,
            source_id=src.entity_id,
            source_name=src.name,
            target_id=tgt.entity_id,
            target_name=tgt.name,
            type=rtype,
            confidence=raw.confidence,
            evidence=[raw.evidence.strip()[:500]],
            sources=[source],
        )
    return list(accepted.values()), rejected
