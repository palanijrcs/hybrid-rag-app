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


# ---------------------------------------------------------------- type rules
# relation -> (allowed source types, allowed target types). None = any type.
# Relations not listed (e.g. RELATED_TO, or custom ones) are not type-checked.
_AGENT = {"Person", "Organization", "Company"}
_ORG = {"Organization", "Company"}
_THING = {"Organization", "Company", "Scheme", "Product", "Technology", "Location",
          "Concept", "Event"}

RELATION_TYPE_RULES: dict[str, tuple[set[str] | None, set[str] | None]] = {
    "WORKS_FOR": ({"Person"}, _ORG),
    "MANAGED_BY": (_ORG | {"Scheme", "Product", "Event", "Concept"}, _AGENT),
    "FOUNDED_BY": (_ORG | {"Scheme"}, _AGENT),
    "FOUNDED_ON": (_ORG | {"Scheme", "Event"}, {"Date"}),
    "PRODUCES": (_ORG, {"Product", "Technology"}),
    "LOCATED_IN": (_AGENT | {"Event", "Location"}, {"Location"}),
    "HEADQUARTERED_IN": (_ORG, {"Location"}),
    "USED_BY": ({"Technology", "Product", "Scheme"}, _AGENT),
    "USES": (_AGENT, {"Technology", "Product", "Scheme"}),
    "PART_OF": (_THING, _THING),
    "OWNS": (_AGENT, _THING),
    "SUBSIDIARY_OF": (_ORG, _ORG),
    "PARTNERED_WITH": (_ORG, _ORG),
    "COMPETES_WITH": (_ORG | {"Product"}, _ORG | {"Product"}),
    "OCCURRED_ON": ({"Event"}, {"Date"}),
    "OCCURRED_IN": ({"Event"}, {"Location"}),
    "PARTICIPATED_IN": (_AGENT | {"Concept"}, {"Event", "Scheme"}),
    "ELIGIBLE_FOR": (_AGENT | {"Concept"}, {"Scheme", "Product"}),
    "PROVIDES": (_ORG | {"Scheme"}, _THING),
    "FUNDED_BY": ({"Scheme", "Event", "Product", "Organization", "Company"}, _AGENT),
}


def type_rule_violation(rel_type: str, source_type: str, target_type: str) -> str | None:
    rule = RELATION_TYPE_RULES.get(rel_type)
    if rule is None:
        return None
    allowed_src, allowed_tgt = rule
    if allowed_src is not None and source_type not in allowed_src:
        return f"source_type_not_allowed:{rel_type}:{source_type}"
    if allowed_tgt is not None and target_type not in allowed_tgt:
        return f"target_type_not_allowed:{rel_type}:{target_type}"
    return None


def names_in_evidence(evidence: str, *names: str) -> bool:
    ev = f" {canonical_key(evidence)} "
    return all(f" {canonical_key(n)} " in ev for n in names)


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
        elif settings.kg_require_names_in_evidence and not names_in_evidence(
            raw.evidence, src.name, tgt.name
        ):
            reason = "entities_not_named_in_evidence"
        elif settings.kg_enforce_type_rules and (
            violation := type_rule_violation(rtype, src.type, tgt.type)
        ):
            reason = violation
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
