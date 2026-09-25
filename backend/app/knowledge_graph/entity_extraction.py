"""LLM call + entity validation.

- LLMJsonClient: tiny protocol so tests can inject a fake (no network).
- OpenAIJsonClient: real implementation using JSON mode.
- LLMGraphExtractor: prompts the LLM, parses/validates JSON, retries on bad output.
- validate_entities: grounding + type checks.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol

from pydantic import ValidationError

from .extraction_config import KGExtractionSettings
from .extraction_prompts import build_messages
from .models import (
    Entity,
    RawEntity,
    RawExtraction,
    Rejection,
    SourceRef,
    canonical_key,
    make_entity_id,
    normalize_name,
)

logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    """Raised when the LLM cannot produce valid extraction output."""


class LLMJsonClient(Protocol):
    async def complete_json(self, system: str, user: str) -> str: ...


class OpenAIJsonClient:
    """OpenAI chat completion in JSON mode. API key comes from settings/env only."""

    def __init__(self, settings: KGExtractionSettings) -> None:
        from openai import AsyncOpenAI  # lazy import keeps tests dependency-light

        if settings.openai_api_key is None:
            raise ExtractionError("OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.kg_request_timeout,
            max_retries=2,  # SDK-level retries for 429/5xx
        )
        self._model = settings.kg_extraction_model
        self._temperature = settings.kg_extraction_temperature

    async def complete_json(self, system: str, user: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or "{}"


class LLMGraphExtractor:
    def __init__(self, client: LLMJsonClient, settings: KGExtractionSettings) -> None:
        self.client = client
        self.settings = settings

    async def extract_raw(
        self, chunk_text: str, document_name: str, section: str | None = None
    ) -> RawExtraction:
        text = chunk_text[: self.settings.kg_max_chunk_chars]
        system, user = build_messages(
            chunk_text=text,
            document_name=document_name,
            section=section,
            entity_types=self.settings.entity_types,
            relation_types=self.settings.relation_types,
            max_entities=self.settings.kg_max_entities_per_chunk,
            max_relationships=self.settings.kg_max_relationships_per_chunk,
        )
        last_err: Exception | None = None
        for attempt in range(self.settings.kg_max_retries + 1):
            try:
                content = await self.client.complete_json(system, user)
                data: Any = json.loads(content)
                if not isinstance(data, dict):
                    raise ValueError("top-level JSON is not an object")
                return RawExtraction.model_validate(data)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_err = exc
                logger.warning("kg.extract.invalid_output attempt=%d err=%s", attempt + 1, exc)
                await asyncio.sleep(0.5 * (attempt + 1))
        raise ExtractionError(f"invalid LLM output after retries: {last_err}")


def _match_type(raw_type: str, allowed: list[str]) -> str | None:
    lookup = {t.casefold(): t for t in allowed}
    return lookup.get(raw_type.strip().casefold())


def validate_entities(
    raw_entities: list[RawEntity],
    chunk_text: str,
    allowed_types: list[str],
    source: SourceRef,
    max_entities: int,
) -> tuple[dict[str, Entity], list[Rejection]]:
    """Return {canonical_name: Entity} for grounded entities, plus rejections."""
    chunk_key = f" {canonical_key(chunk_text)} "
    accepted: dict[str, Entity] = {}
    rejected: list[Rejection] = []

    for raw in raw_entities[:max_entities]:
        name = normalize_name(raw.name)
        key = canonical_key(name)
        etype = _match_type(raw.type, allowed_types)

        reason = None
        if len(key) < 2:
            reason = "empty_or_too_short_name"
        elif etype is None:
            reason = f"type_not_allowed:{raw.type}"
        elif f" {key} " not in chunk_key:
            reason = "name_not_found_in_chunk"  # hallucinated entity
        if reason:
            rejected.append(Rejection(kind="entity", item=raw.model_dump(), reason=reason))
            continue

        if key in accepted:  # duplicate within chunk: keep longer description
            if len(raw.description) > len(accepted[key].description):
                accepted[key].description = raw.description.strip()
            continue
        accepted[key] = Entity(
            entity_id=make_entity_id(etype, name),
            name=name,
            type=etype,
            description=raw.description.strip()[:300],
            sources=[source],
        )
    return accepted, rejected
