"""Phase 8 settings for LLM-based entity/relationship extraction.

Kept separate from core/config.py so it can be merged in later without
touching existing settings. All values come from environment variables / .env.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/knowledge_graph/extraction_config.py -> project root
_ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"

DEFAULT_ENTITY_TYPES = (
    "Person,Organization,Company,Scheme,Product,Location,Technology,Concept,Date,Event"
)
DEFAULT_RELATION_TYPES = (
    "WORKS_FOR,MANAGED_BY,FOUNDED_BY,FOUNDED_ON,PRODUCES,LOCATED_IN,HEADQUARTERED_IN,"
    "USED_BY,USES,PART_OF,OWNS,SUBSIDIARY_OF,PARTNERED_WITH,COMPETES_WITH,"
    "OCCURRED_ON,OCCURRED_IN,PARTICIPATED_IN,ELIGIBLE_FOR,PROVIDES,FUNDED_BY,RELATED_TO"
)


class KGExtractionSettings(BaseSettings):
    # Project-root .env first, then CWD .env (later files override). Missing files are
    # ignored, and real environment variables (Docker) always win.
    model_config = SettingsConfigDict(env_file=(_ROOT_ENV, ".env"), extra="ignore")

    openai_api_key: SecretStr | None = None
    kg_extraction_model: str = "gpt-4o-mini"
    kg_extraction_temperature: float = 0.0
    kg_request_timeout: float = 60.0
    kg_max_retries: int = 2
    kg_max_concurrency: int = 4

    kg_entity_types: str = DEFAULT_ENTITY_TYPES
    kg_relation_types: str = DEFAULT_RELATION_TYPES
    kg_allow_unknown_relation_types: bool = False

    kg_build_on_upload: bool = True
    kg_min_confidence: float = 0.7
    # Both entity names must appear in the evidence quote
    kg_require_names_in_evidence: bool = True
    # Enforce source/target entity types per relationship (see RELATION_TYPE_RULES)
    kg_enforce_type_rules: bool = True
    kg_max_entities_per_chunk: int = 25
    kg_max_relationships_per_chunk: int = 40
    kg_max_chunk_chars: int = 6000
    kg_evidence_min_token_overlap: float = 0.8

    # Phase 10: graph retrieval
    kg_max_seed_entities: int = 5
    kg_min_entity_match: float = 0.6  # share of an entity's name words found in the question
    kg_max_facts: int = 50

    @property
    def entity_types(self) -> list[str]:
        return [t.strip() for t in self.kg_entity_types.split(",") if t.strip()]

    @property
    def relation_types(self) -> list[str]:
        return [t.strip().upper() for t in self.kg_relation_types.split(",") if t.strip()]


@lru_cache
def get_kg_settings() -> KGExtractionSettings:
    return KGExtractionSettings()
