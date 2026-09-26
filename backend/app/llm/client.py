"""LLM client used for grounded answering (Phase 14).

A tiny protocol so the pipeline can be tested with a fake model, and so the
provider can be swapped later without touching the pipeline.
"""
from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


class LLMError(RuntimeError):
    """The language model could not be reached or returned nothing usable."""


class ChatClient(Protocol):
    model: str

    def complete_json(self, system: str, user: str) -> str: ...


class OpenAIChatClient:
    """OpenAI chat completions in JSON mode, temperature 0 for repeatable answers."""

    def __init__(
        self, api_key: str, model: str = DEFAULT_MODEL, timeout: float = 60.0,
        max_tokens: int = 800,
    ) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=2)
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens

    def complete_json(self, system: str, user: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # network, auth, rate limit, timeout
            # never log the key; the exception type and message are enough
            logger.error("llm.call_failed model=%s err=%s", self.model, type(exc).__name__)
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        return response.choices[0].message.content or ""
