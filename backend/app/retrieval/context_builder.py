"""Phase 13: turn re-ranked chunks + graph facts into the LLM's evidence block.

    re-ranked chunks ──┐
                       ├─> clean text ─> drop repeated sentences ─> number sources
    graph facts ───────┘                                        ─> fit size budget
                                                                ─> BuiltContext

The output is the ONLY thing the grounded LLM (Phase 14) may answer from, so:
  * every piece of evidence gets a number [1], [2]... tied to document + page + chunk
  * higher-ranked evidence is kept first; lower-ranked evidence is dropped to fit
  * chunk overlap (from chunking) is removed so the same sentence isn't repeated
  * evidence text is wrapped in <source> tags and can't close them early
    (document text is data, never instructions)
  * an empty context is reported, so the caller can answer "not enough information"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.retrieval.fusion import FusedHit

# PDF bullets often come through as private-use characters such as U+F0B7
_PDF_BULLETS = re.compile(r"[▪●]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n\s*•\s*|\n{2,}")
_WORD = re.compile(r"\w+", re.UNICODE)
# a full stop after these is not the end of a sentence ("i.e. Rs.1500", "No. 5")
_ABBREVIATIONS = re.compile(
    r"\b(i\.e|e\.g|etc|viz|Rs|No|Nos|Dr|Mr|Mrs|Ms|Smt|Shri|Govt|Dept|approx|vs|Sl|St)\.",
    re.IGNORECASE,
)
_DOT = "\u2024"  # placeholder while splitting


# ---------------------------------------------------------------- data classes
@dataclass
class ContextSource:
    ref: int
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    retrievers: list[str]
    rerank_score: float | None
    text: str  # the text actually given to the LLM (cleaned, de-duplicated)

    @property
    def label(self) -> str:
        page = f", page {self.page_number}" if self.page_number else ""
        return f"{self.document_name}{page}"


@dataclass
class ContextFact:
    source_ref: int
    text: str       # "Pension Fund -[MANAGED_BY]-> Life Insurance Corporation of India"
    evidence: str   # verbatim quote from the source chunk


@dataclass
class BuiltContext:
    query: str
    text: str
    sources: list[ContextSource] = field(default_factory=list)
    facts: list[ContextFact] = field(default_factory=list)
    dropped_chunks: int = 0
    removed_duplicate_sentences: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.sources

    @property
    def char_count(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------- text helpers
def clean_text(text: str) -> str:
    """Normalise PDF artefacts: odd bullets, hard line wraps, extra spaces."""
    text = _PDF_BULLETS.sub("•", text)
    text = re.sub(r"[ \t]+", " ", text)
    # a single newline inside a sentence is a PDF line wrap -> space
    text = re.sub(r"(?<![.!?:।\n])\n(?!\s*•)(?!\n)", " ", text)
    text = re.sub(r"\s*•\s*", "\n• ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sentence_key(sentence: str) -> str:
    return " ".join(_WORD.findall(sentence.casefold()))


def split_sentences(text: str) -> list[str]:
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0)[:-1] + _DOT, text)
    protected = re.sub(r"(?<=\b[A-Za-z])\.(?=[A-Za-z]\.)", _DOT, protected)  # U.S.A.
    parts = [p.strip(" •\n") for p in _SENTENCE_SPLIT.split(protected)]
    return [p.replace(_DOT, ".") for p in parts if p]


def _escape(text: str) -> str:
    """Evidence must not be able to close or open our <source> tags."""
    return re.sub(r"</?\s*source", lambda m: m.group(0).replace("source", "source​"),
                  text, flags=re.IGNORECASE)


# ---------------------------------------------------------------- builder
class ContextBuilder:
    def __init__(
        self,
        max_chars: int = 6000,
        max_sources: int = 8,
        min_sentence_words: int = 4,
    ) -> None:
        self.max_chars = max_chars
        self.max_sources = max_sources
        self.min_sentence_words = min_sentence_words

    def build(
        self,
        query: str,
        hits: list[FusedHit],
        graph_evidence: Iterable[Any] | None = None,
    ) -> BuiltContext:
        """`hits` must already be in best-first order (re-ranked).
        `graph_evidence` is a list of GraphEvidence (Phase 10), optional."""
        seen_sentences: set[str] = set()
        sources: list[ContextSource] = []
        by_chunk: dict[str, ContextSource] = {}
        removed = dropped = 0
        used = 0

        for hit in hits:
            if hit.chunk_id in by_chunk:
                continue
            if len(sources) >= self.max_sources:
                dropped += 1
                continue

            kept_sentences = []
            for sentence in split_sentences(clean_text(hit.text)):
                key = _sentence_key(sentence)
                # short fragments left over from chunk overlap ("n scheme.") add nothing
                if len(key.split()) < self.min_sentence_words:
                    continue
                if key in seen_sentences:
                    removed += 1
                    continue
                seen_sentences.add(key)
                kept_sentences.append(sentence)
            if not kept_sentences:
                dropped += 1
                continue

            text = "\n".join(f"• {s}" for s in kept_sentences)
            if used + len(text) > self.max_chars:
                if not sources:  # always keep the best source, trimmed
                    text = text[: self.max_chars].rsplit(" ", 1)[0] + " …"
                else:
                    dropped += 1
                    continue
            used += len(text)

            src = ContextSource(
                ref=len(sources) + 1,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_name=hit.document_name,
                page_number=hit.page_number,
                retrievers=list(hit.retrievers),
                rerank_score=hit.rerank_score,
                text=text,
            )
            sources.append(src)
            by_chunk[hit.chunk_id] = src

        facts = self._attach_facts(graph_evidence or [], by_chunk)
        return BuiltContext(
            query=query,
            text=self.render(sources, facts),
            sources=sources,
            facts=facts,
            dropped_chunks=dropped,
            removed_duplicate_sentences=removed,
        )

    @staticmethod
    def _attach_facts(graph_evidence: Iterable[Any], by_chunk: dict[str, ContextSource]
                      ) -> list[ContextFact]:
        """Graph facts are only used when their source chunk passed re-ranking,
        so every fact the LLM sees is backed by text it can also read."""
        facts: list[ContextFact] = []
        seen: set[tuple[str, int]] = set()
        for ev in graph_evidence:
            src = by_chunk.get(ev.chunk_id)
            if src is None:
                continue
            for fact in ev.facts:
                text = f"{fact.source} -[{fact.type}]-> {fact.target}"
                if (text, src.ref) in seen:
                    continue
                seen.add((text, src.ref))
                facts.append(ContextFact(source_ref=src.ref, text=text, evidence=fact.evidence))
        return facts

    @staticmethod
    def render(sources: list[ContextSource], facts: list[ContextFact]) -> str:
        if not sources:
            return ""
        parts = []
        for s in sources:
            page = f' page="{s.page_number}"' if s.page_number else ""
            parts.append(
                f'<source id="{s.ref}" document="{_escape(s.document_name)}"{page}>\n'
                f"{_escape(s.text)}\n</source>"
            )
        if facts:
            lines = [f"- {_escape(f.text)} [{f.source_ref}]" for f in facts]
            parts.append("<graph_facts>\n" + "\n".join(lines) + "\n</graph_facts>")
        return "\n\n".join(parts)
