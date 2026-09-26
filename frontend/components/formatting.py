"""Pure formatting helpers for the answer view (no Streamlit import, easy to test)."""
from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION = re.compile(r"\[(\d{1,2})\]")

RETRIEVER_BADGES = {
    "vector": ":blue-badge[Semantic]",
    "bm25": ":green-badge[Keyword]",
    "knowledge_graph": ":violet-badge[Graph]",
}
RETRIEVER_NAMES = {"vector": "Vector", "bm25": "BM25", "knowledge_graph": "Knowledge Graph"}


def link_citations(answer: str, sources: list[dict]) -> str:
    """Turn [1] into a link that jumps to source card 1 and shows its page on hover.
    Citations to sources that aren't in the list are left as plain text."""
    labels = {s["ref"]: s.get("label", "") for s in sources}

    def repl(m: re.Match) -> str:
        ref = int(m.group(1))
        if ref not in labels:
            return m.group(0)
        title = labels[ref].replace('"', "'")
        return f'[\\[{ref}\\]](#source-{ref} "{title}")'

    return _CITATION.sub(repl, _escape_markdown(answer))


def _escape_markdown(text: str) -> str:
    """Stop document text from being read as Markdown/LaTeX ($ signs, #, * at line start)."""
    text = text.replace("$", "\\$")
    return re.sub(r"(?m)^(\s*)([#>])", r"\1\\\2", text)


def retriever_badges(retrievers: list[str]) -> str:
    return " ".join(RETRIEVER_BADGES.get(r, f":gray-badge[{r}]") for r in retrievers)


def retrieval_method(retrieval: dict) -> str:
    """'Hybrid (Vector + BM25)' style summary of which retrievers found anything."""
    used = [RETRIEVER_NAMES[k] for k in RETRIEVER_NAMES if retrieval.get(k, 0) > 0]
    if not used:
        return "None"
    return used[0] if len(used) == 1 else f"Hybrid ({' + '.join(used)})"


def relevance_label(score: float | None) -> str:
    if score is None:
        return "not re-ranked"
    if score >= 0.9:
        return f"very high ({score:.2f})"
    if score >= 0.6:
        return f"high ({score:.2f})"
    if score >= 0.3:
        return f"medium ({score:.2f})"
    return f"low ({score:.2f})"


@dataclass
class StatusBanner:
    kind: str      # success | info | warning | error
    message: str


def status_banners(response: dict) -> list[StatusBanner]:
    """What the user should be told about how this answer was produced."""
    banners: list[StatusBanner] = []
    guard_in = response.get("input_guardrail") or {}
    guard_out = response.get("output_guardrail") or {}

    if guard_in and not guard_in.get("allowed", True):
        category = guard_in.get("category", "")
        if category == "chit_chat":
            return []
        kind = "error" if category in ("prompt_injection", "sensitive_data") else "warning"
        return [StatusBanner(kind, f"Question not processed ({category.replace('_', ' ')}).")]

    if response.get("insufficient_evidence"):
        banners.append(StatusBanner(
            "warning", "The uploaded documents don't contain enough information to answer this."))
        return banners

    answer_has_notice = str(response.get("answer", "")).startswith("Note: the uploaded documents")
    if response.get("coverage") == "mentioned_only" and not answer_has_notice:
        banners.append(StatusBanner(
            "info", f"The documents only mention **{response.get('question_subject') or 'this'}** "
                    f"in passing; they are about **{response.get('sources_subject') or 'something else'}**."))
    elif response.get("coverage") == "partial":
        banners.append(StatusBanner("info", "The documents answer only part of this question."))

    removed = guard_out.get("removed_sentences", 0)
    corrected = guard_out.get("corrected_citations", 0)
    if removed:
        banners.append(StatusBanner(
            "warning", f"{removed} statement(s) were removed because the sources don't support them."))
    if corrected:
        banners.append(StatusBanner("info", f"Citations were corrected in {corrected} sentence(s)."))
    if guard_out.get("regenerated"):
        banners.append(StatusBanner("info", "The first draft failed verification and was rewritten."))

    if response.get("grounded") and response.get("sources") and not removed:
        verified = " and verified" if guard_out.get("verifier_used") else ""
        banners.append(StatusBanner(
            "success", f"Grounded in {len(response['sources'])} source(s){verified}."))
    return banners


def source_heading(source: dict) -> str:
    return f"[{source['ref']}] {source.get('label') or source.get('document_name', '')}"
