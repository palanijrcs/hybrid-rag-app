"""Phase 15: input guardrail — runs before any retrieval or LLM call.

Rule-based on purpose: fast, free, predictable and testable. It does not try to
decide whether a question is *answerable* — retrieval + re-ranking + the grounded
LLM already refuse questions the documents don't cover. It blocks input that
should never reach the pipeline:

    empty / too long / not a question        -> "invalid"
    attempts to override the system          -> "prompt_injection"
    secrets pasted into the question         -> "sensitive_data"
    tasks that aren't questions about docs   -> "out_of_scope"
    greetings / thanks                       -> "chit_chat" (friendly reply)

Every change made to the question (Unicode normalisation, removed hidden
characters, collapsed whitespace) is listed in `transformations`, never silent.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------- patterns
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

_INJECTION = [
    (r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|your|system|these|the)\b.{0,20}\b(instructions?|rules?|prompts?|guidelines?|directions?)", "asks to ignore instructions"),
    (r"\b(reveal|show|print|display|repeat|output|leak|tell me|what (is|are))\b.{0,30}\b(system|hidden|initial|original|developer)\s+(prompt|instructions?|message|rules)", "asks for the system prompt"),
    (r"\byou are (now|no longer)\b", "tries to change the assistant's role"),
    (r"\b(pretend|roleplay|role-play)\b.{0,30}\b(you|to be|as)\b", "tries to change the assistant's role"),
    (r"\bact as (?:an? )?(?:unrestricted|unfiltered|jailbroken|different|new|evil)\b", "tries to change the assistant's role"),
    (r"\b(jailbreak|DAN mode|developer mode|god mode|do anything now)\b", "jailbreak attempt"),
    (r"\bnew (instructions?|rules|system prompt)\s*:", "injects new instructions"),
    (r"(^|\n)\s*(system|assistant|developer)\s*:", "imitates a system message"),
    (r"</?\s*(sources?|system|instructions?)\s*>", "contains prompt markup"),
    (r"\b(answer|respond)\b.{0,30}\b(without|ignoring|regardless of)\b.{0,20}\b(sources?|documents?|evidence|context|rules?)", "asks to answer without the documents"),
    (r"\buse your (own )?(general |outside )?knowledge\b", "asks to answer without the documents"),
]
_INJECTION = [(re.compile(p, re.IGNORECASE | re.DOTALL), why) for p, why in _INJECTION]

_SECRETS = [
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),             # OpenAI keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                           # AWS access keys
    re.compile(r"\b(?:\d[ -]?){13,19}\b(?=.*\b(card|credit|debit|cvv)\b)", re.IGNORECASE),
    re.compile(r"\bpassword\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"neo4j\+s://\S+:\S+@", re.IGNORECASE),              # URI with credentials
]

_OUT_OF_SCOPE = re.compile(
    r"^\s*(please\s+)?(write|compose|generate|create|draft)\s+(me\s+)?(an?\s+)?"
    r"(poem|song|story|joke|essay|code|program|script|python|sql|email|letter|tweet|rap)\b"
    r"|^\s*(tell|give)\s+me\s+a\s+joke\b"
    r"|\b(what('s| is) the (weather|time|date)( today| now)?)\s*\??$",
    re.IGNORECASE,
)

_CHIT_CHAT = re.compile(
    r"^\s*(hi+|hello+|hey+|hai|vanakkam|namaste|good (morning|afternoon|evening)|"
    r"thanks?( you)?|thank you( so much)?|ok(ay)?|bye|who are you\??|how are you\??)"
    r"[\s!.?]*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------- messages
MESSAGES = {
    "empty": "Please type a question about your uploaded documents.",
    "too_long": "Your question is too long ({length} characters). Please keep it under {limit}.",
    "not_a_question": "Please ask a question in words about your uploaded documents.",
    "prompt_injection": (
        "I can only answer questions about your uploaded documents, and I can't change "
        "or reveal how I work. Please rephrase your question."
    ),
    "sensitive_data": (
        "Your question seems to contain a password, API key or card number. It was not "
        "processed. Please remove it and ask again."
    ),
    "out_of_scope": (
        "I answer questions using your uploaded documents only. I can't do other tasks "
        "such as writing stories, code or jokes."
    ),
    "chit_chat": (
        "Hello! Ask me a question about your uploaded documents and I'll answer with "
        "sources."
    ),
}


@dataclass
class GuardrailResult:
    allowed: bool
    sanitized_query: str
    category: str = "ok"      # ok | invalid | prompt_injection | sensitive_data | out_of_scope | chit_chat
    reason: str = ""
    message: str = ""         # what to show the user when blocked
    transformations: list[str] = field(default_factory=list)


def sanitize(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    normalized = unicodedata.normalize("NFKC", text)
    if normalized != text:
        changes.append("normalized Unicode characters")
    without_hidden = _ZERO_WIDTH.sub("", normalized)
    if without_hidden != normalized:
        changes.append("removed invisible characters")
    without_control = _CONTROL.sub(" ", without_hidden)
    if without_control != without_hidden:
        changes.append("removed control characters")
    collapsed = re.sub(r"\s+", " ", without_control).strip()
    if collapsed != without_control.strip():
        changes.append("collapsed extra whitespace")
    return collapsed, changes


class InputGuardrail:
    def __init__(self, max_chars: int = 1000, min_letters: int = 2,
                 block_out_of_scope: bool = True) -> None:
        self.max_chars = max_chars
        self.min_letters = min_letters
        self.block_out_of_scope = block_out_of_scope

    def check(self, question: str) -> GuardrailResult:
        cleaned, changes = sanitize(question or "")

        def block(category: str, reason: str, key: str | None = None, **fmt) -> GuardrailResult:
            return GuardrailResult(False, cleaned, category, reason,
                                   MESSAGES[key or category].format(**fmt), changes)

        if not cleaned:
            return block("invalid", "empty question", "empty")
        if len(cleaned) > self.max_chars:
            return block("invalid", "question too long", "too_long",
                         length=len(cleaned), limit=self.max_chars)
        if len(_LETTER.findall(cleaned)) < self.min_letters:
            return block("invalid", "no words in question", "not_a_question")

        # check the raw text too: hidden characters are a common way to disguise injections
        for text in {cleaned, question}:
            for pattern in _SECRETS:
                if pattern.search(text):
                    return block("sensitive_data", "question contains a secret")
            for pattern, why in _INJECTION:
                if pattern.search(text):
                    return block("prompt_injection", why)

        if _CHIT_CHAT.match(cleaned):
            return block("chit_chat", "greeting or small talk")
        if self.block_out_of_scope and _OUT_OF_SCOPE.search(cleaned):
            return block("out_of_scope", "task is not a question about the documents")

        return GuardrailResult(True, cleaned, "ok", "", "", changes)
