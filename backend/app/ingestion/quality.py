"""Detect chunks whose text was extracted incorrectly from the PDF.

Some PDFs (common with Hindi/Tamil government documents) use fonts whose glyphs
don't map to proper Unicode. The extracted text then looks like Devanagari but is
broken, e.g. "्लग -्लग ्पनाने" or "पीएम-िेएमवाई". Such text is useless as evidence
and pollutes retrieval, so it is not indexed.

Rule for Indic scripts: a real word never starts with a vowel sign (matra), a
virama (्) or an anusvara/visarga. Broken extraction produces many such words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Devanagari (Hindi) and Tamil blocks
_INDIC_WORD = re.compile(r"[ऀ-ॿ஀-௿]+")
# combining marks that can never begin a word
_BAD_START = re.compile(
    r"^[ऀ-ःऺ-ॏ॑-ॗॢॣ"   # Devanagari signs
    r"ஂா-்ௗ]"                              # Tamil signs
)
_REPLACEMENT = "�"
_CID = re.compile(r"\(cid:\d+\)")


@dataclass
class TextQuality:
    garbled: bool
    reason: str = ""
    bad_word_ratio: float = 0.0


def assess_text(
    text: str,
    max_bad_indic_word_ratio: float = 0.05,
    min_indic_words: int = 5,
    max_replacement_ratio: float = 0.01,
) -> TextQuality:
    if not text or not text.strip():
        return TextQuality(True, "empty")

    visible = [c for c in text if not c.isspace()]
    bad_chars = sum(c == _REPLACEMENT for c in visible) + 5 * len(_CID.findall(text))
    if visible and bad_chars / len(visible) > max_replacement_ratio:
        return TextQuality(True, "unmapped_characters", bad_chars / len(visible))

    indic_words = _INDIC_WORD.findall(text)
    if len(indic_words) >= min_indic_words:
        bad = sum(1 for w in indic_words if _BAD_START.match(w))
        ratio = bad / len(indic_words)
        if ratio > max_bad_indic_word_ratio:
            return TextQuality(True, "broken_indic_font", round(ratio, 3))
        return TextQuality(False, "", round(ratio, 3))

    return TextQuality(False)


def split_by_quality(chunks: list, **thresholds) -> tuple[list, list[tuple[object, TextQuality]]]:
    """Return (good_chunks, [(bad_chunk, quality), ...]). Works with objects or dicts."""
    good, bad = [], []
    for chunk in chunks:
        text = chunk["text"] if isinstance(chunk, dict) else chunk.text
        quality = assess_text(text, **thresholds)
        (bad.append((chunk, quality)) if quality.garbled else good.append(chunk))
    return good, bad
