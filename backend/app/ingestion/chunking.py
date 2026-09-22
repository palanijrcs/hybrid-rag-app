"""Split cleaned document text into overlapping chunks with metadata."""

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import PageText

# Preferred split points, from best to worst
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """One searchable piece of a document, with everything needed to cite it."""

    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_number: int | None = None
    section: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Cut text into pieces of about chunk_size, breaking at natural points."""
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            pieces.append(text[start:])
            break

        window = text[start:end]
        # Try to end at a paragraph break, then a sentence end, then a space
        split_at = None
        for pattern in (_PARAGRAPH_BREAK, _SENTENCE_END):
            matches = list(pattern.finditer(window))
            # Only accept a break in the second half, so chunks stay reasonably full
            usable = [m for m in matches if m.end() > chunk_size // 2]
            if usable:
                split_at = usable[-1].end()
                break
        if split_at is None:
            space = window.rfind(" ", chunk_size // 2)
            split_at = space if space != -1 else chunk_size

        pieces.append(text[start : start + split_at])
        start = start + split_at - overlap
        if start < 0:
            start = 0

    return [p.strip() for p in pieces if p.strip()]


def chunk_pages(
    pages: list[PageText],
    document_id: str,
    document_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Clean each page and split it into chunks, numbered across the document."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    counter = 0

    for page in pages:
        text = clean_text(page.text)
        if not text:
            continue
        for piece in _split_text(text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}_chunk_{counter}",
                    document_id=document_id,
                    document_name=document_name,
                    text=piece,
                    page_number=page.page_number,
                )
            )
            counter += 1

    return chunks