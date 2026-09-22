"""Tests for text cleaning and chunking."""

import pytest

from app.ingestion.chunking import chunk_pages
from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import PageText


def test_clean_joins_hyphenated_words():
    assert "recommendations" in clean_text("recommen-\ndations")


def test_clean_removes_extra_spacing():
    assert clean_text("A    B\n\n\n\nC") == "A B\n\nC"


def test_chunks_carry_metadata():
    pages = [PageText(page_number=1, text="ABC Company was founded in 1998.")]
    chunks = chunk_pages(pages, "doc1", "company.pdf", 800, 120)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "doc1_chunk_0"
    assert chunk.document_id == "doc1"
    assert chunk.document_name == "company.pdf"
    assert chunk.page_number == 1
    assert chunk.created_at


def test_long_text_is_split_into_several_chunks():
    text = "This is a sentence about the company. " * 100
    chunks = chunk_pages([PageText(1, text)], "doc1", "a.pdf", 400, 60)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_chunks_overlap_so_nothing_is_lost():
    text = "word " * 400
    chunks = chunk_pages([PageText(1, text)], "doc1", "a.pdf", 300, 80)
    first_tail = chunks[0].text[-40:]
    assert first_tail.split()[0] in chunks[1].text


def test_page_numbers_are_kept_and_ids_run_in_order():
    pages = [PageText(1, "Page one text."), PageText(2, "Page two text.")]
    chunks = chunk_pages(pages, "doc1", "a.pdf", 800, 120)
    assert [c.page_number for c in chunks] == [1, 2]
    assert [c.chunk_id for c in chunks] == ["doc1_chunk_0", "doc1_chunk_1"]


def test_blank_pages_are_skipped():
    pages = [PageText(1, "   "), PageText(2, "Real content here.")]
    chunks = chunk_pages(pages, "doc1", "a.pdf", 800, 120)
    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, "text")], "doc1", "a.pdf", 100, 100)