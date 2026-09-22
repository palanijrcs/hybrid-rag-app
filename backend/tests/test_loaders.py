"""Tests for document loaders."""

import pytest
from docx import Document

from app.ingestion.loaders import (
    DocumentLoadError,
    UnsupportedFileTypeError,
    load_document,
)


def test_loads_txt(tmp_path):
    file = tmp_path / "note.txt"
    file.write_text("ABC Company was founded in 1998.", encoding="utf-8")
    pages = load_document(file)
    assert "1998" in pages[0].text


def test_loads_markdown(tmp_path):
    file = tmp_path / "readme.md"
    file.write_text("# Title\nSome content", encoding="utf-8")
    assert "Some content" in load_document(file)[0].text


def test_loads_docx(tmp_path):
    file = tmp_path / "profile.docx"
    doc = Document()
    doc.add_paragraph("John Smith works for ABC Corporation.")
    doc.save(str(file))
    assert "John Smith" in load_document(file)[0].text


def test_rejects_unsupported_type(tmp_path):
    file = tmp_path / "sheet.xlsx"
    file.write_bytes(b"data")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(file)


def test_rejects_corrupted_pdf(tmp_path):
    file = tmp_path / "broken.pdf"
    file.write_bytes(b"this is not a real pdf")
    with pytest.raises(DocumentLoadError):
        load_document(file)


def test_rejects_empty_file(tmp_path):
    file = tmp_path / "empty.txt"
    file.write_text("   ", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        load_document(file)