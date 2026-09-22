"""Tests for the document API endpoints."""

TXT = ("note.txt", b"ABC Company was founded in 1998.", "text/plain")


def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_then_list_then_fetch(client):
    upload = client.post("/documents/upload", files={"file": TXT})
    assert upload.status_code == 201
    document_id = upload.json()["document"]["document_id"]

    listing = client.get("/documents")
    assert listing.status_code == 200
    assert len(listing.json()) == 1

    detail = client.get(f"/documents/{document_id}")
    assert detail.status_code == 200
    assert "1998" in detail.json()["chunks"][0]["text"]


def test_duplicate_upload_is_rejected(client):
    client.post("/documents/upload", files={"file": TXT})
    second = client.post("/documents/upload", files={"file": TXT})
    assert second.status_code == 409
    assert len(client.get("/documents").json()) == 1


def test_unsupported_type_is_rejected(client):
    files = {"file": ("data.xlsx", b"binary", "application/vnd.ms-excel")}
    assert client.post("/documents/upload", files=files).status_code == 415


def test_empty_file_is_rejected(client):
    files = {"file": ("empty.txt", b"", "text/plain")}
    assert client.post("/documents/upload", files=files).status_code == 400


def test_unreadable_pdf_is_rejected(client):
    files = {"file": ("broken.pdf", b"not a real pdf", "application/pdf")}
    assert client.post("/documents/upload", files=files).status_code == 422


def test_missing_document_returns_404(client):
    assert client.get("/documents/doesnotexist").status_code == 404
    assert client.delete("/documents/doesnotexist").status_code == 404


def test_delete_removes_the_document(client):
    document_id = client.post("/documents/upload", files={"file": TXT}).json()[
        "document"
    ]["document_id"]

    assert client.delete(f"/documents/{document_id}").json()["deleted"] is True
    assert client.get("/documents").json() == []
    # The same file can now be uploaded again
    assert client.post("/documents/upload", files={"file": TXT}).status_code == 201