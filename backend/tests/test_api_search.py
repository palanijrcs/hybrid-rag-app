"""Tests for search and index stats."""

PENSION = (
    "scheme.txt",
    b"A fixed pension of Rs.3,000 will be provided to all eligible small "
    b"and marginal farmers. It is a voluntary and contribution based scheme.",
    "text/plain",
)
BUNK = ("bunk.txt", b"The petrol bunk opens at six in the morning.", "text/plain")


def test_upload_indexes_vectors(client):
    client.post("/documents/upload", files={"file": PENSION})
    stats = client.get("/index/stats").json()
    assert stats["documents"] == 1
    assert stats["vectors"] == stats["chunks"] > 0


def test_search_finds_semantically_related_text(client):
    client.post("/documents/upload", files={"file": PENSION})
    client.post("/documents/upload", files={"file": BUNK})

    results = client.get("/search", params={"q": "How much money do farmers get?"}).json()[
        "results"
    ]
    assert results
    assert results[0]["document_name"] == "scheme.txt"
    assert results[0]["retriever"] == "vector"
    assert "3,000" in results[0]["text"]


def test_search_with_no_documents_returns_empty(client):
    assert client.get("/search", params={"q": "anything"}).json()["results"] == []


def test_empty_query_is_rejected(client):
    assert client.get("/search", params={"q": ""}).status_code == 422


def test_delete_also_removes_vectors(client):
    document_id = client.post("/documents/upload", files={"file": PENSION}).json()[
        "document"
    ]["document_id"]

    client.delete(f"/documents/{document_id}")
    assert client.get("/index/stats").json()["vectors"] == 0
    assert client.get("/search", params={"q": "pension"}).json()["results"] == []