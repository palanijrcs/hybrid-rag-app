"""Tests for search and index stats."""

PENSION = (
    "scheme.txt",
    b"A fixed pension of Rs.3,000 will be provided to all eligible small "
    b"and marginal farmers. It is a voluntary and contribution based scheme.",
    "text/plain",
)
BUNK = ("bunk.txt", b"The petrol bunk opens at six in the morning.", "text/plain")


def upload(client, file):
    return client.post("/documents/upload", files={"file": file})


def test_upload_indexes_vectors(client):
    upload(client, PENSION)
    stats = client.get("/index/stats").json()
    assert stats["documents"] == 1
    assert stats["vectors"] == stats["chunks"] > 0


def test_search_finds_semantically_related_text(client):
    upload(client, PENSION)
    upload(client, BUNK)
    body = client.get("/search", params={"q": "How much money do farmers get?"}).json()
    results = body["results"]
    assert results
    assert results[0]["document_name"] == "scheme.txt"
    assert results[0]["retriever"] == "vector"
    assert "3,000" in results[0]["text"]


def test_search_with_no_documents_returns_empty(client):
    assert client.get("/search", params={"q": "anything"}).json()["results"] == []


def test_empty_query_is_rejected(client):
    assert client.get("/search", params={"q": ""}).status_code == 422


def test_delete_also_removes_vectors(client):
    document_id = upload(client, PENSION).json()["document"]["document_id"]
    client.delete(f"/documents/{document_id}")
    assert client.get("/index/stats").json()["vectors"] == 0
    assert client.get("/search", params={"q": "pension"}).json()["results"] == []


def test_bm25_is_indexed_on_upload(client):
    upload(client, PENSION)
    stats = client.get("/index/stats").json()
    assert stats["bm25_chunks"] == stats["chunks"] > 0


def test_bm25_search_matches_exact_words(client):
    upload(client, PENSION)
    upload(client, BUNK)
    body = client.get("/search", params={"q": "petrol bunk", "method": "bm25"}).json()
    assert body["results"][0]["document_name"] == "bunk.txt"
    assert body["results"][0]["retriever"] == "bm25"


def test_compare_returns_both_methods(client):
    upload(client, PENSION)
    body = client.get("/search/compare", params={"q": "pension"}).json()
    assert body["vector"] and body["bm25"]
    assert body["vector"][0]["retriever"] == "vector"
    assert body["bm25"][0]["retriever"] == "bm25"


def test_invalid_method_is_rejected(client):
    assert client.get("/search", params={"q": "x", "method": "magic"}).status_code == 422


def test_delete_also_refreshes_bm25(client):
    document_id = upload(client, PENSION).json()["document"]["document_id"]
    client.delete(f"/documents/{document_id}")
    assert client.get("/index/stats").json()["bm25_chunks"] == 0
