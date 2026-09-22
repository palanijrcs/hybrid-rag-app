"""Entry point for the Hybrid RAG backend API."""

from fastapi import FastAPI

app = FastAPI(
    title="Hybrid RAG API",
    description="Grounded question answering over uploaded documents.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """Simple check that the server is running."""
    return {"status": "ok"}