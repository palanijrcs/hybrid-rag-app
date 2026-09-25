"""Phase 8 live smoke test: run LLM extraction on a text file and print the graph.

Usage (from backend/):
    python ../scripts/extract_graph_preview.py path/to/sample.txt
Nothing is written to Neo4j — that's Phase 9.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.knowledge_graph.extraction_config import get_kg_settings  # noqa: E402
from app.knowledge_graph.extraction_pipeline import build_graph_extraction_service  # noqa: E402


def naive_chunks(text: str, name: str, size: int = 2500) -> list[dict]:
    return [
        {"chunk_id": f"chunk_{i}", "document_id": "preview", "document_name": name,
         "text": text[p : p + size], "page_number": None, "section": None}
        for i, p in enumerate(range(0, len(text), size))
    ]


async def main(path: Path) -> None:
    settings = get_kg_settings()
    service = build_graph_extraction_service(settings)
    graph = await service.extract_document(naive_chunks(path.read_text(encoding="utf-8"), path.name))
    print(json.dumps(graph.stats, indent=2))
    for e in graph.entities:
        print(f"[{e.type}] {e.name}")
    for r in graph.relationships:
        print(f"{r.source_name} -[{r.type} {r.confidence:.2f}]-> {r.target_name}  | \"{r.evidence[0]}\"")
    for c in graph.chunk_results:
        for rej in c.rejected:
            print(f"REJECTED {rej.kind}: {rej.reason} {rej.item}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        sys.exit("usage: extract_graph_preview.py <file.txt>")
    asyncio.run(main(Path(sys.argv[1])))
