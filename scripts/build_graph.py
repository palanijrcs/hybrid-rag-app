"""Build the knowledge graph for documents that are already uploaded.

Usage (from backend/):
    python ../scripts/build_graph.py              # documents not yet in the graph
    python ../scripts/build_graph.py --all        # rebuild every document
    python ../scripts/build_graph.py --document-id <id>
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.dependencies import (  # noqa: E402
    get_document_store,
    get_graph_builder,
    get_graph_indexer,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="rebuild every document")
    parser.add_argument("--document-id", help="build a single document")
    args = parser.parse_args()

    indexer = get_graph_indexer()
    builder = get_graph_builder()
    if indexer is None or builder is None:
        print("Graph building is disabled. Check NEO4J_*, OPENAI_API_KEY, "
              "ENABLE_KG_RETRIEVAL and KG_BUILD_ON_UPLOAD in .env.")
        return 1
    builder.ensure_schema()

    store = get_document_store()
    records = store.list_documents()
    if args.document_id:
        records = [r for r in records if r.document_id == args.document_id]
        if not records:
            print(f"No document with id {args.document_id}")
            return 1

    failures = 0
    for record in records:
        current = builder.get_status(record.document_id)
        if not args.all and not args.document_id and current and current.get("kg_status") == "done":
            print(f"skip  {record.filename} (already built)")
            continue
        print(f"build {record.filename} ({record.chunk_count} chunks) ...", flush=True)
        result = indexer.index_document(
            record.document_id, record.filename, store.get_chunks(record.document_id)
        )
        print(f"      -> {result}")
        failures += result.get("status") == "failed"
    return 1 if failures else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
