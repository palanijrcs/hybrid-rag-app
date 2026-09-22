"""Save uploaded documents and keep a registry of documents and their chunks."""

import hashlib
import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from app.ingestion.chunking import Chunk


class DuplicateDocumentError(Exception):
    """Raised when the same file content has already been ingested."""


@dataclass
class DocumentRecord:
    """What we know about one ingested document."""

    document_id: str
    filename: str
    file_type: str
    file_hash: str
    uploaded_at: str
    chunk_count: int
    page_count: int
    source: str


def compute_hash(content: bytes) -> str:
    """Fingerprint of the file contents, used to spot duplicates."""
    return hashlib.sha256(content).hexdigest()


def new_document_id() -> str:
    return uuid.uuid4().hex[:12]


class DocumentStore:
    """Stores uploaded files, their chunks, and a registry index on disk."""

    def __init__(self, upload_dir: Path, registry_path: Path) -> None:
        self.upload_dir = upload_dir
        self.registry_path = registry_path
        self.chunks_dir = registry_path.parent / "chunks"
        self._lock = Lock()

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- registry ----------
    def _read_registry(self) -> dict[str, dict]:
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write_registry(self, registry: dict[str, dict]) -> None:
        self.registry_path.write_text(
            json.dumps(registry, indent=2), encoding="utf-8"
        )

    # ---------- queries ----------
    def list_documents(self) -> list[DocumentRecord]:
        return [DocumentRecord(**item) for item in self._read_registry().values()]

    def get_document(self, document_id: str) -> DocumentRecord | None:
        item = self._read_registry().get(document_id)
        return DocumentRecord(**item) if item else None

    def find_by_hash(self, file_hash: str) -> DocumentRecord | None:
        for item in self._read_registry().values():
            if item["file_hash"] == file_hash:
                return DocumentRecord(**item)
        return None

    def get_chunks(self, document_id: str) -> list[dict]:
        path = self.chunks_dir / f"{document_id}.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def all_chunks(self) -> list[dict]:
        chunks: list[dict] = []
        for document_id in self._read_registry():
            chunks.extend(self.get_chunks(document_id))
        return chunks

    # ---------- writes ----------
    def save_upload(self, document_id: str, filename: str, content: bytes) -> Path:
        """Write the raw file to disk under a name that cannot collide."""
        suffix = Path(filename).suffix.lower()
        path = self.upload_dir / f"{document_id}{suffix}"
        path.write_bytes(content)
        return path

    def register(
        self,
        document_id: str,
        filename: str,
        file_hash: str,
        chunks: list[Chunk],
        page_count: int,
        stored_path: Path,
    ) -> DocumentRecord:
        """Save chunks and add the document to the registry."""
        with self._lock:
            registry = self._read_registry()
            record = DocumentRecord(
                document_id=document_id,
                filename=filename,
                file_type=Path(filename).suffix.lower().lstrip("."),
                file_hash=file_hash,
                uploaded_at=datetime.now(timezone.utc).isoformat(),
                chunk_count=len(chunks),
                page_count=page_count,
                source=stored_path.name,
            )
            (self.chunks_dir / f"{document_id}.json").write_text(
                json.dumps([c.to_dict() for c in chunks], indent=2),
                encoding="utf-8",
            )
            registry[document_id] = asdict(record)
            self._write_registry(registry)
            return record

    def delete_document(self, document_id: str) -> bool:
        """Remove a document, its chunks, and its stored file."""
        with self._lock:
            registry = self._read_registry()
            record = registry.pop(document_id, None)
            if record is None:
                return False

            self._write_registry(registry)
            (self.chunks_dir / f"{document_id}.json").unlink(missing_ok=True)
            stored = self.upload_dir / record["source"]
            stored.unlink(missing_ok=True)
            return True