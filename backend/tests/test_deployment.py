"""Phase 20 tests: startup hook and Docker configuration sanity checks."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


def test_lifespan_runs_graph_setup_once(monkeypatch):
    import app.main as main

    calls = []
    monkeypatch.setattr(main, "prepare_graph", lambda: calls.append(1))
    with TestClient(main.app):
        pass
    assert calls == [1]


def test_backend_dockerfile_is_safe():
    text = (ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")
    assert "USER app" in text and "HEALTHCHECK" in text
    assert "--reload" not in text and ".env" not in text
    assert "download.pytorch.org/whl/cpu" in text


def test_env_file_never_copied_into_images():
    for folder in ("backend", "frontend"):
        assert ".env" in (ROOT / folder / ".dockerignore").read_text(encoding="utf-8").split()


def test_compose_wires_frontend_to_backend():
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    backend, frontend = compose["services"]["backend"], compose["services"]["frontend"]
    assert frontend["environment"]["BACKEND_URL"] == "http://backend:8000"
    assert frontend["depends_on"]["backend"]["condition"] == "service_healthy"
    assert "./data:/app/data" in backend["volumes"]
    assert "command" not in backend          # no --reload in production
