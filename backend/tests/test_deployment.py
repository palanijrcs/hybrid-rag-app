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


# ---------------------------------------------------------------- Phase 22: VPS
def test_compose_ports_can_be_bound_to_localhost():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "${BACKEND_BIND:-8000}:8000" in text and "${FRONTEND_BIND:-8501}:8501" in text


def test_nginx_site_is_protected_and_supports_streamlit():
    conf = (ROOT / "deploy" / "nginx" / "rag.palanitech.online.conf").read_text(encoding="utf-8")
    assert "server_name rag.palanitech.online;" in conf
    assert "auth_basic_user_file" in conf                 # password in front of the app
    assert "proxy_pass http://127.0.0.1:8601;" in conf    # UI only
    assert "8100" not in conf.split("location", 1)[1]     # API is not exposed
    assert "proxy_set_header Upgrade" in conf             # Streamlit WebSockets


def test_deploy_script_uses_unix_line_endings():
    data = (ROOT / "deploy" / "deploy.sh").read_bytes()
    assert data.startswith(b"#!/usr/bin/env bash") and b"\r\n" not in data
