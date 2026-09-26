"""Phase 21 tests: nothing secret can reach GitHub, and CI never needs secrets."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECRET_KEYS = ("OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")


def test_gitignore_blocks_secrets_and_user_data():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for entry in (".env", "data/uploads/", "data/processed/", "data/faiss/", "venv/"):
        assert entry in lines, f"{entry} missing from .gitignore"
    assert "!.env.example" in lines


def test_env_example_has_no_real_values():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in SECRET_KEYS:
        m = re.search(rf"^{key}=(.*)$", text, re.MULTILINE)
        assert m is not None and m.group(1).strip() == "", f"{key} must be empty in .env.example"


def test_ci_runs_offline_tests_only():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'not slow and not live' in ci
    assert "secrets." not in ci and "OPENAI_API_KEY" not in ci


def test_no_real_looking_keys_in_source():
    pattern = re.compile(r"sk-(proj-)?[A-Za-z0-9_-]{40,}")
    for path in (ROOT / "backend" / "app").rglob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path
