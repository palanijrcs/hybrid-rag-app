"""Tests for the Neo4j client.

These run against the real AuraDB instance when it is configured, and are
skipped otherwise, so the suite still passes without a database.
"""

import pytest

from app.core.config import get_settings
from app.knowledge_graph.neo4j_client import GraphConnectionError, Neo4jClient


def test_unconfigured_client_is_safe():
    """No credentials must mean a disabled feature, not a crash."""
    client = Neo4jClient(uri="", username="", password="")
    assert client.is_configured is False
    assert client.verify() is False
    with pytest.raises(GraphConnectionError):
        with client.session():
            pass


def test_bad_credentials_do_not_crash():
    client = Neo4jClient(
        uri="neo4j+s://nonexistent.databases.neo4j.io",
        username="neo4j",
        password="wrong",
    )
    assert client.verify() is False


@pytest.fixture
def live_client():
    settings = get_settings()
    if not settings.neo4j_uri:
        pytest.skip("Neo4j is not configured.")
    client = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
    if not client.verify():
        pytest.skip("Neo4j is unreachable.")
    yield client
    client.close()


def test_live_connection_works(live_client):
    assert live_client.verify() is True


def test_can_run_a_query(live_client):
    rows = live_client.run("RETURN 1 + 1 AS total")
    assert rows[0]["total"] == 2


def test_schema_can_be_created(live_client):
    live_client.ensure_schema()
    constraints = live_client.run("SHOW CONSTRAINTS")
    names = {row.get("name") for row in constraints}
    assert "entity_key" in names
    assert "chunk_id" in names


def test_stats_returns_counts(live_client):
    live_client.ensure_schema()
    stats = live_client.stats()
    assert set(stats) == {"entities", "chunks", "relationships"}