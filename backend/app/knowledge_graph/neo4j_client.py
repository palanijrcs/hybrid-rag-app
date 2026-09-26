"""Connection to Neo4j AuraDB."""

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

logger = logging.getLogger(__name__)


class GraphConnectionError(Exception):
    """Raised when the graph database cannot be reached."""


class Neo4jClient:
    """A thin wrapper around the Neo4j driver, opened once and reused."""

    def __init__(
        self, uri: str, username: str, password: str, database: str = "neo4j"
    ) -> None:
        self.uri = uri
        self.database = database
        self._driver: Driver | None = None

        if not (uri and username and password):
            logger.warning("Neo4j is not configured; graph features are disabled.")
            return

        try:
            self._driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                # AuraDB closes idle connections; test a pooled connection that has been
                # idle for 30s before reusing it, and replace connections every 30 min.
                liveness_check_timeout=30,
                max_connection_lifetime=1800,
                keep_alive=True,
            )
        except Exception as error:
            # Never include the password in a log or an error message
            logger.error("Could not create the Neo4j driver: %s", type(error).__name__)
            self._driver = None

    @property
    def is_configured(self) -> bool:
        return self._driver is not None

    def verify(self) -> bool:
        """Check the database is reachable. Returns False instead of raising."""
        if self._driver is None:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except (ServiceUnavailable, AuthError, Neo4jError) as error:
            logger.error("Neo4j is unreachable: %s", type(error).__name__)
            return False

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Open a session for a unit of work."""
        if self._driver is None:
            raise GraphConnectionError("Neo4j is not configured.")
        session = self._driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    def run(self, query: str, **parameters: Any) -> list[dict]:
        """Run a Cypher query and return the rows as plain dictionaries."""
        with self.session() as session:
            result = session.run(query, **parameters)
            return [record.data() for record in result]

    def ensure_schema(self) -> None:
        """Create the constraints and indexes the graph relies on."""
        statements = [
            # One node per entity name+type, so repeated mentions merge
            "CREATE CONSTRAINT entity_key IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
            # One node per source chunk, for grounding and citations
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            # Fast lookup when deleting a document's nodes
            "CREATE INDEX chunk_document IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.document_id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        ]
        for statement in statements:
            self.run(statement)
        logger.info("Neo4j schema is ready.")

    def stats(self) -> dict:
        """Count what is currently in the graph."""
        rows = self.run(
            "MATCH (e:Entity) WITH count(e) AS entities "
            "MATCH (c:Chunk) WITH entities, count(c) AS chunks "
            "RETURN entities, chunks"
        )
        if not rows:
            return {"entities": 0, "chunks": 0, "relationships": 0}
        relationships = self.run(
            "MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS relationships"
        )
        return {
            "entities": rows[0]["entities"],
            "chunks": rows[0]["chunks"],
            "relationships": relationships[0]["relationships"] if relationships else 0,
        }

    def clear_all(self) -> int:
        """Delete every node this app created. Used in tests and resets."""
        rows = self.run(
            "MATCH (n) WHERE n:Entity OR n:Chunk "
            "WITH n, count(n) AS ignored DETACH DELETE n RETURN count(*) AS deleted"
        )
        return rows[0]["deleted"] if rows else 0

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None