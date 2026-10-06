"""Neo4j implementation of :class:`~engramdb.store.BrainStore`.

Graph layout (one sub-graph per user)::

    (:User {user_id})-[:HAS_TRAIT]->(:Trait:EngramNode {id, summary, ...})
    (:User {user_id})-[:HAS_MEMORY]->(:Memory:EngramNode {id, text, ...})
    (:Trait)-[:SUPPORTS|CONTRADICTS|REFINES]->(:Trait)
    (:Memory)-[:DERIVED_FROM]->(:Trait)
    (:EngramProcessed {id})            idempotency markers for queue consumers

Every read starts from the ``User`` node, so one user's brain is never visible
from another's. The user label and key are configurable, so the brain can
attach to ``User`` nodes your application already has.

Similarity search has two modes:

- ``"exact"`` (default) scores the user's own nodes with
  ``vector.similarity.cosine``. Per-user graphs are small (hundreds of
  nodes), so this is fast, exact and needs no vector index. Neo4j 5.18+.
- ``"index"`` uses the native vector index. That index spans *all* users, so
  results are over-fetched and then filtered to the user. Use it when a
  single user has many thousands of nodes. Neo4j 5.13+.

Neo4j reports cosine similarity normalised to [0, 1] as ``(1 + cos) / 2``.
This store converts it back to plain cosine, so thresholds mean the same
thing as with :class:`~engramdb.store.InMemoryStore`.

Requires the optional driver: ``pip install "engramdb[neo4j]"``.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any, Iterable, Literal

from .models import Edge, EdgeType, Memory, Polarity, Trait

try:
    import neo4j
    from neo4j.exceptions import ClientError
except ImportError as exc:  # pragma: no cover - depends on the extra
    raise ImportError('Neo4jStore needs the neo4j driver: pip install "engramdb[neo4j]"') from exc

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EDGE_TYPES = "|".join(e.value for e in EdgeType)
_TRAIT_INDEX = "engram_trait_embedding"
_MEMORY_INDEX = "engram_memory_embedding"


class Neo4jStore:
    def __init__(
        self,
        driver: "neo4j.Driver",
        *,
        database: str | None = None,
        user_label: str = "User",
        user_key: str = "user_id",
        knn_mode: Literal["exact", "index"] = "exact",
        index_oversample: int = 20,
    ):
        for name in (user_label, user_key):
            if not _IDENTIFIER.match(name):
                raise ValueError(f"not a valid Neo4j identifier: {name!r}")
        if knn_mode not in ("exact", "index"):
            raise ValueError("knn_mode must be 'exact' or 'index'")
        self.driver = driver
        self.database = database
        self.knn_mode = knn_mode
        self.index_oversample = index_oversample
        # Labels and property keys cannot be query parameters; they are
        # validated identifiers above, so interpolating them is safe.
        self._user = f"(u:{user_label} {{{user_key}: $user_id}})"
        self._user_label = user_label
        self._user_key = user_key

    @classmethod
    def connect(cls, uri: str, user: str, password: str, **kwargs: Any) -> "Neo4jStore":
        """Convenience constructor that owns its driver. Call :meth:`close` when done."""
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return cls(driver, **kwargs)

    def close(self) -> None:
        self.driver.close()

    def _run(self, query: str, *, read: bool = False, **params: Any) -> list[neo4j.Record]:
        # Managed transactions retry transient errors (leader switches, deadlocks).
        # Server notifications are off: on a fresh database Neo4j warns about
        # relationship types that simply have no instances yet.
        with self.driver.session(
            database=self.database, notifications_min_severity=neo4j.NotificationMinimumSeverity.OFF
        ) as session:
            work = lambda tx: list(tx.run(query, params))  # noqa: E731
            return session.execute_read(work) if read else session.execute_write(work)

    # ── schema ─────────────────────────────────────────────────────────────

    def setup(self, *, embedding_dim: int | None = None) -> None:
        """Create constraints (and vector indexes when ``embedding_dim`` is given).

        Idempotent; safe to call on every start-up.
        """
        statements = [
            "CREATE CONSTRAINT engram_node_id IF NOT EXISTS FOR (n:EngramNode) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT engram_processed_id IF NOT EXISTS FOR (p:EngramProcessed) REQUIRE p.id IS UNIQUE",
            f"CREATE CONSTRAINT engram_user_key IF NOT EXISTS "
            f"FOR (u:{self._user_label}) REQUIRE u.{self._user_key} IS UNIQUE",
        ]
        if embedding_dim is not None:
            for name, label in ((_TRAIT_INDEX, "Trait"), (_MEMORY_INDEX, "Memory")):
                statements.append(
                    f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON n.embedding "
                    f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(embedding_dim)}, "
                    f"`vector.similarity_function`: 'cosine'}}}}"
                )
        for statement in statements:
            try:
                self._run(statement)
            except ClientError as exc:
                # The app may already have an equivalent constraint under another name.
                if "EquivalentSchemaRuleAlreadyExists" not in (exc.code or ""):
                    raise

    # ── traits ─────────────────────────────────────────────────────────────

    def put_trait(self, trait: Trait) -> None:
        self._run(
            f"""
            MERGE {self._user}
            MERGE (t:EngramNode:Trait {{id: $id}})
            SET t += $props
            MERGE (u)-[:HAS_TRAIT]->(t)
            """,
            user_id=trait.user_id,
            id=trait.id,
            props=_trait_props(trait),
        )

    def get_trait(self, trait_id: str) -> Trait | None:
        records = self._run("MATCH (t:Trait {id: $id}) RETURN t", id=trait_id, read=True)
        return _trait(records[0]["t"]) if records else None

    def delete_trait(self, trait_id: str) -> None:
        self._run("MATCH (t:Trait {id: $id}) DETACH DELETE t", id=trait_id)

    def traits(self, user_id: str, *, include_archived: bool = False) -> list[Trait]:
        records = self._run(
            f"""
            MATCH {self._user}-[:HAS_TRAIT]->(t:Trait)
            WHERE $include_archived OR NOT coalesce(t.archived, false)
            RETURN t
            """,
            user_id=user_id,
            include_archived=include_archived,
            read=True,
        )
        return [_trait(r["t"]) for r in records]

    def nearest_traits(self, user_id, vector, *, k, min_similarity):
        return [
            (_trait(node), score)
            for node, score in self._nearest(
                "HAS_TRAIT", "Trait", _TRAIT_INDEX, user_id, vector, k, min_similarity,
                extra_filter="NOT coalesce(n.archived, false)",
            )
        ]

    # ── memories ───────────────────────────────────────────────────────────

    def put_memory(self, memory: Memory) -> None:
        self._run(
            f"""
            MERGE {self._user}
            MERGE (m:EngramNode:Memory {{id: $id}})
            SET m += $props
            MERGE (u)-[:HAS_MEMORY]->(m)
            """,
            user_id=memory.user_id,
            id=memory.id,
            props=_memory_props(memory),
        )

    def get_memory(self, memory_id: str) -> Memory | None:
        records = self._run("MATCH (m:Memory {id: $id}) RETURN m", id=memory_id, read=True)
        return _memory(records[0]["m"]) if records else None

    def delete_memory(self, memory_id: str) -> None:
        self._run("MATCH (m:Memory {id: $id}) DETACH DELETE m", id=memory_id)

    def memories(self, user_id: str) -> list[Memory]:
        records = self._run(f"MATCH {self._user}-[:HAS_MEMORY]->(m:Memory) RETURN m", user_id=user_id, read=True)
        return [_memory(r["m"]) for r in records]

    def nearest_memories(self, user_id, vector, *, k, min_similarity):
        return [
            (_memory(node), score)
            for node, score in self._nearest(
                "HAS_MEMORY", "Memory", _MEMORY_INDEX, user_id, vector, k, min_similarity
            )
        ]

    # ── edges / idempotency / users ────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        rel = EdgeType(edge.type).value  # validated enum value, safe to interpolate
        self._run(
            f"""
            MATCH (a:EngramNode {{id: $source}}), (b:EngramNode {{id: $target}})
            MERGE (a)-[:{rel}]->(b)
            """,
            source=edge.source,
            target=edge.target,
        )

    def edges(self, node_ids: Iterable[str]) -> list[Edge]:
        records = self._run(
            f"""
            MATCH (n:EngramNode) WHERE n.id IN $ids
            MATCH (n)-[r:{_EDGE_TYPES}]-()
            WITH DISTINCT r
            RETURN startNode(r).id AS source, endNode(r).id AS target, type(r) AS type
            """,
            ids=list(node_ids),
            read=True,
        )
        return [Edge(r["source"], r["target"], EdgeType(r["type"])) for r in records]

    def mark_processed(self, item_id: str) -> None:
        self._run("MERGE (:EngramProcessed {id: $id})", id=item_id)

    def is_processed(self, item_id: str) -> bool:
        return bool(self._run("MATCH (p:EngramProcessed {id: $id}) RETURN 1 LIMIT 1", id=item_id, read=True))

    def delete_user(self, user_id: str) -> None:
        """Erase a user's whole brain (right to be forgotten). The User node itself is kept."""
        self._run(
            f"""
            MATCH {self._user}-[:HAS_TRAIT|HAS_MEMORY]->(n:EngramNode)
            WITH n, n.evidence AS evidence, n.id AS id
            OPTIONAL MATCH (p:EngramProcessed) WHERE p.id = id OR p.id IN coalesce(evidence, [])
            DETACH DELETE n, p
            """,
            user_id=user_id,
        )

    # ── similarity search ──────────────────────────────────────────────────

    def _nearest(self, rel, label, index, user_id, vector, k, min_similarity, extra_filter="true"):
        # Convert the cosine floor into Neo4j's normalised (1 + cos) / 2 scale.
        params = dict(user_id=user_id, vector=list(vector), k=k, floor=(1 + min_similarity) / 2)
        if self.knn_mode == "exact":
            query = f"""
                MATCH {self._user}-[:{rel}]->(n:{label})
                WHERE n.embedding IS NOT NULL AND {extra_filter}
                WITH n, vector.similarity.cosine(n.embedding, $vector) AS score
                WHERE score >= $floor
                RETURN n, score ORDER BY score DESC LIMIT $k
            """
        else:
            params["fetch"] = max(k * self.index_oversample, k)
            query = f"""
                CALL db.index.vector.queryNodes('{index}', $fetch, $vector) YIELD node AS n, score
                WHERE score >= $floor AND {extra_filter}
                  AND EXISTS {{ MATCH {self._user}-[:{rel}]->(n) }}
                RETURN n, score ORDER BY score DESC LIMIT $k
            """
        return [(r["n"], 2 * r["score"] - 1) for r in self._run(query, read=True, **params)]


# ── (de)serialisation ───────────────────────────────────────────────────────
# Neo4j properties must be primitives or lists of primitives, so facets (a
# dict of lists) are stored as JSON. Datetimes go in as native temporal
# values, which keeps them sortable and queryable in Cypher.


def _trait_props(t: Trait) -> dict[str, Any]:
    return {
        "user_id": t.user_id,
        "summary": t.summary,
        "context": t.context,
        "polarity": t.polarity.value,
        "facets_json": json.dumps(t.facets),
        "strength": t.strength,
        "reinforcement_count": t.reinforcement_count,
        "explicit": t.explicit,
        "first_seen_at": t.first_seen_at,
        "last_seen_at": t.last_seen_at,
        "embedding": t.embedding,
        "archived": t.archived,
        "archived_at": t.archived_at,
        "evidence": list(t.evidence),
    }


def _memory_props(m: Memory) -> dict[str, Any]:
    return {
        "user_id": m.user_id,
        "text": m.text,
        "context": m.context,
        "event_date": m.event_date,
        "created_at": m.created_at,
        "embedding": m.embedding,
        "promoted_to": m.promoted_to,
    }


def _native(value: Any) -> Any:
    return value.to_native() if hasattr(value, "to_native") else value


def _trait(node: Any) -> Trait:
    p = dict(node)
    return Trait(
        id=p["id"],
        user_id=p["user_id"],
        summary=p["summary"],
        context=p["context"],
        polarity=Polarity(p["polarity"]),
        facets=json.loads(p.get("facets_json") or "{}"),
        strength=p["strength"],
        reinforcement_count=p["reinforcement_count"],
        explicit=p.get("explicit", False),
        first_seen_at=_as_datetime(p["first_seen_at"]),
        last_seen_at=_as_datetime(p["last_seen_at"]),
        embedding=list(p["embedding"]) if p.get("embedding") is not None else None,
        archived=p.get("archived", False),
        archived_at=_as_datetime(p["archived_at"]) if p.get("archived_at") else None,
        evidence=list(p.get("evidence") or []),
    )


def _memory(node: Any) -> Memory:
    p = dict(node)
    event_date = _native(p.get("event_date"))
    return Memory(
        id=p["id"],
        user_id=p["user_id"],
        text=p["text"],
        context=p["context"],
        event_date=event_date if isinstance(event_date, date) else None,
        created_at=_as_datetime(p["created_at"]),
        embedding=list(p["embedding"]) if p.get("embedding") is not None else None,
        promoted_to=p.get("promoted_to"),
    )


def _as_datetime(value: Any) -> datetime:
    value = _native(value)
    if not isinstance(value, datetime):
        raise TypeError(f"expected a datetime property, got {type(value).__name__}")
    return value
