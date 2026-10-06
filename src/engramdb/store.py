"""Storage interface and an in-memory implementation.

The brain's logic (consolidation, activation, decay, digest) never talks to a
database directly. Everything goes through :class:`BrainStore`, so swapping
the in-memory store for Neo4j, Postgres + pgvector or Redis means
implementing these methods, nothing else.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Protocol

from .embeddings import cosine
from .models import Edge, EdgeType, Memory, Polarity, Trait


class BrainStore(Protocol):
    # traits
    def put_trait(self, trait: Trait) -> None: ...
    def get_trait(self, trait_id: str) -> Trait | None: ...
    def delete_trait(self, trait_id: str) -> None: ...
    def traits(self, user_id: str, *, include_archived: bool = False) -> list[Trait]: ...
    def nearest_traits(
        self, user_id: str, vector: list[float], *, k: int, min_similarity: float
    ) -> list[tuple[Trait, float]]: ...

    # memories
    def put_memory(self, memory: Memory) -> None: ...
    def get_memory(self, memory_id: str) -> Memory | None: ...
    def delete_memory(self, memory_id: str) -> None: ...
    def memories(self, user_id: str) -> list[Memory]: ...
    def nearest_memories(
        self, user_id: str, vector: list[float], *, k: int, min_similarity: float
    ) -> list[tuple[Memory, float]]: ...

    # edges
    def add_edge(self, edge: Edge) -> None: ...
    def edges(self, node_ids: Iterable[str]) -> list[Edge]: ...

    # idempotency (queues deliver at least once)
    def mark_processed(self, item_id: str) -> None: ...
    def is_processed(self, item_id: str) -> bool: ...


class InMemoryStore:
    """Dict-backed store with brute-force cosine kNN. Can save to / load from JSON."""

    def __init__(self) -> None:
        self._traits: dict[str, Trait] = {}
        self._memories: dict[str, Memory] = {}
        self._edges: set[Edge] = set()
        self._processed: set[str] = set()

    # traits ─────────────────────────────────────────────────────────────────

    def put_trait(self, trait: Trait) -> None:
        self._traits[trait.id] = trait

    def get_trait(self, trait_id: str) -> Trait | None:
        return self._traits.get(trait_id)

    def delete_trait(self, trait_id: str) -> None:
        self._traits.pop(trait_id, None)
        self._edges = {e for e in self._edges if trait_id not in (e.source, e.target)}

    def traits(self, user_id: str, *, include_archived: bool = False) -> list[Trait]:
        return [
            t for t in self._traits.values() if t.user_id == user_id and (include_archived or not t.archived)
        ]

    def nearest_traits(self, user_id, vector, *, k, min_similarity):
        return _nearest(
            ((t, t.embedding) for t in self.traits(user_id) if t.embedding), vector, k, min_similarity
        )

    # memories ───────────────────────────────────────────────────────────────

    def put_memory(self, memory: Memory) -> None:
        self._memories[memory.id] = memory

    def get_memory(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    def delete_memory(self, memory_id: str) -> None:
        self._memories.pop(memory_id, None)
        self._edges = {e for e in self._edges if memory_id not in (e.source, e.target)}

    def memories(self, user_id: str) -> list[Memory]:
        return [m for m in self._memories.values() if m.user_id == user_id]

    def nearest_memories(self, user_id, vector, *, k, min_similarity):
        return _nearest(
            ((m, m.embedding) for m in self.memories(user_id) if m.embedding), vector, k, min_similarity
        )

    # edges / idempotency ────────────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        self._edges.add(edge)

    def edges(self, node_ids: Iterable[str]) -> list[Edge]:
        ids = set(node_ids)
        return [e for e in self._edges if e.source in ids or e.target in ids]

    def mark_processed(self, item_id: str) -> None:
        self._processed.add(item_id)

    def is_processed(self, item_id: str) -> bool:
        return item_id in self._processed

    # persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        data = {
            "traits": [asdict(t) for t in self._traits.values()],
            "memories": [asdict(m) for m in self._memories.values()],
            "edges": [asdict(e) for e in self._edges],
            "processed": sorted(self._processed),
        }
        Path(path).write_text(json.dumps(data, default=_json_default, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryStore":
        data = json.loads(Path(path).read_text())
        store = cls()
        for raw in data["traits"]:
            store.put_trait(_trait_from_json(raw))
        for raw in data["memories"]:
            store.put_memory(_memory_from_json(raw))
        for raw in data["edges"]:
            store.add_edge(Edge(raw["source"], raw["target"], EdgeType(raw["type"])))
        store._processed = set(data["processed"])
        return store


def _nearest(items, vector, k, min_similarity):
    scored = [(item, cosine(vector, emb)) for item, emb in items]
    scored = [(item, sim) for item, sim in scored if sim >= min_similarity]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Polarity, EdgeType)):
        return value.value
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _trait_from_json(raw: dict) -> Trait:
    raw = {k: v for k, v in raw.items() if k in {f.name for f in fields(Trait)}}
    raw["polarity"] = Polarity(raw["polarity"])
    for key in ("first_seen_at", "last_seen_at", "archived_at"):
        if raw.get(key):
            raw[key] = datetime.fromisoformat(raw[key])
    return Trait(**raw)


def _memory_from_json(raw: dict) -> Memory:
    raw = {k: v for k, v in raw.items() if k in {f.name for f in fields(Memory)}}
    raw["created_at"] = datetime.fromisoformat(raw["created_at"])
    if raw.get("event_date"):
        raw["event_date"] = date.fromisoformat(raw["event_date"])
    return Memory(**raw)
