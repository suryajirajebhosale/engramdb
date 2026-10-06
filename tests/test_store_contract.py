"""One behavioural contract, run against every BrainStore implementation.

Neo4j cases run only when a test database is configured:

    docker compose up -d neo4j
    export ENGRAM_TEST_NEO4J_URI=bolt://localhost:7687
    export ENGRAM_TEST_NEO4J_USER=neo4j ENGRAM_TEST_NEO4J_PASSWORD=engramdb-test
    pytest

WARNING: the Neo4j cases delete all engramdb nodes and all ``User`` nodes in
the target database between tests. Point them at a throwaway database only.
"""

import math
import os
from datetime import date, timedelta

import pytest

from engramdb import Brain, Edge, EdgeType, InMemoryStore, Memory, Observation, Polarity, Trait

from .conftest import NOW, KeyedEmbedder

DIM = 4
NEO4J_URI = os.getenv("ENGRAM_TEST_NEO4J_URI")


@pytest.fixture(scope="session")
def neo4j_driver():
    if not NEO4J_URI:
        pytest.skip("set ENGRAM_TEST_NEO4J_URI to run Neo4j store tests")
    import neo4j

    driver = neo4j.GraphDatabase.driver(
        NEO4J_URI,
        auth=(os.getenv("ENGRAM_TEST_NEO4J_USER", "neo4j"), os.getenv("ENGRAM_TEST_NEO4J_PASSWORD", "")),
    )
    driver.verify_connectivity()
    from engramdb.neo4j_store import Neo4jStore

    Neo4jStore(driver).setup(embedding_dim=DIM)
    yield driver
    driver.close()


@pytest.fixture(params=["memory", "neo4j-exact", "neo4j-index"])
def store(request):
    if request.param == "memory":
        return InMemoryStore()
    driver = request.getfixturevalue("neo4j_driver")
    driver.execute_query("MATCH (n) WHERE n:EngramNode OR n:EngramProcessed OR n:User DETACH DELETE n")
    from engramdb.neo4j_store import Neo4jStore

    return Neo4jStore(driver, knn_mode=request.param.split("-")[1])


def unit(*xs):
    norm = math.sqrt(sum(x * x for x in xs))
    return [x / norm for x in xs]


def make_trait(id, user="u1", embedding=None, **kw):
    defaults = dict(summary=f"trait {id}", strength=0.7, last_seen_at=NOW, first_seen_at=NOW - timedelta(days=3))
    defaults.update(kw)
    return Trait(user, id=id, embedding=embedding, **defaults)


# ── round-trips ─────────────────────────────────────────────────────────────


def test_trait_roundtrip_preserves_every_field(store):
    original = make_trait(
        "t1", context="work", polarity=Polarity.AVOID, facets={"colour": ["red", "orange"], "fit": ["slim"]},
        reinforcement_count=3, explicit=True, embedding=unit(1, 2, 3, 4), evidence=["obs_a", "obs_b"],
    )
    store.put_trait(original)
    loaded = store.get_trait("t1")
    assert loaded.embedding == pytest.approx(original.embedding)
    loaded.embedding = original.embedding
    assert loaded == original


def test_trait_update_overwrites_and_clears_fields(store):
    store.put_trait(make_trait("t1", archived=True, archived_at=NOW))
    store.put_trait(make_trait("t1", summary="revised", archived=False, archived_at=None))
    loaded = store.get_trait("t1")
    assert (loaded.summary, loaded.archived, loaded.archived_at) == ("revised", False, None)


def test_memory_roundtrip_with_event_date(store):
    original = Memory("u1", "Trip to Lisbon", context="travel", event_date=date(2026, 6, 14),
                      created_at=NOW, embedding=unit(0, 1, 0, 0), promoted_to=None, id="m1")
    store.put_memory(original)
    assert store.get_memory("m1") == original


def test_missing_ids_return_none(store):
    assert store.get_trait("nope") is None
    assert store.get_memory("nope") is None


# ── per-user isolation and filtering ───────────────────────────────────────


def test_users_are_isolated_and_archived_traits_hidden_by_default(store):
    store.put_trait(make_trait("a", user="u1"))
    store.put_trait(make_trait("b", user="u1", archived=True, archived_at=NOW))
    store.put_trait(make_trait("c", user="u2"))
    store.put_memory(Memory("u2", "theirs", created_at=NOW, id="m2"))
    assert {t.id for t in store.traits("u1")} == {"a"}
    assert {t.id for t in store.traits("u1", include_archived=True)} == {"a", "b"}
    assert {t.id for t in store.traits("u2")} == {"c"}
    assert store.memories("u1") == []


# ── similarity search ───────────────────────────────────────────────────────


def test_nearest_traits_returns_plain_cosine_scoped_to_user(store):
    store.put_trait(make_trait("same", embedding=unit(1, 0, 0, 0)))
    store.put_trait(make_trait("close", embedding=unit(1, 1, 0, 0)))  # cos 0.707
    store.put_trait(make_trait("far", embedding=unit(0, 0, 1, 0)))  # cos 0
    store.put_trait(make_trait("archived", embedding=unit(1, 0, 0, 0), archived=True))
    store.put_trait(make_trait("other_user", user="u2", embedding=unit(1, 0, 0, 0)))

    hits = store.nearest_traits("u1", unit(1, 0, 0, 0), k=5, min_similarity=0.5)
    assert [t.id for t, _ in hits] == ["same", "close"]
    assert [s for _, s in hits] == pytest.approx([1.0, math.sqrt(0.5)], abs=1e-4)

    assert [t.id for t, _ in store.nearest_traits("u1", unit(1, 0, 0, 0), k=1, min_similarity=0)] == ["same"]


def test_nearest_memories(store):
    store.put_memory(Memory("u1", "a", created_at=NOW, embedding=unit(0, 1, 0, 0), id="m_a"))
    store.put_memory(Memory("u1", "b", created_at=NOW, embedding=unit(0, 0, 0, 1), id="m_b"))
    store.put_memory(Memory("u1", "no vector yet", created_at=NOW, id="m_c"))
    hits = store.nearest_memories("u1", unit(0, 1, 0, 0), k=5, min_similarity=0.9)
    assert [m.id for m, _ in hits] == ["m_a"]


# ── edges, idempotency, deletion ───────────────────────────────────────────


def test_edges_are_deduplicated_directional_and_removed_with_their_node(store):
    for tid in ("a", "b", "c"):
        store.put_trait(make_trait(tid))
    store.put_memory(Memory("u1", "m", created_at=NOW, id="m1"))
    store.add_edge(Edge("a", "b", EdgeType.CONTRADICTS))
    store.add_edge(Edge("a", "b", EdgeType.CONTRADICTS))  # duplicate
    store.add_edge(Edge("c", "a", EdgeType.REFINES))
    store.add_edge(Edge("m1", "a", EdgeType.DERIVED_FROM))

    assert set(store.edges(["a"])) == {
        Edge("a", "b", EdgeType.CONTRADICTS),
        Edge("c", "a", EdgeType.REFINES),
        Edge("m1", "a", EdgeType.DERIVED_FROM),
    }
    assert len(store.edges(["a", "b"])) == 3
    store.delete_trait("a")
    assert store.edges(["b", "c", "m1"]) == []


def test_processed_markers(store):
    assert not store.is_processed("obs_1")
    store.mark_processed("obs_1")
    store.mark_processed("obs_1")
    assert store.is_processed("obs_1")


def test_delete_user_erases_only_that_user(store):
    store.put_trait(make_trait("mine", evidence=["obs_mine"]))
    store.put_memory(Memory("u1", "m", created_at=NOW, id="m_mine"))
    store.put_trait(make_trait("theirs", user="u2", evidence=["obs_theirs"]))
    store.add_edge(Edge("m_mine", "mine", EdgeType.DERIVED_FROM))
    for item in ("obs_mine", "m_mine", "obs_theirs"):
        store.mark_processed(item)

    store.delete_user("u1")
    assert store.traits("u1", include_archived=True) == [] and store.memories("u1") == []
    assert not store.is_processed("obs_mine") and not store.is_processed("m_mine")
    assert [t.id for t in store.traits("u2")] == ["theirs"]
    assert store.is_processed("obs_theirs")


# ── the whole Brain on top of each store ───────────────────────────────────


def test_brain_end_to_end(store, clock):
    vectors = {
        "Not a morning person": unit(1, 0, 0, 0),
        "Hates early meetings": unit(0.95, 0.31, 0, 0),
        "Loves early-morning runs": unit(0.6, 0.8, 0, 0),
        "what time for the standup?": unit(1, 0.1, 0, 0),
    }
    brain = Brain(store=store, embedder=KeyedEmbedder(vectors, default=unit(0, 0, 1, 0)), clock=clock)

    def observe(text, polarity):
        return brain.observe(Observation("u1", text, polarity=polarity, observed_at=NOW))

    first = observe("Not a morning person", Polarity.DISLIKE)
    assert observe("Hates early meetings", Polarity.DISLIKE).action == "merge"
    runs = observe("Loves early-morning runs", Polarity.LIKE)
    assert store.get_trait(first.trait_id).reinforcement_count == 2

    lit = brain.activate("u1", message="what time for the standup?")
    by_id = {n.id: n for n in lit}
    assert lit[0].id == first.trait_id
    assert by_id[runs.trait_id].tensions == [first.trait_id]

    brain.forget("u1")
    assert brain.activate("u1") == []
