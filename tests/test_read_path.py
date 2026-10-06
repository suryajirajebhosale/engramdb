from datetime import timedelta

from engramdb import (
    GLOBAL,
    ActivationConfig,
    Brain,
    Edge,
    EdgeType,
    InMemoryStore,
    Memory,
    Polarity,
    Trait,
    format_for_prompt,
)
from engramdb.activation import spread

from .conftest import NOW, KeyedEmbedder


def trait(id, summary, context=GLOBAL, polarity=Polarity.LIKE, days_ago=0, **kw):
    return Trait("u1", summary, context=context, polarity=polarity, strength=0.8,
                 last_seen_at=NOW - timedelta(days=days_ago), id=id, **kw)


def seeded_brain(clock):
    store = InMemoryStore()
    store.put_trait(trait("bold", "Loves bold colours", embedding=[1, 0, 0]))
    store.put_trait(trait("muted_work", "Keeps it muted at work", context="work", polarity=Polarity.DISLIKE,
                          embedding=[0, 1, 0]))
    store.put_trait(trait("loafers", "Lives in loafers", context="work", embedding=[0, 0, 1]))
    store.put_trait(trait("ancient", "Liked neon in 2019", days_ago=2000))
    store.put_trait(trait("travel", "Packs light", context="travel"))
    store.add_edge(Edge("muted_work", "bold", EdgeType.CONTRADICTS))
    store.put_memory(Memory("u1", "Board meeting on Friday", context="work", created_at=NOW, id="m1"))
    return Brain(store=store, embedder=KeyedEmbedder({"what for friday?": [0, 0, 1]}), clock=clock)


def test_activation_lights_context_and_global_but_not_stale_or_other_context(clock):
    lit = {n.id: n for n in seeded_brain(clock).activate("u1", context="work")}
    assert {"muted_work", "loafers", "m1", "bold"} <= set(lit)
    assert "ancient" not in lit  # decayed below threshold
    assert "travel" not in lit  # different context, not connected
    assert lit["muted_work"].score > lit["bold"].score  # context seed > global seed


def test_contradiction_is_carried_into_prompt_block(clock):
    brain = seeded_brain(clock)
    lit = brain.activate("u1", context="work")
    bold = next(n for n in lit if n.id == "bold")
    assert bold.tensions == ["muted_work"]
    assert "in tension with: Keeps it muted at work" in format_for_prompt(lit)


def test_message_knn_seeds_full_activation(clock):
    lit = seeded_brain(clock).activate("u1", message="what for friday?")
    assert lit[0].id == "loafers"
    assert lit[0].activation == 1.0


def test_spread_uses_max_not_sum_and_respects_hops():
    edges = [Edge("a", "b", EdgeType.SUPPORTS), Edge("c", "b", EdgeType.SUPPORTS), Edge("b", "d", EdgeType.SUPPORTS)]
    one_hop, _ = spread({"a": 1.0, "c": 1.0}, edges, ActivationConfig(hops=1))
    assert one_hop["b"] == 0.6 and "d" not in one_hop
    two_hop, _ = spread({"a": 1.0}, edges, ActivationConfig(hops=2))
    assert round(two_hop["d"], 2) == 0.36


def test_preferences_put_stated_before_inferred(clock):
    brain = seeded_brain(clock)
    stated = trait("stated", "Said: no suede", context="work", polarity=Polarity.AVOID, explicit=True)
    stated.strength = 0.3
    brain.store.put_trait(stated)
    rows = brain.preferences("u1", context="work")
    assert rows[0][0].id == "stated"
    assert {t.id for t, _ in rows} >= {"bold", "muted_work", "loafers"}
    assert "travel" not in {t.id for t, _ in rows}


def test_digest_facets_and_known_unknowns(clock):
    store = InMemoryStore()
    store.put_trait(trait("a", "x", context="dining", facets={"cuisine": ["Thai", "korean"]}))
    store.put_trait(trait("b", "y", context="dining", facets={"cuisine": ["thai"]}))
    store.put_trait(trait("c", "z", context="dining", polarity=Polarity.AVOID, facets={"ingredient": ["peanuts"]}))
    store.put_trait(trait("d", "w", context="dining", facets={"cuisine": ["french"]}, archived=True))
    digest = Brain(store=store, clock=clock).digest("u1", dimensions=["cuisine", "ingredient", "budget"])
    assert digest.facets["dining"]["like"]["cuisine"] == ["thai", "korean"]
    assert digest.facets["dining"]["avoid"]["ingredient"] == ["peanuts"]
    assert digest.known_unknowns == [("dining", "budget")]
    assert digest.trait_count == 3
    per_context = Brain(store=store, clock=clock).digest("u1", dimensions={"dining": ["budget"], "travel": ["seat"]})
    assert per_context.known_unknowns == [("dining", "budget")]  # travel has no traits yet


def test_rerank_boosts_likes_and_buries_avoids(clock):
    store = InMemoryStore()
    store.put_trait(trait("a", "x", context="dining", facets={"cuisine": ["thai"]}))
    store.put_trait(trait("b", "y", polarity=Polarity.AVOID, facets={"ingredient": ["peanut"]}))
    brain = Brain(store=store, clock=clock)
    candidates = [("Pad thai with peanut sauce", 0.9), ("Green thai curry", 0.7), ("Burger", 0.8)]
    ranked = brain.rerank("u1", candidates, text_of=lambda s: s, context="dining")
    assert [r.item for r in ranked] == ["Green thai curry", "Burger", "Pad thai with peanut sauce"]
    assert ranked[-1].matched == {"like": ["thai"], "avoid": ["peanut"]}


def test_sweep_archives_faded_inferred_traits_but_not_stated_ones(clock):
    store = InMemoryStore()
    store.put_trait(trait("faded", "x", days_ago=400))
    store.put_trait(trait("faded_stated", "y", days_ago=400, explicit=True))
    store.put_trait(trait("dormant", "z", days_ago=800, archived=True))
    store.put_memory(Memory("u1", "old", created_at=NOW - timedelta(days=365), id="old_mem"))
    report = Brain(store=store, clock=clock).sweep("u1")
    assert report.archived == ["faded"]
    assert report.deleted_traits == ["dormant"]
    assert report.deleted_memories == ["old_mem"]
    assert store.get_trait("faded_stated").archived is False


def test_store_json_roundtrip(tmp_path, clock):
    brain = seeded_brain(clock)
    path = tmp_path / "brain.json"
    brain.store.save(path)
    loaded = InMemoryStore.load(path)
    assert {t.id for t in loaded.traits("u1")} == {t.id for t in brain.store.traits("u1")}
    assert loaded.get_trait("muted_work").polarity is Polarity.DISLIKE
    assert loaded.edges(["bold"]) == brain.store.edges(["bold"])
    assert loaded.get_memory("m1").created_at == NOW
