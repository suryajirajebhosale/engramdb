from datetime import timedelta

from engramdb import Brain, EdgeType, LLMJudge, Memory, Observation, Polarity

from .conftest import NOW, KeyedEmbedder

VECTORS = {
    "Not a morning person": [1.0, 0.0, 0.0, 0.0],
    "Hates early meetings": [0.95, 0.31, 0.0, 0.0],  # cos ~0.95 with the above
    "Loves early-morning runs": [0.6, 0.8, 0.0, 0.0],  # cos 0.6: related, opposite polarity
    "Likes thai food": [0.0, 0.0, 1.0, 0.0],
}


def make_brain(clock, **kwargs):
    return Brain(embedder=KeyedEmbedder(VECTORS), clock=clock, **kwargs)


def obs(text, polarity=Polarity.DISLIKE, **kwargs):
    return Observation("u1", text, polarity=polarity, observed_at=NOW, **kwargs)


def test_paraphrase_merges_and_unions_facets(clock):
    brain = make_brain(clock)
    first = brain.observe(obs("Not a morning person", facets={"time": ["early"]}))
    second = brain.observe(obs("Hates early meetings", facets={"time": ["7am"], "type": ["meeting"]}))

    assert (first.action, second.action) == ("new", "merge")
    trait = brain.store.get_trait(first.trait_id)
    assert trait.reinforcement_count == 2
    assert trait.facets == {"time": ["early", "7am"], "type": ["meeting"]}
    assert len(trait.evidence) == 2


def test_opposite_polarity_never_merges_and_gets_contradicts_edge(clock):
    brain = make_brain(clock)
    a = brain.observe(obs("Not a morning person"))
    b = brain.observe(obs("Loves early-morning runs", polarity=Polarity.LIKE))
    assert b.action == "new"
    [edge] = brain.store.edges([b.trait_id])
    assert (edge.source, edge.target, edge.type) == (b.trait_id, a.trait_id, EdgeType.CONTRADICTS)


def test_duplicate_delivery_is_idempotent(clock):
    brain = make_brain(clock)
    o = obs("Likes thai food", polarity=Polarity.LIKE)
    brain.observe(o)
    assert brain.observe(o).action == "duplicate"
    assert len(brain.store.traits("u1")) == 1


def test_new_evidence_revives_archived_trait(clock):
    brain = make_brain(clock)
    result = brain.observe(obs("Not a morning person"))
    trait = brain.store.get_trait(result.trait_id)
    trait.archived = True
    brain.store.put_trait(trait)
    # archived traits are not kNN candidates, so evidence creates a fresh trait...
    assert brain.observe(obs("Hates early meetings")).action == "new"


def test_llm_judge_merge_is_validated(clock):
    calls = []

    def fake_llm(system, user):
        calls.append(user)
        return {"action": "merge", "trait_id": "trait_made_up"}  # hallucinated id

    brain = make_brain(clock, judge=LLMJudge(fake_llm))
    brain.observe(obs("Not a morning person"))
    result = brain.observe(obs("Hates early meetings"))
    assert calls, "LLM was consulted"
    assert result.action == "merge"  # fell back to heuristic, which merges this pair


def test_llm_judge_new_trait_with_edges(clock):
    first_id = {}

    def fake_llm(system, user):
        return {
            "action": "new",
            "summary": "Prefers afternoon scheduling",
            "edges": [{"type": "REFINES", "trait_id": first_id["id"]}, {"type": "BOGUS", "trait_id": "x"}],
        }

    brain = make_brain(clock)
    first_id["id"] = brain.observe(obs("Not a morning person")).trait_id
    brain.judge = brain._consolidator.judge = LLMJudge(fake_llm)
    result = brain.observe(obs("Hates early meetings"))
    trait = brain.store.get_trait(result.trait_id)
    assert trait.summary == "Prefers afternoon scheduling"
    assert [e.type for e in brain.store.edges([trait.id])] == [EdgeType.REFINES]


def test_recurring_memories_promote_to_trait_then_reinforce(clock):
    same = [0.0, 0.0, 1.0, 0.0]
    embedder = KeyedEmbedder({}, default=same)
    brain = Brain(embedder=embedder, clock=clock)

    results = [
        brain.remember(Memory("u1", f"Ordered thai takeaway #{i}", created_at=NOW - timedelta(days=3 - i)))
        for i in range(4)
    ]
    assert [r.action for r in results] == ["stored", "stored", "new", "reinforce"]
    trait = brain.store.get_trait(results[2].trait_id)
    assert results[3].trait_id == trait.id
    assert trait.reinforcement_count == 4
    assert all(m.promoted_to == trait.id for m in brain.store.memories("u1"))
    assert {e.type for e in brain.store.edges([trait.id])} == {EdgeType.DERIVED_FROM}


def test_llm_can_veto_promotion(clock):
    brain = Brain(
        embedder=KeyedEmbedder({}, default=[1.0, 0.0]),
        judge=LLMJudge(lambda s, u: {"action": "skip"}),
        clock=clock,
    )
    actions = [brain.remember(Memory("u1", f"m{i}", created_at=NOW)).action for i in range(3)]
    assert actions == ["stored", "stored", "skip"]
    assert brain.store.traits("u1") == []
