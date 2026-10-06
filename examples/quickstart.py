"""Offline demo: a travel assistant that remembers. No API keys needed.

    pip install -e .
    python examples/quickstart.py
"""

from datetime import date, timedelta

from engramdb import Brain, HeuristicJudge, Memory, Observation, Polarity

# In-memory store, offline hashing embedder, heuristic judge. The hashing
# embedder is lexical, so paraphrases score lower than with a semantic model.
# Loosen the merge bar here; keep the default with real embeddings.
brain = Brain(judge=HeuristicJudge(merge_similarity=0.6))
U = "user-42"

# ── write path: what an extractor pulled out of past conversations ──────────
observations = [
    Observation(U, "Prefers aisle seats on flights", context="flights", polarity=Polarity.LIKE,
                facets={"seat": ["aisle"]}, explicit=True, strength=0.9),
    Observation(U, "Avoids red-eye flights", context="flights", polarity=Polarity.AVOID,
                facets={"timing": ["red-eye"]}, strength=0.7),
    Observation(U, "Never books red-eye flights", context="flights", polarity=Polarity.AVOID,
                facets={"timing": ["overnight"]}, strength=0.6),  # paraphrase -> merges
    Observation(U, "Likes boutique hotels", context="hotels", polarity=Polarity.LIKE,
                facets={"style": ["boutique"]}),
    Observation(U, "Travels on a tight budget", polarity=Polarity.LIKE,
                facets={"budget": ["budget", "cheap"]}),
    Observation(U, "Splurges on hotels with rooftop pools", context="hotels", polarity=Polarity.LIKE,
                facets={"amenity": ["rooftop pool"]}),
]
for obs in observations:
    result = brain.observe(obs)
    print(f"{result.action:>6}  {obs.text}")

brain.remember(Memory(U, "Flying to Lisbon for a friend's wedding", context="flights",
                      event_date=date.today() + timedelta(days=10)))

# ── read path ───────────────────────────────────────────────────────────────
print("\nLit for 'find me a flight to Lisbon' (context=flights):")
for node in brain.activate(U, context="flights", message="find me a flight to Lisbon"):
    print(f"  {node.score:.2f}  {node.kind:<6} {node.text}")

print("\nPrompt block:")
print(brain.prompt_block(U, context="flights", message="find me a flight to Lisbon"))

print("\nDigest facets:")
digest = brain.digest(U, dimensions={"flights": ["seat", "timing", "airline"], "hotels": ["style", "area"]})
for context, polarities in digest.facets.items():
    print(f"  {context}: {polarities}")
print("  known unknowns:", digest.known_unknowns)

print("\nRe-ranked flight options:")
flights = [("Overnight red-eye, window seat, $310", 0.92), ("Morning direct, aisle seat, $380", 0.85),
           ("Afternoon, 1 stop, $290", 0.80)]
for r in brain.rerank(U, flights, text_of=lambda s: s, context="flights"):
    print(f"  {r.base_score:.2f} -> {r.score:.2f}  {r.item}   {r.matched or ''}")
