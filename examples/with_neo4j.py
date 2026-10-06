"""Persist the brain in Neo4j. No LLM keys needed (offline embedder + heuristic judge).

    docker compose up -d neo4j          # or point NEO4J_* at your own instance
    pip install -e ".[neo4j]"
    cp .env.example .env                # set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
    python examples/with_neo4j.py

Then open http://localhost:7474 and run:
    MATCH (u:User {user_id: 'demo-user'})-[*1..2]-(n) RETURN u, n
"""

import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from engramdb import Brain, HashingEmbedder, HeuristicJudge, Observation, Polarity
from engramdb.neo4j_store import Neo4jStore

missing = [v for v in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD") if not os.getenv(v)]
if missing:
    sys.exit(f"Set {', '.join(missing)} (see .env.example).")

embedder = HashingEmbedder(dim=256)
store = Neo4jStore.connect(
    os.environ["NEO4J_URI"],
    os.environ["NEO4J_USER"],
    os.environ["NEO4J_PASSWORD"],
    database=os.getenv("NEO4J_DATABASE") or None,
)
store.setup(embedding_dim=embedder.dim)  # constraints + vector indexes, idempotent

brain = Brain(store=store, embedder=embedder, judge=HeuristicJudge(merge_similarity=0.6))
U = "demo-user"
brain.forget(U)  # start the demo from a clean slate

for text, context, polarity in [
    ("Prefers aisle seats on flights", "flights", Polarity.LIKE),
    ("Avoids red-eye flights", "flights", Polarity.AVOID),
    ("Never books red-eye flights", "flights", Polarity.AVOID),
    ("Takes overnight trains when possible", "trains", Polarity.LIKE),
]:
    result = brain.observe(Observation(U, text, context=context, polarity=polarity))
    print(f"{result.action:>6}  {text}")

print("\nStored in Neo4j:", [t.summary for t in store.traits(U)])
print("\nPrompt block for a flight request:")
print(brain.prompt_block(U, context="flights", message="book me a flight"))
store.close()
