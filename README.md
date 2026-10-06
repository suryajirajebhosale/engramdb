# engramdb

[![CI](https://github.com/suryajirajebhosale/engramdb/actions/workflows/ci.yml/badge.svg)](https://github.com/suryajirajebhosale/engramdb/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Long-term user memory for LLM apps, as a graph that learns, forgets and lights up.**

Most "memory" for LLM apps is a vector store of past messages. That fails in
predictable ways:

- **Duplicates.** "Not a morning person", "hates early meetings" and "avoid 8am calls"
  become three memories competing for prompt space.
- **No sense of time.** A preference from two years ago carries the same weight as
  one from yesterday.
- **No structure.** It can't tell that "loves bold colours" and "keeps it muted at
  work" are in *tension*, and that tension is exactly what a good assistant should
  notice.
- **Everything or nothing.** You either stuff every memory into the prompt or hope
  top-k retrieval picks the right ones.

`engramdb` treats memory as a **graph of canonical traits** that are consolidated
from noisy observations, linked by typed relations, decayed over time and retrieved
by **spreading activation**.

```python
from engramdb import Brain, Observation, Polarity

brain = Brain()
brain.observe(Observation("u1", "Avoids red-eye flights", context="flights",
                          polarity=Polarity.AVOID, facets={"timing": ["red-eye"]}))

system_prompt += brain.prompt_block("u1", context="flights", message=user_message)
```

## Concepts

| | |
|---|---|
| **Observation** | One thing learned about a user in one interaction. It's noisy, and many observations say the same thing. |
| **Trait** | A canonical, durable belief, consolidated from observations. It has a *strength* that grows with reinforcement and decays with time, plus a context (`global`, `work`, `flights`…), a polarity (`like` / `dislike` / `avoid`) and structured *facets*. |
| **Memory** | An episodic fact ("flying to Lisbon on the 14th"). It fades faster than a trait. Dated events get *more* relevant as the date approaches, then drop off after it passes. A memory that keeps recurring is **promoted** into a trait. |
| **Edges** | `SUPPORTS`, `CONTRADICTS` and `REFINES` between traits. `DERIVED_FROM` links a memory to the trait it fed. |

## How it works

### Write path: consolidation

```
observation ─► embed ─► kNN shortlist of user's traits ─► judge ─┬─ merge: reinforce trait, union facets
                                                                 └─ new:   create trait + typed edges
```

Similarity alone can't decide a merge. Sentences built on the same template score
high, and opposites ("loves spicy" / "avoids spicy") are often near neighbours. So a
pluggable **judge** makes the final call:

- `HeuristicJudge` is deterministic: similarity thresholds plus polarity rules. It
  never merges opposite polarity, and links opposites with `CONTRADICTS`.
- `LLMJudge` takes any `complete_json(system, user) -> dict` callable. Its output is
  **validated**: a merge into an id the model was never shown, or malformed JSON,
  falls back to the heuristic.

Writes are idempotent, so at-least-once queues (SQS, Kafka) are safe.

### Decay

```
base      = min(1, strength + 0.05 · ln(1 + reinforcement_count))
effective = base · 0.5^(days_since_last_seen / half_life)        # traits: 90d, memories: 30d
```

`brain.sweep()` archives inferred traits that have faded below a floor and deletes
long-dormant ones. Traits the user **stated explicitly** never auto-archive.

### Read path: spreading activation

1. **Seed:** nodes in the active context (1.0), global traits (0.6), and kNN hits
   on the current message (1.0).
2. **Spread:** activation flows along edges, damped per edge type
   (`REFINES` 0.7, `SUPPORTS` 0.6, `CONTRADICTS` 0.5), for N hops. A node takes the
   *max* of incoming activation, not the sum, so dense hubs don't take over.
3. **Score:** `activation × decayed strength`, then threshold and top-k.
4. **Tensions:** contradictions between lit nodes are surfaced in the prompt block.

```
- Flying to Lisbon for a friend's wedding [flights, recent]
- Prefers aisle seats on flights [flights, stated]
- Avoids red-eye flights [flights]
- Loves bold colours [global] (in tension with: Keeps it muted at work)
```

### Structured views

- **`brain.preferences(user, context=, dimension=, polarity=)`:** "what do we know
  about X?", with stated preferences first and then by strength.
- **`brain.digest(user, dimensions=)`:** deterministic facet aggregation
  `{context: {polarity: {dimension: [values]}}}` weighted by strength, plus
  **known unknowns** (dimensions you care about that are still empty). A clarifier
  can turn those into questions. An optional `summarize` callable generates the
  prose. Compile it in the background and cache it.
- **`brain.rerank(user, candidates, text_of)`:** personalised re-ranking of any
  retriever's output. Likes boost, dislikes demote, avoids nearly filter. Each
  result says which facet values moved it.

```
0.85 -> 0.98  Morning direct, aisle seat, $380       {'like': ['aisle']}
0.80 -> 0.80  Afternoon, 1 stop, $290
0.92 -> 0.09  Overnight red-eye, window seat, $310   {'avoid': ['red-eye', 'overnight']}
```

## Plug and play

Everything external is an interface, so you can swap pieces without touching the
logic:

| Interface | Default | Swap in |
|---|---|---|
| `BrainStore` | `InMemoryStore` (with JSON save/load) | **`Neo4jStore` (included)**; Postgres + pgvector or Redis by implementing the interface |
| `Embedder` | `HashingEmbedder` (offline, lexical) | OpenAI, bge, sentence-transformers, Cohere |
| `Judge` | `HeuristicJudge` | `LLMJudge(any_llm_json_fn)` |

The core has **zero dependencies**. See [`examples/with_openai.py`](examples/with_openai.py)
for real embeddings and an LLM judge.

## Neo4j storage

`Neo4jStore` keeps each user's brain as a sub-graph in Neo4j, so traits, memories
and their relations persist, are shared across workers and can be explored in the
Neo4j Browser.

```
(:User {user_id})-[:HAS_TRAIT]->(:Trait:EngramNode {id, summary, context, polarity, strength, embedding, ...})
(:User {user_id})-[:HAS_MEMORY]->(:Memory:EngramNode {id, text, event_date, embedding, ...})
(:Trait)-[:SUPPORTS | CONTRADICTS | REFINES]->(:Trait)
(:Memory)-[:DERIVED_FROM]->(:Trait)
(:EngramProcessed {id})                      idempotency markers for queue consumers
```

```python
from engramdb import Brain
from engramdb.neo4j_store import Neo4jStore

store = Neo4jStore.connect("bolt://localhost:7687", "neo4j", "password")
store.setup(embedding_dim=1536)          # constraints + vector indexes; idempotent
brain = Brain(store=store, embedder=my_embedder)
```

- **Per-user isolation.** Every query starts from the user's node, so one user's
  brain is never visible from another's.
- **Attach to your existing users.** `Neo4jStore(driver, user_label="Customer", user_key="customer_id")`
  hangs the brain off `Customer` nodes your application already has.
- **Similarity search, two modes:**
  - `knn_mode="exact"` (default) scores only that user's nodes with
    `vector.similarity.cosine`. Per-user graphs are small, so it's fast and exact,
    and needs no vector index.
  - `knn_mode="index"` uses Neo4j's native vector index. That index spans all users,
    so results are over-fetched (`index_oversample`, default 20×) and then filtered to
    the user. Use it when one user has many thousands of nodes.
- **Same scores as in memory.** Neo4j reports cosine as `(1 + cos) / 2`. The store
  converts it back to plain cosine, so thresholds behave the same on every backend.
- **Right to be forgotten.** `brain.forget(user_id)` deletes the user's traits,
  memories, relations and processed markers, and keeps the `User` node.
- **Your own driver.** Pass your own `neo4j.Driver` to share connection pooling with
  your app, plus `database=` for a non-default database.

Requires **Neo4j 5.18+** (Community, Enterprise or Aura) for exact mode, or 5.13+ for
index mode only. Try it with [`examples/with_neo4j.py`](examples/with_neo4j.py),
which needs no LLM keys, then open <http://localhost:7474> and run
`MATCH (u:User {user_id: 'demo-user'})-[*1..2]-(n) RETURN u, n`.

The same 11-case contract test suite runs against `InMemoryStore` and against
`Neo4jStore` in both modes. CI runs it against a real Neo4j service container.

## Deployment shape

The write path is slower (an embedding call plus an LLM call per observation) and
belongs **off the request path**. Publish observations to a queue and run
`brain.observe()` in a worker. The read path (`activate`, `prompt_block`, cached
`digest`) is cheap and runs inline.

```
chat turn ──► extractor ──► queue ──► worker: brain.observe() / remember() ──► store
    ▲                                       └── periodic: brain.sweep(), recompile digest
    └──── brain.prompt_block() ◄────────────────────────────────────────────────── store
```

## Setup and API keys

### Requirements

| What | Needed for | Version |
|---|---|---|
| Python | everything | **3.10+** |
| *(nothing else)* | the core library, `examples/quickstart.py`, tests | The core has **zero runtime dependencies**. |
| `openai` | `examples/with_openai.py`, or your own OpenAI-backed embedder/judge | ≥ 1.40 (installed by the `openai` extra) |
| `python-dotenv` | loading keys from a `.env` file (optional; plain env vars also work) | ≥ 1.0 (installed by the `openai` and `neo4j` extras) |
| `neo4j` (Python driver) | `Neo4jStore`, `examples/with_neo4j.py` | ≥ 5.14 (installed by the `neo4j` extra) |
| Neo4j server | `Neo4jStore` | **5.18+**: Docker (`docker compose up -d neo4j`), Neo4j Desktop or [Aura](https://neo4j.com/cloud/aura/) |
| Docker | optional, the easiest way to run Neo4j locally | any recent version |
| `pytest` | running the test suite | ≥ 8 (installed by the `dev` extra, along with the `neo4j` driver) |

### 1. Install

```bash
git clone https://github.com/suryajirajebhosale/engramdb.git
cd engramdb
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # core only: no keys, no dependencies
pip install -e ".[dev]"              # + pytest
pip install -e ".[openai]"           # + openai and python-dotenv for the real-LLM example
pip install -e ".[neo4j]"            # + neo4j driver and python-dotenv for Neo4jStore
pip install -e ".[openai,neo4j]"     # extras combine
```

### 2. Run without any keys

```bash
python examples/quickstart.py        # offline demo: hashing embedder + heuristic judge
pytest                               # 33 tests, no network (the Neo4j cases skip)
```

### 3. Add API keys and connection settings

Only needed for real embeddings and the LLM judge (OpenAI), and for Neo4j storage.

The library never reads keys itself. You pass an `Embedder` and a `complete_json`
function into `Brain`, and they bring their own credentials. The bundled OpenAI
example reads them from environment variables:

```bash
cp .env.example .env                 # .env is git-ignored. Never commit it.
```

Then edit `.env`:

| Variable | Required | What it is | Where to get it |
|---|---|---|---|
| `OPENAI_API_KEY` | **yes** (for the OpenAI example) | API key used for embeddings *and* the LLM judge | <https://platform.openai.com/api-keys> |
| `OPENAI_BASE_URL` | no | Send requests to any OpenAI-compatible server instead (Azure OpenAI, vLLM, Ollama `http://localhost:11434/v1`, LiteLLM, OpenRouter) | Your provider's docs |
| `ENGRAM_LLM_MODEL` | no | Model the `LLMJudge` uses. Default `gpt-4o-mini`. It must support JSON mode. | |
| `ENGRAM_EMBEDDING_MODEL` | no | Embedding model. Default `text-embedding-3-small`. | |
| `NEO4J_URI` | **yes** (for Neo4j) | Bolt URI, e.g. `bolt://localhost:7687`; Aura uses `neo4j+s://<id>.databases.neo4j.io` | `docker compose` default, or Aura's credentials file |
| `NEO4J_USER` | **yes** (for Neo4j) | Usually `neo4j` | same |
| `NEO4J_PASSWORD` | **yes** (for Neo4j) | The `docker-compose.yml` default is `engramdb-dev` | Set it yourself, or use the one Aura generates |
| `NEO4J_DATABASE` | no | Non-default database name (Enterprise / Aura) | |
| `ENGRAM_TEST_NEO4J_URI` / `_USER` / `_PASSWORD` | no | Enables the Neo4j cases in `pytest`. **The tests wipe engramdb and `User` nodes, so use a throwaway database.** | |

Instead of a `.env` file you can export the variables in your shell:

```bash
export OPENAI_API_KEY=sk-...
python examples/with_openai.py
```

The library itself never reads `NEO4J_*` variables either. They're only read by
the examples, which pass them to `Neo4jStore.connect(...)`. In your app, pass
credentials however you manage secrets.

To run Neo4j locally and run the full test suite against it:

```bash
docker compose up -d neo4j           # browser at http://localhost:7474
export ENGRAM_TEST_NEO4J_URI=bolt://localhost:7687
export ENGRAM_TEST_NEO4J_PASSWORD=engramdb-dev
pytest                               # 55 tests, including 22 against Neo4j
python examples/with_neo4j.py        # reads NEO4J_* from .env
```

> **Neo4j password gotcha:** `docker compose` reads `NEO4J_PASSWORD` from your
> `.env` (default `engramdb-dev`), but Neo4j only applies it the **first** time the
> data volume is created. To change it later, reset the volume with
> `docker compose down -v`. This deletes the local graph.

### 4. Using another provider

You don't need OpenAI. Anything that satisfies these two shapes works:

```python
class MyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]: ...   # e.g. sentence-transformers, Cohere, bge

def complete_json(system: str, user: str) -> dict: ...            # e.g. Anthropic, Gemini, a local model

brain = Brain(embedder=MyEmbedder(), judge=LLMJudge(complete_json))
```

Put each provider's key wherever that provider's SDK expects it, usually its own
environment variable (for example `ANTHROPIC_API_KEY` or `COHERE_API_KEY`). Install
that SDK yourself. engramdb doesn't depend on it.

> **Tip:** use the same embedding model for everything you store. Vectors from
> different models aren't comparable. If you switch models, re-embed existing traits
> and memories.

## Tuning

All knobs live in `BrainConfig` (`DecayConfig`, `ConsolidationConfig`,
`ActivationConfig`): half-lives, archive floor, kNN sizes and thresholds,
promotion recurrence, seed levels, edge weights, hops and activation threshold.

## License

MIT
