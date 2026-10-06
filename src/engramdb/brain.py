"""The facade: one object that wires store, embedder and judge together.

    brain = Brain()                                    # in-memory, offline defaults
    brain.observe(Observation("u1", "Avoids red-eye flights", context="travel",
                              polarity=Polarity.AVOID, facets={"timing": ["red-eye"]}))
    lit = brain.activate("u1", context="travel", message="book me a flight to NYC")
    prompt_block = brain.prompt_block("u1", context="travel", message="...")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Mapping, Sequence, TypeVar

from . import activation as act
from .config import BrainConfig
from .consolidate import ConsolidationResult, Consolidator, MemoryResult
from .decay import memory_strength, trait_strength
from .digest import Digest, compile_digest
from .embeddings import Embedder, HashingEmbedder
from .judge import HeuristicJudge, Judge
from .models import GLOBAL, Memory, Observation, Polarity, Trait, utcnow
from .rerank import Ranked, RerankFactors, rerank
from .store import BrainStore, InMemoryStore

T = TypeVar("T")


@dataclass
class SweepReport:
    archived: list[str]
    deleted_traits: list[str]
    deleted_memories: list[str]


class Brain:
    def __init__(
        self,
        store: BrainStore | None = None,
        embedder: Embedder | None = None,
        judge: Judge | None = None,
        config: BrainConfig = BrainConfig(),
        clock: Callable[[], datetime] = utcnow,
    ):
        self.store = store if store is not None else InMemoryStore()
        self.embedder = embedder or HashingEmbedder()
        self.judge = judge or HeuristicJudge()
        self.config = config
        self.clock = clock
        self._consolidator = Consolidator(self.store, self.embedder, self.judge, config.consolidation)

    # ── write path ─────────────────────────────────────────────────────────

    def observe(self, observation: Observation) -> ConsolidationResult:
        """Fold one observation into the user's canonical traits."""
        return self._consolidator.observe(observation)

    def remember(self, memory: Memory) -> MemoryResult:
        """Store an episodic memory; promote it to a trait if the theme keeps recurring."""
        return self._consolidator.remember(memory)

    # ── read path ──────────────────────────────────────────────────────────

    def activate(
        self, user_id: str, *, context: str | None = None, message: str | None = None
    ) -> list[act.Lit]:
        """What is relevant right now, ranked. See :mod:`engramdb.activation`."""
        cfg = self.config.activation
        now = self.clock()
        traits = {t.id: t for t in self.store.traits(user_id)}
        memories = {m.id: m for m in self.store.memories(user_id)}

        strengths = {tid: trait_strength(t, now, self.config.decay) for tid, t in traits.items()}
        strengths.update({mid: memory_strength(m, now, self.config.decay) for mid, m in memories.items()})

        seeds: dict[str, float] = {}

        def seed(node_id: str, value: float) -> None:
            seeds[node_id] = max(seeds.get(node_id, 0.0), value)

        for trait in traits.values():
            if context and trait.context == context:
                seed(trait.id, cfg.context_seed)
            elif trait.context == GLOBAL:
                seed(trait.id, cfg.global_seed)
        for memory in memories.values():
            if context and memory.context == context:
                seed(memory.id, cfg.context_seed)

        if message:
            [vector] = self.embedder.embed([message])
            k, floor = cfg.query_seed_k, cfg.query_min_similarity
            for trait, _ in self.store.nearest_traits(user_id, vector, k=k, min_similarity=floor):
                seed(trait.id, 1.0)
            for memory, _ in self.store.nearest_memories(user_id, vector, k=k, min_similarity=floor):
                seed(memory.id, 1.0)

        node_ids = set(traits) | set(memories)
        edges = self.store.edges(node_ids)
        activation, via = act.spread(seeds, edges, cfg, allowed=node_ids)

        lit = []
        for node_id, score in act.rank(activation, strengths, cfg):
            if node_id in traits:
                t = traits[node_id]
                lit.append(act.Lit(t.id, "trait", t.summary, t.context, score, activation[node_id],
                                   strengths[node_id], t.polarity.value, t.explicit, via.get(node_id, [])))
            else:
                m = memories[node_id]
                lit.append(act.Lit(m.id, "memory", m.text, m.context, score, activation[node_id],
                                   strengths[node_id], via=via.get(node_id, [])))
        act.mark_tensions(lit, edges)
        return lit

    def prompt_block(self, user_id: str, *, context: str | None = None, message: str | None = None,
                     limit: int = 8) -> str:
        return act.format_for_prompt(self.activate(user_id, context=context, message=message), limit)

    def preferences(
        self,
        user_id: str,
        *,
        context: str | None = None,
        dimension: str | None = None,
        polarity: Polarity | None = None,
    ) -> list[tuple[Trait, float]]:
        """Structured query: "what do we know about X in context Y?".

        Ordered stated-first, then by decayed strength: one thing the user
        said outright outranks any amount of accumulated inference.
        """
        now = self.clock()
        rows = []
        for trait in self.store.traits(user_id):
            if context and trait.context not in (context, GLOBAL):
                continue
            if dimension and not trait.facets.get(dimension):
                continue
            if polarity and trait.polarity is not polarity:
                continue
            strength = trait_strength(trait, now, self.config.decay)
            if strength >= self.config.min_preference_strength:
                rows.append((trait, strength))
        rows.sort(key=lambda r: (r[0].explicit, r[1]), reverse=True)
        return rows

    def digest(self, user_id: str, *, dimensions: Sequence[str] | Mapping[str, Sequence[str]] = (),
               summarize: Callable[[list[Trait]], str] | None = None) -> Digest:
        return compile_digest(self.store.traits(user_id), now=self.clock(), decay=self.config.decay,
                              dimensions=dimensions, summarize=summarize)

    def rerank(self, user_id: str, candidates: Sequence[tuple[T, float]], text_of: Callable[[T], str], *,
               context: str | None = None, factors: RerankFactors = RerankFactors()) -> list[Ranked[T]]:
        return rerank(candidates, text_of, self.digest(user_id), context=context, factors=factors)

    # ── maintenance ────────────────────────────────────────────────────────

    def sweep(self, user_id: str) -> SweepReport:
        """Archive faded traits, delete long-dormant ones and expired memories.

        Run it on a schedule (e.g. weekly), not per request.
        """
        now = self.clock()
        decay = self.config.decay
        archived, deleted_traits, deleted_memories = [], [], []

        for trait in self.store.traits(user_id, include_archived=True):
            if trait.archived:
                if now - trait.last_seen_at > timedelta(days=decay.delete_after_dormant_days):
                    self.store.delete_trait(trait.id)
                    deleted_traits.append(trait.id)
            elif not trait.explicit and trait_strength(trait, now, decay) < decay.archive_floor:
                # Stated preferences never auto-archive; only the user retracts them.
                trait.archived, trait.archived_at = True, now
                self.store.put_trait(trait)
                archived.append(trait.id)

        for memory in self.store.memories(user_id):
            if memory.promoted_to is None and memory_strength(memory, now, decay) < decay.archive_floor / 3:
                self.store.delete_memory(memory.id)
                deleted_memories.append(memory.id)

        return SweepReport(archived, deleted_traits, deleted_memories)
