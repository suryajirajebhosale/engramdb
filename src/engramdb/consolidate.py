"""Write path: observations -> traits, memories -> (sometimes) traits.

Observation flow:
  1. skip if already processed (queues deliver at least once)
  2. embed
  3. kNN shortlist of the user's traits
  4. judge: merge into an existing trait, or create a new one (+ typed edges)
  5. apply, mark processed

Memory flow:
  1. skip if processed; embed; store
  2. find close matches among the user's earlier memories
  3. enough of them -> judge decides: reinforce the linked trait / new trait / skip
  4. link memories to the trait with DERIVED_FROM edges
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .config import ConsolidationConfig
from .embeddings import Embedder
from .judge import Judge
from .models import Edge, EdgeType, Facets, Memory, Observation, Trait
from .store import BrainStore


@dataclass
class ConsolidationResult:
    action: Literal["merge", "new", "duplicate"]
    trait_id: str | None


@dataclass
class MemoryResult:
    action: Literal["stored", "reinforce", "new", "skip", "duplicate"]
    memory_id: str
    trait_id: str | None = None


class Consolidator:
    def __init__(self, store: BrainStore, embedder: Embedder, judge: Judge, config: ConsolidationConfig):
        self.store = store
        self.embedder = embedder
        self.judge = judge
        self.config = config

    # ── observations ───────────────────────────────────────────────────────

    def observe(self, obs: Observation) -> ConsolidationResult:
        if self.store.is_processed(obs.id):
            return ConsolidationResult("duplicate", None)

        [vector] = self.embedder.embed([obs.text])
        candidates = self.store.nearest_traits(
            obs.user_id, vector, k=self.config.knn_k, min_similarity=self.config.knn_min_similarity
        )
        decision = self.judge.decide(obs, candidates)

        if decision.action == "merge" and decision.trait_id:
            trait = self.store.get_trait(decision.trait_id)
            assert trait is not None
            self._reinforce(trait, obs.observed_at, obs.id, obs.facets, obs.strength, obs.explicit)
            if decision.summary:
                trait.summary = decision.summary
                [trait.embedding] = self.embedder.embed([trait.summary])
            self.store.put_trait(trait)
            result = ConsolidationResult("merge", trait.id)
        else:
            trait = Trait(
                user_id=obs.user_id,
                summary=decision.summary or obs.text,
                context=obs.context,
                polarity=obs.polarity,
                facets=_cap_facets(obs.facets, self.config.max_facet_values),
                strength=obs.strength,
                explicit=obs.explicit,
                first_seen_at=obs.observed_at,
                last_seen_at=obs.observed_at,
                embedding=vector,
                evidence=[obs.id],
            )
            self.store.put_trait(trait)
            for edge_type, target in decision.edges:
                self.store.add_edge(Edge(trait.id, target, edge_type))
            result = ConsolidationResult("new", trait.id)

        self.store.mark_processed(obs.id)
        return result

    def _reinforce(
        self,
        trait: Trait,
        seen_at: datetime,
        evidence_id: str,
        facets: Facets | None = None,
        strength: float = 0.0,
        explicit: bool = False,
    ) -> None:
        trait.reinforcement_count += 1
        trait.last_seen_at = max(trait.last_seen_at, seen_at)
        trait.strength = max(trait.strength, strength)
        trait.explicit = trait.explicit or explicit
        trait.facets = _cap_facets(_union_facets(trait.facets, facets or {}), self.config.max_facet_values)
        trait.archived = False  # fresh evidence revives an archived trait
        trait.archived_at = None
        trait.evidence.append(evidence_id)

    # ── memories ───────────────────────────────────────────────────────────

    def remember(self, memory: Memory) -> MemoryResult:
        if self.store.is_processed(memory.id):
            return MemoryResult("duplicate", memory.id)

        [memory.embedding] = self.embedder.embed([memory.text])
        cluster = [
            m
            for m, _ in self.store.nearest_memories(
                memory.user_id,
                memory.embedding,
                k=self.config.knn_k,
                min_similarity=self.config.memory_knn_min_similarity,
            )
            if m.id != memory.id
        ]
        self.store.put_memory(memory)
        result = MemoryResult("stored", memory.id)

        if len(cluster) >= self.config.memory_min_recurrence:
            linked = next(
                (self.store.get_trait(m.promoted_to) for m in cluster if m.promoted_to), None
            )
            decision = self.judge.decide_promotion(memory, cluster, linked)
            trait: Trait | None = None
            if decision.action == "reinforce" and decision.trait_id:
                trait = self.store.get_trait(decision.trait_id)
                if trait is not None:
                    self._reinforce(trait, memory.created_at, memory.id)
            elif decision.action == "new":
                trait = Trait(
                    user_id=memory.user_id,
                    summary=decision.summary or memory.text,
                    context=memory.context,
                    reinforcement_count=len(cluster) + 1,
                    embedding=self.embedder.embed([decision.summary or memory.text])[0],
                    first_seen_at=min(m.created_at for m in [memory, *cluster]),
                    last_seen_at=memory.created_at,
                    evidence=[m.id for m in [*cluster, memory]],
                )
            if trait is not None:
                self.store.put_trait(trait)
                for m in [memory, *cluster]:
                    if m.promoted_to is None:
                        m.promoted_to = trait.id
                        self.store.put_memory(m)
                        self.store.add_edge(Edge(m.id, trait.id, EdgeType.DERIVED_FROM))
                result = MemoryResult(decision.action, memory.id, trait.id)
            else:
                result = MemoryResult("skip", memory.id)

        self.store.mark_processed(memory.id)
        return result


def _union_facets(current: Facets, incoming: Facets) -> Facets:
    merged = {k: list(v) for k, v in current.items()}
    for key, values in incoming.items():
        bucket = merged.setdefault(key, [])
        for value in values:
            if value not in bucket:
                bucket.append(value)
    return merged


def _cap_facets(facets: Facets, cap: int) -> Facets:
    return {k: list(dict.fromkeys(v))[:cap] for k, v in facets.items() if v}

