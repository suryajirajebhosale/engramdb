"""Consolidation decisions: is this observation an existing trait, or new?

Embedding similarity alone is not enough. Paraphrases of one disposition
should merge ("hates early meetings" / "not a morning person"), but
sentences built on a shared template can score high while meaning different
things, and opposite preferences ("loves spicy food" / "avoids spicy food")
are often near neighbours in embedding space. So a *judge* makes the final call:

- :class:`HeuristicJudge`  deterministic: similarity thresholds plus polarity.
  No LLM, good for tests and cheap paths.
- :class:`LLMJudge`        asks an LLM through any ``complete_json(system, user)``
  callable. Its output is validated, and anything malformed or pointing at
  ids it was never shown falls back to the heuristic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from .models import GLOBAL, EdgeType, Memory, Observation, Polarity, Trait

logger = logging.getLogger(__name__)

Candidates = list[tuple[Trait, float]]  # (trait, cosine similarity), best first


@dataclass
class Decision:
    action: Literal["merge", "new"]
    trait_id: str | None = None  # merge target
    summary: str | None = None  # new trait summary, or improved summary on merge
    edges: list[tuple[EdgeType, str]] = field(default_factory=list)  # (type, existing trait id)


@dataclass
class PromotionDecision:
    action: Literal["reinforce", "new", "skip"]
    trait_id: str | None = None
    summary: str | None = None


class Judge(Protocol):
    def decide(self, observation: Observation, candidates: Candidates) -> Decision: ...

    def decide_promotion(
        self, memory: Memory, cluster: list[Memory], linked_trait: Trait | None
    ) -> PromotionDecision: ...


# ── heuristic ───────────────────────────────────────────────────────────────


@dataclass
class HeuristicJudge:
    merge_similarity: float = 0.82
    relate_similarity: float = 0.55

    def decide(self, observation: Observation, candidates: Candidates) -> Decision:
        for trait, similarity in candidates:
            if (
                similarity >= self.merge_similarity
                and _same_direction(trait.polarity, observation.polarity)
                and trait.context == observation.context
            ):
                return Decision("merge", trait_id=trait.id)

        edges: list[tuple[EdgeType, str]] = []
        for trait, similarity in candidates:
            if similarity < self.relate_similarity:
                continue
            if not _same_direction(trait.polarity, observation.polarity):
                edges.append((EdgeType.CONTRADICTS, trait.id))
            elif trait.context == GLOBAL and observation.context != GLOBAL:
                edges.append((EdgeType.REFINES, trait.id))
            else:
                edges.append((EdgeType.SUPPORTS, trait.id))
        return Decision("new", summary=observation.text, edges=edges)

    def decide_promotion(self, memory, cluster, linked_trait):
        if linked_trait is not None:
            return PromotionDecision("reinforce", trait_id=linked_trait.id)
        return PromotionDecision("new", summary=f"Recurring: {memory.text}")


def _same_direction(a: Polarity, b: Polarity) -> bool:
    return a.sign * b.sign >= 0


# ── LLM-backed ──────────────────────────────────────────────────────────────

CompleteJSON = Callable[[str, str], dict[str, Any]]

CONSOLIDATION_SYSTEM = """\
You maintain a long-term model of one user as a set of canonical traits.
You get ONE new observation and the user's existing traits that look similar.
Decide whether the observation restates an existing trait or is new.

Return JSON only:
{"action": "merge" | "new",
 "trait_id": "<existing id, required for merge>",
 "summary": "<merge: improved summary or null; new: 1-2 sentence canonical summary>",
 "edges": [{"type": "SUPPORTS" | "CONTRADICTS" | "REFINES", "trait_id": "<existing id>"}]}

Rules:
- merge when it is the SAME underlying disposition, even if worded differently.
  Prefer merging; a model full of paraphrases is useless.
- Never merge opposite polarity, and never merge different dispositions that merely share words.
- For new traits add edges to candidates when clearly related: SUPPORTS (same
  direction), CONTRADICTS (opposing pull; these tensions are valuable),
  REFINES (the new trait is a narrower, context-specific form of the candidate).
- Summaries are factual third-person statements of behaviour, not prose.
"""

PROMOTION_SYSTEM = """\
A user has mentioned similar things several times. Decide whether this
recurring theme reflects a durable trait.

Return JSON only:
{"action": "reinforce" | "new" | "skip",
 "trait_id": "<required for reinforce: the linked trait id shown to you>",
 "summary": "<required for new: 1-2 sentence canonical trait summary>"}

- reinforce only if the linked trait is about the SAME topic as these memories.
  Similar sentence phrasing alone is not the same topic.
- new if the memories show a durable pattern that no linked trait captures.
- skip if they are coincidental one-off events.
"""


@dataclass
class LLMJudge:
    complete_json: CompleteJSON
    fallback: HeuristicJudge = field(default_factory=HeuristicJudge)

    def decide(self, observation: Observation, candidates: Candidates) -> Decision:
        payload = {
            "observation": {
                "text": observation.text,
                "context": observation.context,
                "polarity": observation.polarity.value,
                "facets": observation.facets,
            },
            "candidates": [
                {
                    "trait_id": t.id,
                    "summary": t.summary,
                    "context": t.context,
                    "polarity": t.polarity.value,
                    "similarity": round(sim, 3),
                }
                for t, sim in candidates
            ],
        }
        try:
            raw = self.complete_json(CONSOLIDATION_SYSTEM, json.dumps(payload))
            return self._parse_decision(raw, {t.id for t, _ in candidates})
        except Exception as exc:  # noqa: BLE001 - any LLM failure degrades to heuristic
            logger.warning("LLM consolidation failed, using heuristic: %s", exc)
            return self.fallback.decide(observation, candidates)

    def decide_promotion(self, memory, cluster, linked_trait):
        payload = {
            "memory": memory.text,
            "similar_memories": [m.text for m in cluster],
            "linked_trait": {"trait_id": linked_trait.id, "summary": linked_trait.summary}
            if linked_trait
            else None,
        }
        try:
            raw = self.complete_json(PROMOTION_SYSTEM, json.dumps(payload))
            action = raw.get("action")
            if action == "reinforce" and linked_trait and raw.get("trait_id") == linked_trait.id:
                return PromotionDecision("reinforce", trait_id=linked_trait.id)
            if action == "new" and raw.get("summary"):
                return PromotionDecision("new", summary=str(raw["summary"]))
            if action == "skip":
                return PromotionDecision("skip")
            raise ValueError(f"unusable promotion decision: {raw!r}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM promotion failed, using heuristic: %s", exc)
            return self.fallback.decide_promotion(memory, cluster, linked_trait)

    @staticmethod
    def _parse_decision(raw: dict[str, Any], allowed_ids: set[str]) -> Decision:
        action = raw.get("action")
        edges = []
        for edge in raw.get("edges") or []:
            try:
                edge_type = EdgeType(edge["type"])
            except (KeyError, ValueError, TypeError):
                continue
            if edge_type is not EdgeType.DERIVED_FROM and edge.get("trait_id") in allowed_ids:
                edges.append((edge_type, edge["trait_id"]))

        if action == "merge":
            if raw.get("trait_id") not in allowed_ids:
                raise ValueError(f"merge target {raw.get('trait_id')!r} was not a candidate")
            return Decision("merge", trait_id=raw["trait_id"], summary=raw.get("summary") or None)
        if action in ("new", "link"):
            if not raw.get("summary"):
                raise ValueError("new trait without a summary")
            return Decision("new", summary=str(raw["summary"]), edges=edges)
        raise ValueError(f"unknown action {action!r}")
