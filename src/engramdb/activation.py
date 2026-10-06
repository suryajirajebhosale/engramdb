"""Read path: spreading activation, i.e. which parts of the brain "light up" now.

Dumping every trait into the prompt does not scale and buries what matters.
Activation instead:

1. **Seeds** nodes relevant to the moment: traits and memories in the active
   context, global traits at a lower level, and kNN hits on the current
   message at full activation.
2. **Spreads** activation along typed edges (damped per edge type) for N hops,
   so a seeded trait pulls in what supports, refines or *contradicts* it.
3. **Scores** each node as ``activation x decayed strength``, which weights
   stale beliefs down, and keeps the ones above a threshold.

The arithmetic is a pure function over plain dicts, so it is unit-testable
and storage-agnostic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping

from .config import ActivationConfig
from .models import Edge, EdgeType


@dataclass
class Lit:
    id: str
    kind: Literal["trait", "memory"]
    text: str
    context: str
    score: float
    activation: float
    strength: float
    polarity: str | None = None
    explicit: bool = False
    via: list[tuple[EdgeType, str]] = field(default_factory=list)  # edges that carried activation in
    tensions: list[str] = field(default_factory=list)  # other lit node ids this one CONTRADICTS


def spread(
    seeds: Mapping[str, float],
    edges: Iterable[Edge],
    config: ActivationConfig,
    *,
    allowed: set[str] | None = None,
) -> tuple[dict[str, float], dict[str, list[tuple[EdgeType, str]]]]:
    """Propagate activation; returns (activation per node, incoming edges per node).

    Edges are traversed in both directions. A node's activation is the max of
    what reaches it (not the sum), which keeps scores in [0, 1] and stops
    densely connected hubs from dominating.
    """
    activation = dict(seeds)
    via: dict[str, list[tuple[EdgeType, str]]] = defaultdict(list)
    edges = list(edges)
    frontier = dict(seeds)

    for _ in range(config.hops):
        reached: dict[str, float] = {}
        for edge in edges:
            weight = config.edge_weights.get(edge.type, 0.0)
            if weight <= 0:
                continue
            for src, dst in ((edge.source, edge.target), (edge.target, edge.source)):
                if src not in frontier or (allowed is not None and dst not in allowed):
                    continue
                value = frontier[src] * weight
                if value > activation.get(dst, 0.0):
                    via[dst].append((edge.type, src))
                    reached[dst] = max(reached.get(dst, 0.0), value)
        if not reached:
            break
        activation.update(reached)
        frontier = reached
    return activation, dict(via)


def rank(
    activation: Mapping[str, float],
    strengths: Mapping[str, float],
    config: ActivationConfig,
) -> list[tuple[str, float]]:
    scored = [(node, activation[node] * strengths[node]) for node in activation if node in strengths]
    scored = [(node, score) for node, score in scored if score >= config.threshold]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[: config.limit]


def mark_tensions(lit: list[Lit], edges: Iterable[Edge]) -> None:
    """Fill ``Lit.tensions`` from CONTRADICTS edges whose ends are both lit.

    Done after ranking, not during spread: two seeded nodes can contradict
    each other without the edge ever raising either one's activation.
    """
    lit_ids = {node.id for node in lit}
    by_id = {node.id: node for node in lit}
    for edge in edges:
        if edge.type is EdgeType.CONTRADICTS and edge.source in lit_ids and edge.target in lit_ids:
            by_id[edge.source].tensions.append(edge.target)
            by_id[edge.target].tensions.append(edge.source)


def format_for_prompt(lit: list[Lit], limit: int = 8) -> str:
    """Compact bullet block to drop into a system prompt."""
    lines = []
    names = {node.id: node.text for node in lit}
    for node in lit[:limit]:
        tags = [node.context]
        if node.explicit:
            tags.append("stated")
        if node.kind == "memory":
            tags.append("recent")
        line = f"- {node.text} [{', '.join(tags)}]"
        tensions = [names[other] for other in node.tensions]
        if tensions:
            line += f" (in tension with: {tensions[0]})"
        lines.append(line)
    return "\n".join(lines)
