"""Compiled, cacheable views of a user's brain.

- ``facets``          deterministic aggregation {context: {polarity: {dimension: [values]}}}
                      weighted by decayed strength. Rankers and filters use it; no LLM needed.
- ``known_unknowns``  (context, dimension) pairs where the user is active but
                      nothing is known yet. A clarifier can turn these into questions.
                      ``dimensions`` is either one list for every context or a
                      ``{context: [dimensions]}`` mapping.
- ``text``            short prose summary for prompts: from your ``summarize``
                      callable (an LLM) or a deterministic fallback.

Recompile it in the background after consolidation and cache it (it changes
only when the brain does). The hot path reads the cache.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Sequence

from .decay import DecayConfig, trait_strength
from .models import Trait


@dataclass
class Digest:
    facets: dict[str, dict[str, dict[str, list[str]]]] = field(default_factory=dict)
    known_unknowns: list[tuple[str, str]] = field(default_factory=list)
    text: str = ""
    trait_count: int = 0


def compile_digest(
    traits: Sequence[Trait],
    *,
    now: datetime,
    decay: DecayConfig = DecayConfig(),
    dimensions: Sequence[str] | Mapping[str, Sequence[str]] = (),
    summarize: Callable[[list[Trait]], str] | None = None,
    max_values: int = 6,
) -> Digest:
    live = [(t, trait_strength(t, now, decay)) for t in traits if not t.archived]
    live = [(t, s) for t, s in live if s >= decay.archive_floor]
    live.sort(key=lambda pair: (pair[0].explicit, pair[1]), reverse=True)

    weights: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    for trait, strength in live:
        for dimension, values in trait.facets.items():
            for value in values:
                weights[trait.context][trait.polarity.value][dimension][value.lower()] += strength

    facets = {
        context: {
            polarity: {
                dim: [v for v, _ in sorted(vals.items(), key=lambda kv: kv[1], reverse=True)][:max_values]
                for dim, vals in dims.items()
            }
            for polarity, dims in polarities.items()
        }
        for context, polarities in weights.items()
    }

    unknowns = []
    for context in sorted({t.context for t, _ in live}):
        known_dims = {dim for dims in facets.get(context, {}).values() for dim in dims}
        wanted = dimensions.get(context, ()) if isinstance(dimensions, Mapping) else dimensions
        unknowns.extend((context, dim) for dim in wanted if dim not in known_dims)

    ordered = [t for t, _ in live]
    text = summarize(ordered) if summarize and ordered else _plain_summary(ordered)
    return Digest(facets=facets, known_unknowns=unknowns, text=text, trait_count=len(ordered))


def _plain_summary(traits: list[Trait], limit: int = 10) -> str:
    by_context: dict[str, list[str]] = defaultdict(list)
    for trait in traits[:limit]:
        by_context[trait.context].append(trait.summary)
    return "\n".join(f"{context}: " + "; ".join(items) for context, items in by_context.items())
