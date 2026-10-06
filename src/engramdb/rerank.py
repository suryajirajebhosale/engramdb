"""Personalised re-ranking from digest facets.

Given candidates from any retriever (products, articles, restaurants...) with
base scores, nudge them by what the brain knows:

    like    value appears in candidate -> score x boost (capped at 1.0)
    dislike value appears              -> score x dislike_factor
    avoid   value appears              -> score x avoid_factor (near-hard filter)

Matching is case-insensitive substring over the text you extract from each
candidate, deliberately simple and explainable. Each result says which
facet values moved it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, Sequence, TypeVar

from .digest import Digest
from .models import GLOBAL

T = TypeVar("T")


@dataclass(frozen=True)
class RerankFactors:
    like: float = 1.15
    dislike: float = 0.4
    avoid: float = 0.1
    min_term_length: int = 3


@dataclass
class Ranked(Generic[T]):
    item: T
    score: float
    base_score: float
    matched: dict[str, list[str]] = field(default_factory=dict)  # polarity -> values


def rerank(
    candidates: Sequence[tuple[T, float]],
    text_of: Callable[[T], str],
    digest: Digest,
    *,
    context: str | None = None,
    factors: RerankFactors = RerankFactors(),
) -> list[Ranked[T]]:
    terms = _terms(digest, context, factors.min_term_length)
    multiplier = {"like": factors.like, "dislike": factors.dislike, "avoid": factors.avoid}

    ranked = []
    for item, base in candidates:
        blob = text_of(item).lower()
        score = base
        matched: dict[str, list[str]] = {}
        for polarity, values in terms.items():
            hits = [v for v in values if v in blob]
            if hits:
                matched[polarity] = hits
                score *= multiplier[polarity]
        if "like" in matched:
            score = min(score, max(base, 1.0))  # a boost never pushes a [0,1] score past 1
        ranked.append(Ranked(item, score, base, matched))
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def _terms(digest: Digest, context: str | None, min_len: int) -> dict[str, list[str]]:
    contexts = [GLOBAL] + ([context] if context and context != GLOBAL else [])
    terms: dict[str, list[str]] = {}
    for ctx in contexts:
        for polarity, dims in digest.facets.get(ctx, {}).items():
            if polarity not in ("like", "dislike", "avoid"):
                continue
            bucket = terms.setdefault(polarity, [])
            for values in dims.values():
                bucket.extend(v for v in values if len(v) >= min_len and v not in bucket)
    return terms
