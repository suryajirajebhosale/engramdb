from __future__ import annotations

from dataclasses import dataclass, field

from .decay import DecayConfig
from .models import EdgeType


@dataclass(frozen=True)
class ConsolidationConfig:
    knn_k: int = 5
    knn_min_similarity: float = 0.45  # shortlist floor; the judge makes the final call
    max_facet_values: int = 8

    # Memory promotion. Template-like short texts score high similarity even
    # when unrelated, so the recurrence bar is deliberately strict and the
    # judge still confirms topical match.
    memory_knn_min_similarity: float = 0.85
    memory_min_recurrence: int = 2  # this memory + N existing close matches


@dataclass(frozen=True)
class ActivationConfig:
    context_seed: float = 1.0  # nodes in the active context
    global_seed: float = 0.6  # traits that hold everywhere
    query_seed_k: int = 4  # kNN hits on the current message
    query_min_similarity: float = 0.3
    hops: int = 1
    edge_weights: dict[EdgeType, float] = field(
        default_factory=lambda: {
            EdgeType.SUPPORTS: 0.6,
            EdgeType.REFINES: 0.7,
            # Contradictions spread too: surfacing the tension ("loves bold
            # colour, but not at work") is what makes the model useful.
            EdgeType.CONTRADICTS: 0.5,
            EdgeType.DERIVED_FROM: 0.5,
        }
    )
    threshold: float = 0.25
    limit: int = 12


@dataclass(frozen=True)
class BrainConfig:
    decay: DecayConfig = DecayConfig()
    consolidation: ConsolidationConfig = ConsolidationConfig()
    activation: ActivationConfig = ActivationConfig()
    min_preference_strength: float = 0.1
