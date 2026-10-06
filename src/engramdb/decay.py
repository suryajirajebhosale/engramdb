"""Strength over time.

Traits (dispositions) decay slowly; reinforcement pushes them back up:

    base      = min(1, strength + boost * ln(1 + reinforcement_count))
    effective = base * 0.5 ** (days_since_last_seen / half_life)

Memories (episodes) decay faster. Dated events also get an event factor:
they grow more relevant as the date approaches, then fall off sharply once
the date has passed plus a grace period.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

from .models import Memory, Trait, utcnow


@dataclass(frozen=True)
class DecayConfig:
    half_life_days: float = 90.0
    reinforcement_boost: float = 0.05
    archive_floor: float = 0.3  # below this a trait is archived by the sweep
    delete_after_dormant_days: int = 365  # archived and unseen this long -> deleted

    memory_half_life_days: float = 30.0
    event_approach_window_days: int = 14
    event_boost_max: float = 0.5
    event_grace_days: int = 7
    event_expired_factor: float = 0.2


def _half_life_factor(last_seen: datetime, now: datetime, half_life_days: float) -> float:
    days = max(0.0, (now - last_seen).total_seconds() / 86400)
    return 0.5 ** (days / half_life_days)


def trait_strength(trait: Trait, now: datetime | None = None, config: DecayConfig = DecayConfig()) -> float:
    now = now or utcnow()
    base = min(1.0, trait.strength + config.reinforcement_boost * math.log1p(max(trait.reinforcement_count, 1)))
    return base * _half_life_factor(trait.last_seen_at, now, config.half_life_days)


def event_factor(event_date: date | None, today: date, config: DecayConfig = DecayConfig()) -> float:
    """1.0 for undated memories; ramps up to 1 + boost_max on the event day."""
    if event_date is None:
        return 1.0
    days_until = (event_date - today).days
    if days_until >= 0:
        if days_until > config.event_approach_window_days:
            return 1.0
        closeness = 1 - days_until / config.event_approach_window_days
        return 1.0 + config.event_boost_max * closeness
    if -days_until <= config.event_grace_days:
        return 1.0
    return config.event_expired_factor


def memory_strength(memory: Memory, now: datetime | None = None, config: DecayConfig = DecayConfig()) -> float:
    now = now or utcnow()
    recency = _half_life_factor(memory.created_at, now, config.memory_half_life_days)
    return min(1.0, recency * event_factor(memory.event_date, now.date(), config))
