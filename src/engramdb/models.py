"""Core data model.

Observation  one thing learned about a user in one interaction ("prefers
             window seats"). Raw and noisy; many observations say the same thing.
Trait        a canonical, durable belief about the user, consolidated from
             observations. Has strength, which grows with reinforcement and
             decays with time.
Memory       an episodic fact ("flying to Tokyo on 3 March"). Fades faster than
             a trait; it is promoted into a trait when it keeps recurring.
Edge         typed relation between nodes. Traits relate to each other through
             SUPPORTS / CONTRADICTS / REFINES; a memory relates to the trait it
             fed through DERIVED_FROM.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum

GLOBAL = "global"  # context for traits that hold everywhere


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Polarity(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    AVOID = "avoid"  # hard constraint: never suggest
    NEUTRAL = "neutral"

    @property
    def sign(self) -> int:
        return {"like": 1, "dislike": -1, "avoid": -1, "neutral": 0}[self.value]


class EdgeType(str, Enum):
    SUPPORTS = "SUPPORTS"  # both push choices the same way
    CONTRADICTS = "CONTRADICTS"  # they pull in opposite directions
    REFINES = "REFINES"  # source is a narrower, context-specific form of target
    DERIVED_FROM = "DERIVED_FROM"  # memory -> trait it was promoted into / reinforced


Facets = dict[str, list[str]]  # e.g. {"cuisine": ["thai"], "spice": ["hot"]}


@dataclass
class Observation:
    user_id: str
    text: str
    context: str = GLOBAL
    polarity: Polarity = Polarity.NEUTRAL
    facets: Facets = field(default_factory=dict)
    strength: float = 0.5  # extractor's confidence in [0, 1]
    explicit: bool = False  # user said it outright vs. we inferred it
    observed_at: datetime = field(default_factory=utcnow)
    id: str = field(default_factory=lambda: new_id("obs"))


@dataclass
class Trait:
    user_id: str
    summary: str
    context: str = GLOBAL
    polarity: Polarity = Polarity.NEUTRAL
    facets: Facets = field(default_factory=dict)
    strength: float = 0.5
    reinforcement_count: int = 1
    explicit: bool = False
    first_seen_at: datetime = field(default_factory=utcnow)
    last_seen_at: datetime = field(default_factory=utcnow)
    embedding: list[float] | None = None
    archived: bool = False
    archived_at: datetime | None = None
    evidence: list[str] = field(default_factory=list)  # observation / memory ids
    id: str = field(default_factory=lambda: new_id("trait"))


@dataclass
class Memory:
    user_id: str
    text: str
    context: str = GLOBAL
    event_date: date | None = None  # set for dated events; drives event-aware decay
    created_at: datetime = field(default_factory=utcnow)
    embedding: list[float] | None = None
    promoted_to: str | None = None  # trait id once recurrence promoted it
    id: str = field(default_factory=lambda: new_id("mem"))


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    type: EdgeType
