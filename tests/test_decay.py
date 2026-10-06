from datetime import date, timedelta

import pytest

from engramdb import DecayConfig, Memory, Trait, memory_strength, trait_strength
from engramdb.decay import event_factor

from .conftest import NOW


def test_trait_halves_after_one_half_life():
    fresh = Trait("u", "x", strength=0.6, last_seen_at=NOW)
    stale = Trait("u", "x", strength=0.6, last_seen_at=NOW - timedelta(days=90))
    assert trait_strength(stale, NOW) == pytest.approx(trait_strength(fresh, NOW) / 2)


def test_reinforcement_raises_strength_but_caps_at_one():
    once = Trait("u", "x", strength=0.5, reinforcement_count=1, last_seen_at=NOW)
    often = Trait("u", "x", strength=0.5, reinforcement_count=50, last_seen_at=NOW)
    maxed = Trait("u", "x", strength=0.99, reinforcement_count=1000, last_seen_at=NOW)
    assert trait_strength(often, NOW) > trait_strength(once, NOW)
    assert trait_strength(maxed, NOW) == 1.0


def test_event_factor_rises_toward_event_then_drops_after_grace():
    cfg = DecayConfig()
    today = NOW.date()
    assert event_factor(None, today) == 1.0
    assert event_factor(today + timedelta(days=30), today) == 1.0
    assert event_factor(today + timedelta(days=7), today) == pytest.approx(1.25)
    assert event_factor(today, today) == pytest.approx(1 + cfg.event_boost_max)
    assert event_factor(today - timedelta(days=3), today) == 1.0
    assert event_factor(today - timedelta(days=30), today) == cfg.event_expired_factor


def test_memory_decays_faster_than_trait():
    then = NOW - timedelta(days=30)
    memory = Memory("u", "x", created_at=then)
    trait = Trait("u", "x", strength=1.0, last_seen_at=then)
    assert memory_strength(memory, NOW) == pytest.approx(0.5)
    assert trait_strength(trait, NOW) > 0.75


def test_upcoming_event_memory_is_boosted():
    memory = Memory("u", "trip", created_at=NOW, event_date=date(2026, 6, 2))
    assert memory_strength(memory, NOW) == 1.0  # capped
