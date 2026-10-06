from datetime import datetime, timezone

import pytest

NOW = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)


class KeyedEmbedder:
    """Test embedder: texts map to hand-picked vectors so similarity is exact."""

    def __init__(self, table: dict[str, list[float]], default: list[float] | None = None):
        self.table = table
        self.default = default or [0.0, 0.0, 0.0, 1.0]

    def embed(self, texts):
        return [self.table.get(t, self.default) for t in texts]


@pytest.fixture
def clock():
    return lambda: NOW
