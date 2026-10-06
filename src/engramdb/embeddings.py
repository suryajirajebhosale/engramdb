"""Embedding interface plus a dependency-free default.

Plug in any model (OpenAI, sentence-transformers, bge, Cohere) by passing
an object with ``embed(texts) -> list[list[float]]``. The bundled
:class:`HashingEmbedder` is lexical, not semantic. It exists so the library,
tests and demos run offline with no model download.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


_STOPWORDS = frozenset(
    "a an and are as at be by for from has have i in is it its of on or that the they "
    "this to was were will with user users their them likes prefers".split()
)


class HashingEmbedder:
    """Feature-hashed bag of unigrams and bigrams, L2-normalised."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec
