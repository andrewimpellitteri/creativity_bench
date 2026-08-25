from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from creativity_bench.client import Usage


class FakeClient:
    """Stands in for LLMClient: routes prompts through a responder callable."""

    def __init__(self, responder, model="fake-model"):
        self.responder = responder
        self.model = model
        self.provider = SimpleNamespace(name="fake")
        self.usage = Usage()
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        self.usage.requests += 1
        return self.responder(messages)

    def generate(self, prompt, *, system=None, **kwargs):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)


class FakeEmbedder:
    """Deterministic embeddings: fixed vectors for registered texts, hashed
    unit vectors otherwise (so distinct texts land far apart)."""

    def __init__(self, fixed: dict[str, np.ndarray] | None = None, dim: int = 16):
        self.fixed = fixed or {}
        self.dim = dim
        self.model = "fake-embed"
        self.usage = Usage()

    def _vector(self, text: str) -> np.ndarray:
        for key, vector in self.fixed.items():
            if key in text:
                return np.asarray(vector, dtype=np.float64)
        digest = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        vector = rng.standard_normal(self.dim)
        return vector / np.linalg.norm(vector)

    def embed(self, texts):
        return np.stack([self._vector(t) for t in texts])

    def embed_one(self, text):
        return self._vector(text)
