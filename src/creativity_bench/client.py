"""OpenAI-compatible LLM client and embedder with provider presets."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
from openai import APIConnectionError, APIStatusError, BadRequestError, OpenAI, RateLimitError


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str | None
    api_key_env: str


PROVIDERS: dict[str, Provider] = {
    "openai": Provider("openai", None, "OPENAI_API_KEY"),  # SDK default base URL
    "zai": Provider("zai", "https://api.z.ai/api/paas/v4", "ZAI_API_KEY"),
    # GLM Coding Plan keys only work against the coding endpoint:
    "zai-coding": Provider("zai-coding", "https://api.z.ai/api/coding/paas/v4", "ZAI_API_KEY"),
    "custom": Provider("custom", None, "LLM_API_KEY"),
}

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_TOKEN_BUDGET = 16_000  # ceiling when doubling the budget for reasoning models


def resolve_provider(name: str, base_url: str | None = None) -> Provider:
    try:
        provider = PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {', '.join(PROVIDERS)}") from None
    if base_url:
        provider = Provider(provider.name, base_url, provider.api_key_env)
    if provider.name == "custom" and not provider.base_url:
        raise ValueError("Provider 'custom' requires --base-url")
    return provider


def _api_key_for(provider: Provider) -> str:
    key = os.environ.get(provider.api_key_env)
    if not key:
        raise RuntimeError(
            f"Set {provider.api_key_env} in your environment to use provider '{provider.name}'."
        )
    return key


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0

    def add(self, usage) -> None:
        self.requests += 1
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0


@dataclass
class LLMClient:
    """Chat-completions client with retries, usage tracking, and <think>-block stripping."""

    provider: Provider
    model: str
    max_retries: int = 4
    timeout: float = 120.0
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        self._client = OpenAI(
            base_url=self.provider.base_url,
            api_key=_api_key_for(self.provider),
            timeout=self.timeout,
            max_retries=0,  # we handle retries ourselves
        )
        self._temperature_supported = True

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        budget = max_tokens
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                text, finish_reason = self._request(messages, temperature, budget)
                if text:
                    return text
                if finish_reason == "length" and budget < MAX_TOKEN_BUDGET:
                    # Reasoning models spend the budget on hidden reasoning before
                    # any visible output; give them headroom and retry immediately.
                    budget = min(budget * 2, MAX_TOKEN_BUDGET)
                    last_error = RuntimeError(
                        f"empty response with finish_reason=length; raised budget to {budget}"
                    )
                    continue
                last_error = RuntimeError(
                    f"Model returned an empty response (finish_reason={finish_reason})"
                )
            except (RateLimitError, APIConnectionError) as e:
                last_error = e
            except APIStatusError as e:
                if e.status_code not in RETRYABLE_STATUS:
                    raise
                last_error = e
            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Generation failed after {self.max_retries + 1} attempts: {last_error}")

    def _request(
        self, messages: list[dict], temperature: float | None, max_tokens: int
    ) -> tuple[str, str | None]:
        kwargs: dict = {}
        if temperature is not None and self._temperature_supported:
            kwargs["temperature"] = temperature
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs,
            )
        except BadRequestError as e:
            # Some models (e.g. OpenAI reasoning models) reject non-default temperature,
            # others reject max_completion_tokens in favor of the legacy max_tokens param.
            msg = str(e).lower()
            if "temperature" in msg and "temperature" in kwargs:
                self._temperature_supported = False
                return self._request(messages, None, max_tokens)
            if "max_completion_tokens" in msg or "max_tokens" in msg:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            else:
                raise
        self.usage.add(getattr(response, "usage", None))
        choice = response.choices[0]
        text = choice.message.content or ""
        return THINK_TAG_RE.sub("", text).strip(), choice.finish_reason


@dataclass
class Embedder:
    """Embedding client for any OpenAI-compatible /embeddings endpoint."""

    provider: Provider
    model: str = "text-embedding-3-small"
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        self._client = OpenAI(
            base_url=self.provider.base_url,
            api_key=_api_key_for(self.provider),
            timeout=60.0,
            max_retries=3,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model, input=texts)
        self.usage.add(getattr(response, "usage", None))
        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        return np.asarray(vectors, dtype=np.float64)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
