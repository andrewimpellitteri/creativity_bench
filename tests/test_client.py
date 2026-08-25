from types import SimpleNamespace

import pytest

from creativity_bench.client import (
    LLMClient,
    is_free_openrouter_model,
    resolve_provider,
)


def make_response(content, finish_reason):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class StubCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return LLMClient(provider=resolve_provider("openai"), model="test-model")


def install_stub(client, responses):
    stub = StubCompletions(responses)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return stub


def test_doubles_budget_when_reasoning_eats_the_tokens(client):
    stub = install_stub(
        client,
        [
            make_response("", "length"),
            make_response("", "length"),
            make_response("finally some text", "stop"),
        ],
    )
    assert client.generate("hi", max_tokens=1000) == "finally some text"
    assert [c["max_completion_tokens"] for c in stub.calls] == [1000, 2000, 4000]


def test_strips_think_blocks(client):
    install_stub(client, [make_response("<think>hidden reasoning</think>answer", "stop")])
    assert client.generate("hi") == "answer"


def test_empty_stop_response_eventually_raises(client):
    client.max_retries = 1
    install_stub(client, [make_response("", "stop")] * 2)
    with pytest.raises(RuntimeError, match="empty response"):
        client.generate("hi")


def test_tracks_usage(client):
    install_stub(client, [make_response("ok", "stop")])
    client.generate("hi")
    assert client.usage.requests == 1
    assert client.usage.prompt_tokens == 10
    assert client.usage.completion_tokens == 5


def test_openrouter_provider_preset():
    provider = resolve_provider("openrouter")
    assert provider.name == "openrouter"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    assert provider.api_key_env == "OPENROUTER_API_KEY"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("stealth/ox-alpha", True),
        ("openrouter/free", True),
        ("z-ai/glm-5.2:free", True),
        ("deepseek/deepseek-chat-v3-0324", False),
    ],
)
def test_free_openrouter_model_classifier(model, expected):
    assert is_free_openrouter_model(model) is expected
