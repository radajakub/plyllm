import asyncio

import pytest

from plybench.llm import (
    LLM,
    LLMCallOptions,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMTokens,
    ModelConfig,
    ModelLimits,
    OpenAIProviderConfig,
    Provider,
)
from plybench.llm.client import LLMClient
from plybench.llm.model import LLMModel
from plybench.llm.response import EmbeddingResponse, OutputText
from plybench.llm.tokens import EmbeddingTokens


class _StubModel(LLMModel):
    def extract_params(self, options: LLMCallOptions) -> dict:
        return {}


class _StubClient(LLMClient):
    provider_key = Provider.OPENAI

    def __init__(self) -> None:
        super().__init__([_StubModel("stub-model", "stub-model-v1")], concurrency=8)
        self.calls: list[tuple[str, LLMCallOptions]] = []
        self.in_flight = 0
        self.peak_in_flight = 0

    @classmethod
    def build(cls, config: LLMConfig) -> "_StubClient":
        return cls()

    def _should_retry_on_error(self, error: Exception) -> bool:
        return False

    async def generate(self, model_name, system, messages, options, output_schema=None) -> LLMResponse:
        model = self.resolve_model(model_name)
        self.calls.append((model_name, options))
        # routed through _dispatch like every real provider, so limits set on this model apply
        tokens = await self._dispatch(model, system, messages, options, self._api_call, (), tokens_of=lambda value: value)
        return LLMResponse(
            provider=self.provider_key,
            model_string=model.model_string,
            tokens=LLMTokens(input_tokens=5, output_tokens=tokens - 5, reasoning_tokens=3),
            items=[OutputText(["ok"])],
            output_text="ok",
        )

    async def _api_call(self) -> int:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        return 12

    async def embed(self, model_name, texts) -> EmbeddingResponse:
        return EmbeddingResponse(self.provider_key, model_name, [[0.0]], EmbeddingTokens(1))


def test_self_disable_when_no_credentials():
    llm = LLM(LLMConfig())
    assert llm.available_providers == []
    with pytest.raises(ValueError):
        asyncio.run(
            llm.generate(
                ModelConfig(Provider.OPENAI, "gpt-5.4"),
                LLMMessage.system("s"),
                [LLMMessage.user("u")],
            )
        )


def test_build_creates_configured_provider_only():
    config = LLMConfig(openai=OpenAIProviderConfig(api_key="sk-test"))
    llm = LLM(config)
    assert llm.available_providers == [Provider.OPENAI]


def test_routing_dispatches_on_provider_and_records_model_string():
    llm = LLM(LLMConfig())
    stub = _StubClient()
    llm._provider_map[Provider.OPENAI] = stub

    mc = ModelConfig(Provider.OPENAI, "stub-model", LLMCallOptions(reasoning_effort="high"))
    resp = asyncio.run(llm.generate(mc, LLMMessage.system("s"), [LLMMessage.user("u")]))

    assert resp.model_string == "stub-model-v1"
    assert resp.tokens.reasoning_tokens == 3
    assert stub.calls == [("stub-model", LLMCallOptions(reasoning_effort="high"))]


def test_resolve_unknown_model_raises():
    stub = _StubClient()
    with pytest.raises(ValueError):
        stub.resolve_model("does-not-exist")


def test_model_limits_apply_to_any_provider():
    # the gate is wired into the shared dispatch path, so limits are not Mistral-specific
    llm = LLM(LLMConfig())
    stub = _StubClient()
    llm._provider_map[Provider.OPENAI] = stub
    llm.set_model_limits(Provider.OPENAI, "stub-model", ModelLimits(max_concurrent=2))

    async def scenario() -> None:
        await asyncio.gather(*(llm.generate(ModelConfig(Provider.OPENAI, "stub-model"), LLMMessage.system("s"), [LLMMessage.user("u")]) for _ in range(6)))

    asyncio.run(scenario())

    # the provider semaphore allows 8, so anything below 6 proves the per-model gate bound it
    assert stub.peak_in_flight == 2


def test_dispatch_is_unbounded_without_limits():
    llm = LLM(LLMConfig())
    stub = _StubClient()
    llm._provider_map[Provider.OPENAI] = stub

    async def scenario() -> None:
        await asyncio.gather(*(llm.generate(ModelConfig(Provider.OPENAI, "stub-model"), LLMMessage.system("s"), [LLMMessage.user("u")]) for _ in range(6)))

    asyncio.run(scenario())

    # only the provider semaphore (8) applies, so all six run together
    assert stub.peak_in_flight == 6


def test_cost_uses_routed_model():
    llm = LLM(LLMConfig())
    stub = _StubClient()
    stub._models["stub-model"].input_cost = 1_000_000  # $1 per input token -> easy assertion
    llm._provider_map[Provider.OPENAI] = stub
    # cost = input_tokens / 1e6 * input_cost = 2 / 1e6 * 1e6 = 2.0
    cost = llm.calculate_cost(ModelConfig(Provider.OPENAI, "stub-model"), LLMTokens(input_tokens=2))
    assert cost == pytest.approx(2.0)
