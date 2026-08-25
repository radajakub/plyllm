import asyncio

import pytest

from plybench.llm import (
    LLM,
    EmbeddingModelConfig,
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
from plybench.llm.model import EmbeddingModel, EmbeddingTask, LLMModel
from plybench.llm.response import EmbeddingBatch, OutputText
from plybench.llm.tokens import EmbeddingTokens


class _StubModel(LLMModel):
    def extract_params(self, options: LLMCallOptions) -> dict:
        return {}


class _StubEmbeddingModel(EmbeddingModel):
    def format_texts(self, texts: list[str], task: EmbeddingTask) -> list[str]:
        return [f"{task.value}:{text}" for text in texts]


class _StubClient(LLMClient):
    provider_key = Provider.OPENAI

    def __init__(self) -> None:
        super().__init__(
            [_StubModel("stub-model", "stub-model-v1")],
            [_StubEmbeddingModel("stub-embed", "stub-embed-v1", context_size=10, input_cost=1.0, max_batch_size=2)],
            concurrency=8,
        )
        self.calls: list[tuple[str, LLMCallOptions]] = []
        self.embed_batches: list[list[str]] = []
        self.short_batch = False
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

    async def _embed_batch(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        self.embed_batches.append(texts)
        vectors = [[float(len(text))] for text in texts]
        if self.short_batch:
            vectors = vectors[:-1]
        return EmbeddingBatch(vectors, EmbeddingTokens(len(texts)))


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


def test_resolve_unknown_embedding_model_raises():
    stub = _StubClient()
    with pytest.raises(ValueError, match="Embedding model"):
        stub.resolve_embedding_model("does-not-exist")


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


def _stub_llm() -> tuple[LLM, _StubClient]:
    llm = LLM(LLMConfig())
    stub = _StubClient()
    llm._provider_map[Provider.OPENAI] = stub
    return llm, stub


def test_embed_formats_batches_and_merges_tokens():
    llm, stub = _stub_llm()
    config = EmbeddingModelConfig(Provider.OPENAI, "stub-embed")

    resp = asyncio.run(llm.embed(config, ["a", "bb", "ccc", "dddd", "e"], EmbeddingTask.CLUSTERING))

    # max_batch_size=2 splits five inputs into 2 + 2 + 1, each already task-formatted
    assert stub.embed_batches == [["clustering:a", "clustering:bb"], ["clustering:ccc", "clustering:dddd"], ["clustering:e"]]
    assert len(resp.embeddings) == 5
    assert resp.tokens == EmbeddingTokens(5)
    assert resp.model_string == "stub-embed-v1"
    assert llm.calculate_embedding_cost(config, resp.tokens) == pytest.approx(5 / 1_000_000)


def test_embed_on_empty_input_skips_the_provider():
    llm, stub = _stub_llm()

    resp = asyncio.run(llm.embed(EmbeddingModelConfig(Provider.OPENAI, "stub-embed"), [], EmbeddingTask.SEARCH_QUERY))

    assert stub.embed_batches == []
    assert resp.embeddings == []
    assert resp.tokens == EmbeddingTokens(0)


def test_embed_rejects_a_vector_count_that_cannot_be_aligned():
    llm, stub = _stub_llm()
    stub.short_batch = True

    with pytest.raises(ValueError, match="cannot be aligned"):
        asyncio.run(llm.embed(EmbeddingModelConfig(Provider.OPENAI, "stub-embed"), ["a", "bb"], EmbeddingTask.SEARCH_QUERY))


def test_embed_rejects_input_over_the_context_size():
    llm, stub = _stub_llm()

    # context_size=10 tokens against the 4-chars-per-token estimate
    with pytest.raises(ValueError, match="over the 10-token context"):
        asyncio.run(llm.embed(EmbeddingModelConfig(Provider.OPENAI, "stub-embed"), ["x" * 200], EmbeddingTask.SEARCH_QUERY))
    assert stub.embed_batches == []


def test_embedding_limits_shape_the_embedding_gate_only():
    llm, stub = _stub_llm()
    llm.set_embedding_model_limits(Provider.OPENAI, "stub-embed", ModelLimits(max_concurrent=1))

    assert stub.resolve_embedding_model("stub-embed").limits == ModelLimits(max_concurrent=1)
    # the chat model keeps its own (unset) gate
    assert stub.resolve_model("stub-model").limits is None
