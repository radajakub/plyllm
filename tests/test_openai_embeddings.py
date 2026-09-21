import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from openai import AsyncOpenAI

from plybench.llm.model import EmbeddingTask
from plybench.llm.providers.openai.client import OpenAILLMClient

_MODEL = "text-embedding-3-small"


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        # the API does not promise input order, so hand them back reversed
        data = [SimpleNamespace(index=index, embedding=[float(index)]) for index in range(len(kwargs["input"]))]
        return SimpleNamespace(data=data[::-1], usage=SimpleNamespace(total_tokens=11))


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


class _FakeOpenAIClient(OpenAILLMClient):
    def __init__(self) -> None:
        fake = _FakeClient()
        self.fake_embeddings = fake.embeddings
        # This test double implements the embedding methods exercised here.
        super().__init__(cast(AsyncOpenAI, fake), concurrency=3)


def _client() -> _FakeOpenAIClient:
    return _FakeOpenAIClient()


def test_embed_realigns_vectors_to_their_inputs():
    client = _client()

    resp = asyncio.run(client.embed(_MODEL, ["a", "b", "c"], EmbeddingTask.SEARCH_QUERY))

    call = client.fake_embeddings.calls[0]
    assert call["model"] == _MODEL
    # OpenAI embeddings take no task conditioning
    assert call["input"] == ["a", "b", "c"]
    assert "dimensions" not in call
    assert resp.embeddings == [[0.0], [1.0], [2.0]]
    assert client.calculate_embedding_cost(_MODEL, resp.tokens) == pytest.approx(11 / 1_000_000 * 0.02)


def test_embed_forwards_dimensions_only_when_set():
    client = _client()
    client.resolve_embedding_model(_MODEL).output_dimensionality = 256

    asyncio.run(client.embed(_MODEL, ["a"], EmbeddingTask.SEARCH_QUERY))

    assert client.fake_embeddings.calls[0]["dimensions"] == 256


def test_unknown_embedding_model_is_rejected():
    client = _client()

    with pytest.raises(ValueError, match="Embedding model"):
        asyncio.run(client.embed("text-embedding-9", ["a"], EmbeddingTask.SEARCH_QUERY))
