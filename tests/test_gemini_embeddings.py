import asyncio

import pytest
from google.genai.errors import APIError
from google.genai.types import ContentEmbedding, ContentEmbeddingStatistics, EmbedContentResponse

from plybench.llm.model import EmbeddingTask
from plybench.llm.providers.gemini.client import GeminiLLMClient
from plybench.llm.providers.gemini.models import GeminiEmbeddingModel
from plybench.llm.providers.providers import Provider

_MODEL = "gemini-embedding-2"


def _embedding(values: list[float], token_count: int | None = None, truncated: bool = False) -> ContentEmbedding:
    statistics = None if token_count is None else ContentEmbeddingStatistics(token_count=token_count, truncated=truncated)
    return ContentEmbedding(values=values, statistics=statistics)


class _FakeModels:
    def __init__(self, embeddings_per_call: list[list[ContentEmbedding]]) -> None:
        self._responses = list(embeddings_per_call)
        self.calls: list[dict] = []

    async def embed_content(self, **kwargs) -> EmbedContentResponse:
        self.calls.append(kwargs)
        return EmbedContentResponse(embeddings=self._responses.pop(0))


class _FakeClient:
    def __init__(self, embeddings_per_call: list[list[ContentEmbedding]]) -> None:
        self.models = _FakeModels(embeddings_per_call)


def _client(embeddings_per_call: list[list[ContentEmbedding]]) -> GeminiLLMClient:
    return GeminiLLMClient(_FakeClient(embeddings_per_call), concurrency=4)


def test_gemini_prefixes_query_and_document():
    model = GeminiEmbeddingModel(_MODEL, _MODEL, context_size=8192, input_cost=0.2)

    assert model.format_texts(["what is nim?"], EmbeddingTask.SEARCH_QUERY) == ["task: search result | query: what is nim?"]
    assert model.format_texts(["Nim is a game."], EmbeddingTask.SEARCH_DOCUMENT) == ["title: none | text: Nim is a game."]
    assert model.format_texts(["who wins?"], EmbeddingTask.QUESTION_ANSWERING) == ["task: question answering | query: who wins?"]
    assert model.format_texts(["same"], EmbeddingTask.SEMANTIC_SIMILARITY) == ["task: sentence similarity | query: same"]


def test_every_task_has_a_documented_prefix():
    model = GeminiEmbeddingModel(_MODEL, _MODEL, context_size=8192, input_cost=0.2)

    # format_texts raises on an unmapped task, so covering the enum proves every task is conditioned
    for task in EmbeddingTask:
        assert model.format_texts(["x"], task)[0].endswith(("| query: x", "| text: x"))


def test_embed_sends_one_content_per_text_and_reports_tokens():
    client = _client([[_embedding([0.1, 0.2], token_count=7), _embedding([0.3, 0.4], token_count=5)]])

    resp = asyncio.run(client.embed(_MODEL, ["hello", "world"], EmbeddingTask.SEARCH_QUERY))

    call = client._client.models.calls[0]
    assert call["model"] == _MODEL
    # one Content per text: bare parts would come back as a single aggregated vector
    assert [content.parts[0].text for content in call["contents"]] == ["task: search result | query: hello", "task: search result | query: world"]
    assert resp.provider == Provider.GEMINI
    assert resp.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert resp.tokens.input_tokens == 12
    assert client.calculate_embedding_cost(_MODEL, resp.tokens) == pytest.approx(12 / 1_000_000 * 0.2)


def test_embed_falls_back_to_an_estimate_when_statistics_are_absent():
    client = _client([[_embedding([0.1])]])

    resp = asyncio.run(client.embed(_MODEL, ["a text with no statistics"], EmbeddingTask.CLASSIFICATION))

    # reporting zero here would silently under-report spend
    assert resp.tokens.input_tokens > 0


def test_embed_splits_batches_over_the_model_limit():
    model = GeminiEmbeddingModel(_MODEL, _MODEL, context_size=8192, input_cost=0.2, max_batch_size=2)
    client = _client([[_embedding([1.0], 1), _embedding([2.0], 1)], [_embedding([3.0], 1)]])
    client._embedding_models[_MODEL] = model

    resp = asyncio.run(client.embed(_MODEL, ["a", "b", "c"], EmbeddingTask.CLUSTERING))

    assert [len(call["contents"]) for call in client._client.models.calls] == [2, 1]
    assert resp.embeddings == [[1.0], [2.0], [3.0]]


def test_embed_passes_output_dimensionality():
    client = _client([[_embedding([0.1], 1)]])
    client.resolve_embedding_model(_MODEL).output_dimensionality = 768

    asyncio.run(client.embed(_MODEL, ["hello"], EmbeddingTask.SEARCH_QUERY))

    assert client._client.models.calls[0]["config"].output_dimensionality == 768


def test_embed_rejects_a_truncated_input():
    client = _client([[_embedding([0.1], token_count=9, truncated=True)]])

    with pytest.raises(ValueError, match="truncated"):
        asyncio.run(client.embed(_MODEL, ["hello"], EmbeddingTask.SEARCH_QUERY))


def test_embed_rejects_an_embedding_without_values():
    client = _client([[ContentEmbedding(values=None)]])

    with pytest.raises(ValueError, match="no values"):
        asyncio.run(client.embed(_MODEL, ["hello"], EmbeddingTask.SEARCH_QUERY))


def test_embed_rejects_a_response_that_drops_vectors():
    client = _client([[_embedding([0.1], 1)]])

    with pytest.raises(ValueError, match="cannot be aligned"):
        asyncio.run(client.embed(_MODEL, ["hello", "world"], EmbeddingTask.SEARCH_QUERY))


def test_embed_rejects_input_over_the_context_size():
    client = _client([])

    with pytest.raises(ValueError, match="over the 8192-token context"):
        asyncio.run(client.embed(_MODEL, ["x" * 40_000], EmbeddingTask.SEARCH_DOCUMENT))
    assert client._client.models.calls == []


@pytest.mark.parametrize(
    ("code", "expected"),
    [(429, True), (503, True), (500, True), (400, False), (403, False), (404, False)],
)
def test_only_throttling_and_server_faults_are_retried(code, expected):
    client = _client([])
    error = APIError.__new__(APIError)
    error.code = code

    assert client._should_retry_on_error(error) is expected


def test_non_api_errors_are_not_retried():
    client = _client([])

    assert client._should_retry_on_error(ValueError("boom")) is False
