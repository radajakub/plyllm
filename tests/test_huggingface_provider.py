import asyncio

import pytest

from plybench.llm import LLM, EmbeddingTask, HuggingFaceProviderConfig, LLMConfig, Provider
from plybench.llm.providers.huggingface.client import HuggingFaceLLMClient


def test_from_env_builds_hf_config_only_when_models_given(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test")

    without_models = LLMConfig.from_env()
    assert without_models.huggingface is None

    with_models = LLMConfig.from_env(huggingface_models=["a/model", "b/model"])
    assert with_models.huggingface is not None
    assert with_models.huggingface.models == ("a/model", "b/model")
    assert with_models.huggingface.token == "hf-test"


def test_router_wires_huggingface_when_configured():
    config = LLMConfig(huggingface=HuggingFaceProviderConfig(models=("sup-simcse-bert",)))
    llm = LLM(config)
    assert llm.available_providers == [Provider.HUGGINGFACE]


def test_embed_on_unverified_supported_model_raises():
    client = HuggingFaceLLMClient(HuggingFaceProviderConfig(models=("sup-simcse-bert",)))
    with pytest.raises(ValueError, match="has not been verified"):
        asyncio.run(client.embed("sup-simcse-bert", ["hello"], EmbeddingTask.SEARCH_QUERY))


def test_embed_on_unsupported_model_raises():
    client = HuggingFaceLLMClient(HuggingFaceProviderConfig(models=("sup-simcse-bert",)))
    with pytest.raises(ValueError, match="not found"):
        asyncio.run(client.embed("not/supported", ["hello"], EmbeddingTask.SEARCH_QUERY))


def test_generate_not_supported():
    client = HuggingFaceLLMClient(HuggingFaceProviderConfig(models=("sup-simcse-bert",)))
    with pytest.raises(NotImplementedError):
        asyncio.run(client.generate("sup-simcse-bert", None, [], None))


def test_embed_produces_normalized_vectors():
    # real embedding needs the `huggingface` extra and downloads the model; skip otherwise
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    client = HuggingFaceLLMClient(HuggingFaceProviderConfig(models=("sup-simcse-bert",)))
    client.bootstrap()

    resp = asyncio.run(client.embed("sup-simcse-bert", ["a game of nim", "a game of nim", "connect four"], EmbeddingTask.SEARCH_QUERY))

    assert resp.provider == Provider.HUGGINGFACE
    assert resp.model_string == "princeton-nlp/sup-simcse-bert-base-uncased"
    assert resp.tokens.input_tokens > 0

    # one vector per input, all the same dimensionality
    assert len(resp.embeddings) == 3
    dims = {len(vector) for vector in resp.embeddings}
    assert len(dims) == 1

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    first, duplicate, other = resp.embeddings
    # L2-normalized -> self dot product is ~1
    assert dot(first, first) == pytest.approx(1.0, abs=1e-4)
    # identical inputs embed identically; a different sentence is less similar
    assert dot(first, duplicate) == pytest.approx(1.0, abs=1e-5)
    assert dot(first, other) < dot(first, duplicate)


def test_embed_empty_texts_returns_no_vectors():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    client = HuggingFaceLLMClient(HuggingFaceProviderConfig(models=("sup-simcse-bert",)))
    client.bootstrap()

    resp = asyncio.run(client.embed("sup-simcse-bert", [], EmbeddingTask.SEARCH_QUERY))
    assert resp.embeddings == []
    assert resp.tokens.input_tokens == 0
