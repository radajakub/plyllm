from __future__ import annotations

from openai import AsyncOpenAI

from plyllm.client import DEFAULT_RETRIES
from plyllm.llm_config import LLMConfig
from plyllm.providers.ollama.models import ollama_models
from plyllm.providers.openweight.client import OpenWeightLLMClient
from plyllm.providers.providers import Provider


class OllamaLLMClient(OpenWeightLLMClient):
    provider_key = Provider.OLLAMA
    # Ollama answers one request per model at a time unless OLLAMA_NUM_PARALLEL is raised; more would only queue
    # on the server and run into the timeout there
    default_concurrency = 1

    def __init__(self, client: AsyncOpenAI, concurrency: int | None = None, retries: int = DEFAULT_RETRIES) -> None:
        super().__init__(ollama_models(), client, concurrency, retries)

    @classmethod
    def build(cls, config: LLMConfig) -> OllamaLLMClient | None:
        if config.ollama is None:
            return None
        client = AsyncOpenAI(api_key=config.ollama.api_key, base_url=config.ollama.base_url, timeout=config.ollama.timeout, max_retries=config.ollama.max_retries)
        return cls(client, config.default_concurrency, config.ollama.retries)
