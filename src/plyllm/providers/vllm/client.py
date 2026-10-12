from __future__ import annotations

from openai import AsyncOpenAI

from plyllm.client import DEFAULT_RETRIES
from plyllm.llm_config import LLMConfig
from plyllm.providers.openweight.client import OpenWeightLLMClient
from plyllm.providers.providers import Provider
from plyllm.providers.vllm.models import vllm_models


class VllmLLMClient(OpenWeightLLMClient):
    provider_key = Provider.VLLM
    # our own server batches requests continuously, so it gains from many in flight; the throughput check planned
    # before a full run uses 32
    default_concurrency = 32

    def __init__(self, client: AsyncOpenAI, concurrency: int | None = None, retries: int = DEFAULT_RETRIES) -> None:
        super().__init__(vllm_models(), client, concurrency, retries)

    @classmethod
    def build(cls, config: LLMConfig) -> VllmLLMClient | None:
        if config.vllm is None:
            return None
        client = AsyncOpenAI(api_key=config.vllm.api_key, base_url=config.vllm.base_url, timeout=config.vllm.timeout, max_retries=config.vllm.max_retries)
        return cls(client, config.default_concurrency, config.vllm.retries)
