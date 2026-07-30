from __future__ import annotations

import asyncio
from types import ModuleType
from typing import TYPE_CHECKING

from pydantic import BaseModel

from plybench.llm.client import LLMClient
from plybench.llm.llm_config import HuggingFaceProviderConfig, LLMConfig
from plybench.llm.message import LLMMessage
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.huggingface.models import huggingface_models
from plybench.llm.providers.providers import Provider
from plybench.llm.response import EmbeddingResponse, LLMResponse
from plybench.llm.tokens import EmbeddingTokens

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

_MISSING_EXTRA = "HuggingFace provider requires the 'huggingface' extra. Install it with: pip install plybench[huggingface]"


def _import_torch() -> tuple[ModuleType, ModuleType]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise ImportError(_MISSING_EXTRA) from error
    return torch, functional


def _import_transformers() -> tuple[type, type, ModuleType]:
    try:
        from transformers import AutoModel, AutoTokenizer
        from transformers import logging as hf_logging
    except ImportError as error:
        raise ImportError(_MISSING_EXTRA) from error
    return AutoModel, AutoTokenizer, hf_logging


class HuggingFaceLLMClient(LLMClient):
    provider_key = Provider.HUGGINGFACE
    BATCH_SIZE = 64

    def __init__(self, config: HuggingFaceProviderConfig) -> None:
        # local inference is CPU/GPU bound, so concurrent calls only contend -- keep it serial
        super().__init__(huggingface_models(), concurrency=1)
        self._enabled = config.models  # supported aliases to download/verify for this environment
        self._token = config.token
        self._device_override = config.device

        self._verified: set[str] = set()
        self._tokenizers: dict[str, PreTrainedTokenizerBase] = {}
        self._embedders: dict[str, PreTrainedModel] = {}
        self._device: torch.device | None = None

    @classmethod
    def build(cls, config: LLMConfig) -> HuggingFaceLLMClient | None:
        if config.huggingface is None:
            return None
        return cls(config.huggingface)

    def _should_retry_on_error(self, error: Exception) -> bool:
        return False

    def _resolve_device(self, torch: ModuleType) -> torch.device:
        if self._device_override is not None:
            return torch.device(self._device_override)
        if torch.accelerator.is_available():
            return torch.device(torch.accelerator.current_accelerator())
        return torch.device("cpu")

    def bootstrap(self) -> None:
        auto_model, auto_tokenizer, hf_logging = _import_transformers()
        torch, _ = _import_torch()
        hf_logging.set_verbosity_error()

        self._device = self._resolve_device(torch)
        for name in self._enabled:
            if name in self._verified:
                continue
            model = self.resolve_model(name)  # validates the alias against the supported registry
            # from_pretrained downloads to the HF cache when absent, loads it when already cached
            self._tokenizers[name] = auto_tokenizer.from_pretrained(model.model_string, token=self._token)
            self._embedders[name] = auto_model.from_pretrained(model.model_string, token=self._token).to(self._device).eval()
            self._verified.add(name)

    def _embed_sync(self, model_name: str, texts: list[str]) -> EmbeddingResponse:
        torch, functional = _import_torch()
        model = self.resolve_model(model_name)
        tokenizer = self._tokenizers[model_name]
        embedder = self._embedders[model_name]

        vectors: list[torch.Tensor] = []
        total_tokens = 0
        with torch.no_grad():
            for start in range(0, len(texts), self.BATCH_SIZE):
                batch = texts[start : start + self.BATCH_SIZE]
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=model.max_length, return_tensors="pt")
                encoded = {key: value.to(self._device) for key, value in encoded.items()}

                token_embeddings = embedder(**encoded).last_hidden_state  # [B, T, H]
                mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)  # [B, T, 1]
                summed = (token_embeddings * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                normalized = functional.normalize(summed / counts, p=2, dim=1)

                vectors.append(normalized.cpu())
                total_tokens += int(encoded["attention_mask"].sum().item())

        embeddings = torch.cat(vectors, dim=0).tolist() if vectors else []
        return EmbeddingResponse(self.provider_key, model.model_string, embeddings, EmbeddingTokens(total_tokens))

    async def embed(self, model_name: str, texts: list[str]) -> EmbeddingResponse:
        model = self.resolve_model(model_name)  # reject models outside the supported registry
        if model_name not in self._verified:
            raise ValueError(
                f"HuggingFace model '{model_name}' has not been verified. Add it to hf_models=[...] in your PlyBench(...) bootstrap so it is downloaded and verified before use."
            )
        if not texts:
            return EmbeddingResponse(self.provider_key, model.model_string, [], EmbeddingTokens(0))
        return await self._semaphore.run(lambda: asyncio.to_thread(self._embed_sync, model_name, texts))

    async def generate(
        self,
        model_name: str,
        system: LLMMessage,
        messages: list[LLMMessage],
        options: LLMCallOptions,
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError("HuggingFace generate is not supported yet")
