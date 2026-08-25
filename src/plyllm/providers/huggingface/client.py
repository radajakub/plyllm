from __future__ import annotations

import asyncio
from types import ModuleType
from typing import TYPE_CHECKING

from pydantic import BaseModel

from plybench.llm.client import LLMClient
from plybench.llm.llm_config import HuggingFaceProviderConfig, LLMConfig
from plybench.llm.message import LLMMessage
from plybench.llm.model import EmbeddingModel
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.huggingface.models import huggingface_embedding_models
from plybench.llm.providers.providers import Provider
from plybench.llm.response import EmbeddingBatch, LLMResponse
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

    def __init__(self, config: HuggingFaceProviderConfig) -> None:
        # local inference is CPU/GPU bound, so concurrent calls only contend -- keep it serial.
        # generation is not supported, so the chat registry stays empty.
        super().__init__([], huggingface_embedding_models(), concurrency=1)
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
            model = self.resolve_embedding_model(name)  # validates the alias against the supported registry
            # from_pretrained downloads to the HF cache when absent, loads it when already cached
            self._tokenizers[name] = auto_tokenizer.from_pretrained(model.model_string, token=self._token)
            self._embedders[name] = auto_model.from_pretrained(model.model_string, token=self._token).to(self._device).eval()
            self._verified.add(name)

    def _embed_sync(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        # the base client already split `texts` to the model's max_batch_size
        torch, functional = _import_torch()
        tokenizer = self._tokenizers[model.model_name]
        embedder = self._embedders[model.model_name]

        with torch.no_grad():
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=model.context_size, return_tensors="pt")
            encoded = {key: value.to(self._device) for key, value in encoded.items()}

            token_embeddings = embedder(**encoded).last_hidden_state  # [B, T, H]
            mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)  # [B, T, 1]
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            normalized = functional.normalize(summed / counts, p=2, dim=1)

            total_tokens = int(encoded["attention_mask"].sum().item())

        return EmbeddingBatch(normalized.cpu().tolist(), EmbeddingTokens(total_tokens))

    async def _embed_batch(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        if model.model_name not in self._verified:
            raise ValueError(
                f"HuggingFace model '{model.model_name}' has not been verified. Add it to hf_models=[...] in your PlyBench(...) bootstrap so it is downloaded and verified before use."
            )
        return await self._dispatch_embedding(model, texts, lambda: asyncio.to_thread(self._embed_sync, model, texts), ())

    async def generate(
        self,
        model_name: str,
        system: LLMMessage,
        messages: list[LLMMessage],
        options: LLMCallOptions,
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError("HuggingFace generate is not supported yet")
