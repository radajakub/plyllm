from __future__ import annotations

from google import genai
from google.genai.errors import APIError
from google.genai.types import Content, EmbedContentConfig, EmbedContentResponse, GenerateContentResponse, Part
from pydantic import BaseModel

from plybench.llm.client import LLMClient
from plybench.llm.llm_config import LLMConfig
from plybench.llm.message import LLMMessage
from plybench.llm.model import EmbeddingModel
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.gemini.models import GeminiLLMModel, gemini_embedding_models, gemini_models
from plybench.llm.providers.providers import Provider
from plybench.llm.rate_limit import estimate_prompt_tokens
from plybench.llm.response import EmbeddingBatch, LLMResponse, OutputText, ReasoningTrace
from plybench.llm.tokens import EmbeddingTokens, LLMTokens

_RETRY_ERRORS = (APIError,)
# every genai failure surfaces as an APIError, so only throttling and server-side faults are worth
# another attempt; a 400 (oversized input, bad request) would just burn the whole retry budget
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_ROLE_MAP = {"user": "user", "assistant": "model", "system": "user"}


def _total_tokens(response: GenerateContentResponse) -> int:
    # what the rate gate charges against a tokens_per_minute quota
    usage = response.usage_metadata
    if usage is None:
        return 0
    return (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)


def _extract_reasoning(response: GenerateContentResponse) -> list[str]:
    if not response.parts:
        return []
    return [part.text for part in response.parts if part.thought and part.text is not None]


def _embedding_vectors(response: EmbedContentResponse) -> list[list[float]]:
    vectors: list[list[float]] = []
    for index, embedding in enumerate(response.embeddings or []):
        if embedding.values is None:
            raise ValueError(f"Gemini returned an embedding with no values at index {index}")
        # check_context works off a char-ratio estimate, so the API is the authority on overflow
        if embedding.statistics is not None and embedding.statistics.truncated:
            raise ValueError(f"Gemini truncated the input at index {index}; chunk it before embedding")
        vectors.append(embedding.values)
    return vectors


def _embedding_tokens(response: EmbedContentResponse, texts: list[str]) -> int:
    counted = [embedding.statistics.token_count for embedding in response.embeddings or [] if embedding.statistics is not None and embedding.statistics.token_count is not None]
    if counted:
        return int(sum(counted))
    # embedding-2 may omit per-vector statistics; estimating beats reporting a zero-cost call
    return estimate_prompt_tokens(*texts)


class GeminiLLMClient(LLMClient):
    provider_key = Provider.GEMINI

    def __init__(self, client: genai.client.AsyncClient, concurrency: int = 20) -> None:
        super().__init__(gemini_models(), gemini_embedding_models(), concurrency)
        self._client = client

    @classmethod
    def build(cls, config: LLMConfig) -> GeminiLLMClient | None:
        if config.gemini is None:
            return None
        client = genai.Client(api_key=config.gemini.api_key).aio
        return cls(client, config.default_concurrency)

    def _should_retry_on_error(self, error: Exception) -> bool:
        if not isinstance(error, _RETRY_ERRORS):
            return False
        code = getattr(error, "code", None)
        # a missing code means the request never reached the API, which is worth another attempt
        return code is None or code in _RETRYABLE_STATUS

    async def generate(
        self,
        model_name: str,
        system: LLMMessage,
        messages: list[LLMMessage],
        options: LLMCallOptions,
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        model: GeminiLLMModel = self.resolve_model(model_name)

        params = model.extract_params(options)
        params.system_instruction = system.content
        if output_schema is not None:
            params.response_mime_type = "application/json"
            params.response_schema = output_schema

        contents = [Content(role=_ROLE_MAP[message.role], parts=[Part(text=message.content)]) for message in messages]

        response = await self._dispatch(
            model,
            system,
            messages,
            options,
            lambda: self._client.models.generate_content(model=model.model_string, contents=contents, config=params),
            _RETRY_ERRORS,
            retry_if=self._should_retry_on_error,
            tokens_of=_total_tokens,
        )

        usage = response.usage_metadata
        reasoning_tokens = usage.thoughts_token_count or 0
        candidate_tokens = usage.candidates_token_count or 0
        input_tokens = usage.prompt_token_count or 0
        cached_tokens = usage.cached_content_token_count or 0

        tokens = LLMTokens(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=reasoning_tokens + candidate_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        reasoning = _extract_reasoning(response)
        output_text = response.text or ""

        items: list[ReasoningTrace | OutputText] = []
        if reasoning:
            items.append(ReasoningTrace(reasoning))
        items.append(OutputText([output_text]))

        return LLMResponse(self.provider_key, model.model_string, tokens, items, output_text, output_schema)

    async def _embed_batch(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        # one Content per text: passing bare parts would return a single aggregated vector instead
        contents = [Content(parts=[Part.from_text(text=text)]) for text in texts]
        config = EmbedContentConfig(output_dimensionality=model.output_dimensionality)

        response = await self._dispatch_embedding(
            model,
            texts,
            lambda: self._client.models.embed_content(model=model.model_string, contents=contents, config=config),
            _RETRY_ERRORS,
            tokens_of=lambda result: _embedding_tokens(result, texts),
        )

        return EmbeddingBatch(_embedding_vectors(response), EmbeddingTokens(_embedding_tokens(response, texts)))
