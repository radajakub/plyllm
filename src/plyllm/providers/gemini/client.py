from __future__ import annotations

from google import genai
from google.genai.errors import APIError
from google.genai.types import Content, GenerateContentResponse, Part
from pydantic import BaseModel

from omniplay.llm.client import LLMClient
from omniplay.llm.concurrency import safe_call
from omniplay.llm.llm_config import LLMConfig
from omniplay.llm.message import LLMMessage
from omniplay.llm.options import LLMCallOptions
from omniplay.llm.providers.gemini.models import GeminiLLMModel, gemini_models
from omniplay.llm.providers.providers import Provider
from omniplay.llm.response import EmbeddingResponse, LLMResponse, OutputText, ReasoningTrace
from omniplay.llm.tokens import LLMTokens

_RETRY_ERRORS = (APIError,)
_ROLE_MAP = {"user": "user", "assistant": "model", "system": "user"}


def _extract_reasoning(response: GenerateContentResponse) -> list[str]:
    if not response.parts:
        return []
    return [part.text for part in response.parts if part.thought and part.text is not None]


class GeminiLLMClient(LLMClient):
    provider_key = Provider.GEMINI

    def __init__(self, client: genai.client.AsyncClient, concurrency: int = 20) -> None:
        super().__init__(gemini_models(), concurrency)
        self._client = client

    @classmethod
    def build(cls, config: LLMConfig) -> GeminiLLMClient | None:
        if config.gemini is None:
            return None
        client = genai.Client(api_key=config.gemini.api_key).aio
        return cls(client, config.default_concurrency)

    def _should_retry_on_error(self, error: Exception) -> bool:
        return isinstance(error, _RETRY_ERRORS)

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

        response = await self._semaphore.run(
            lambda: safe_call(
                lambda: self._client.models.generate_content(model=model.model_string, contents=contents, config=params),
                retry_errors=_RETRY_ERRORS,
            )
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

    async def embed(self, model_name: str, texts: list[str]) -> EmbeddingResponse:
        raise NotImplementedError("Gemini embeddings are not supported in this package")
