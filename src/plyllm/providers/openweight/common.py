from __future__ import annotations

import json
import re
from collections.abc import Sequence

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel

from plyllm.client import DEFAULT_RETRIES, LLMClient, ModelT, ResponseT
from plyllm.errors import FailureKind, retryable_status, status_kind
from plyllm.providers.openweight.models import SchemaDelivery

# What every client of an open-weight model server shares, whether we run the server (vllm) or someone else does
# (metacentrum). Only the reading of answers and the handling of errors live here: nothing that sets how the
# server generates, so a change here changes no request a hosted provider receives.
# Metacentrum imports this module, so a change here also changes how its answers are read, and its recorded
# results were read with the current code. After a change, check that Metacentrum's requests and parsed answers
# are unchanged on the same inputs; sampling, template switches and anything else for our own servers belong in
# models.py and client.py.

# APITimeoutError subclasses APIConnectionError and RateLimitError subclasses APIStatusError, so these two
# cover every failure the SDK raises for a request that was sent
_RETRY_ERRORS = (APIConnectionError, APIStatusError)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def schema_instructions(system: str, output_schema: type[BaseModel]) -> str:
    # "matching this schema" alone made gemma-4 copy the schema back ({"properties": {...}}) in 5 of 6 game
    # prompts on 2026-10-09; saying it is an instance to fill in fixed all 10 test prompts
    schema = json.dumps(output_schema.model_json_schema(), indent=1)
    return (
        f"{system}\n\nReply with a single JSON object that is an instance of the JSON Schema below: fill in the values, "
        f"do not repeat the schema itself. No prose, no code fence.\n{schema}"
    )


def asks_in_prompt(delivery: SchemaDelivery, output_schema: type[BaseModel] | None) -> bool:
    return output_schema is not None and delivery.in_prompt


def enforces(delivery: SchemaDelivery, output_schema: type[BaseModel] | None) -> bool:
    return output_schema is not None and delivery.enforced


def json_body(text: str) -> str:
    return _FENCE.sub("", text).strip()


def split_think(text: str) -> tuple[str, str | None]:
    # some servers leave the model's <think>...</think> block inline before the answer: (answer, reasoning)
    if "</think>" not in text:
        return text, None
    reasoning, tail = text.rsplit("</think>", 1)
    reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()
    return tail.strip(), reasoning or None


class OpenWeightClientBase(LLMClient[ModelT, ResponseT]):
    # an OpenAI-compatible server of open-weight models, reached through the openai SDK
    _retry_errors = _RETRY_ERRORS

    def __init__(self, models: Sequence[ModelT], client: AsyncOpenAI, concurrency: int | None = None, retries: int = DEFAULT_RETRIES) -> None:
        super().__init__(models, [], concurrency, retries)
        self._client = client

    def _should_retry_on_error(self, error: Exception) -> bool:
        if isinstance(error, APIConnectionError):
            return True  # never reached the server, or timed out on the way
        return isinstance(error, APIStatusError) and retryable_status(error.status_code)

    def error_kind(self, error: Exception) -> FailureKind:
        if isinstance(error, APITimeoutError):
            return FailureKind.TIMEOUT  # before APIConnectionError, which it subclasses
        if isinstance(error, APIConnectionError):
            return FailureKind.CONNECTION
        if isinstance(error, APIStatusError):
            return status_kind(error.status_code)
        # e.g. APIResponseValidationError: the server answered, with something the SDK could not read
        return FailureKind.PROVIDER if isinstance(error, APIError) else FailureKind.OTHER

    async def aclose(self) -> None:
        await self._client.close()

    async def served_models(self) -> set[str]:
        return {model.id async for model in self._client.models.list()}
