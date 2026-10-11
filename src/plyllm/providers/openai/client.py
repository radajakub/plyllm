from __future__ import annotations

import json
from typing import Any

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI

# private: how the SDK turns response JSON into its types; a batch line's body goes through it, so the batch
# path reads the same object generate() gets. A test pins it against SDK upgrades
from openai._models import construct_type_unchecked

# private: the helper responses.parse() uses to build the strict schema; a test pins it against SDK upgrades
from openai.lib._parsing._responses import type_to_text_format_param
from openai.types import Batch
from openai.types.responses import Response
from pydantic import BaseModel

from plyllm.batch import BatchJob
from plyllm.client import LLMClient
from plyllm.errors import FailureKind, LLMCallError, retryable_status, status_kind
from plyllm.llm_config import DEFAULT_PROMPT_CACHE_KEY, LLMConfig
from plyllm.message import LLMMessage
from plyllm.model import EmbeddingModel
from plyllm.options import LLMCallOptions
from plyllm.providers.openai.models import OpenAILLMModel, openai_embedding_models, openai_models
from plyllm.providers.providers import Provider
from plyllm.response import EmbeddingBatch, LLMResponse
from plyllm.tokens import EmbeddingTokens, LLMTokens

# APITimeoutError subclasses APIConnectionError and RateLimitError subclasses APIStatusError, so these two
# cover every failure the SDK raises for a request that was sent
_RETRY_ERRORS = (APIConnectionError, APIStatusError)

# the endpoint generate() calls; one batch carries requests to one endpoint
_BATCH_ENDPOINT = "/v1/responses"
# "up to 50,000 requests, and a batch input file can be up to 200 MB in size" (https://developers.openai.com/api/docs/guides/batch)
_MAX_BATCH_REQUESTS = 50_000
_MAX_BATCH_BYTES = 200 * 1024 * 1024
# a batch in any of these has finished with every request it will ever finish; `failed` means its input
# file failed validation and nothing ran
_BATCH_ENDED = frozenset({"completed", "expired", "cancelled", "failed"})
# the error codes the error file gives a request that never ran
_UNRUN_KINDS = {"batch_expired": FailureKind.EXPIRED, "batch_cancelled": FailureKind.CANCELLED}


def _input_line(request_id: str, body: dict[str, Any]) -> bytes:
    # one line of the input file; its length is what the file's size limit counts
    return (json.dumps({"custom_id": request_id, "method": "POST", "url": _BATCH_ENDPOINT, "body": body}, ensure_ascii=False) + "\n").encode()


def _batch_line(line: dict[str, Any]) -> Response | LLMCallError:
    # one line of a batch's output or error file: {"custom_id", "response": {"status_code", "body"}, "error"}
    error = line.get("error")
    if error:
        code = error.get("code")
        return LLMCallError(_UNRUN_KINDS.get(code, FailureKind.PROVIDER), f"{code}: {error.get('message')}")
    response = line["response"]
    status, body = response["status_code"], response["body"]
    if status != 200:
        message = (body.get("error") or {}).get("message") if isinstance(body, dict) else body
        return LLMCallError(status_kind(status), f"HTTP {status}: {message}")
    return construct_type_unchecked(value=body, type_=Response)


def _unrun(batch: Batch, job: BatchJob) -> dict[str, LLMCallError]:
    # the error file should list every request an expired or cancelled batch never ran; this covers any it
    # leaves out, which would otherwise look lost
    kind = {"expired": FailureKind.EXPIRED, "cancelled": FailureKind.CANCELLED}.get(batch.status)
    if kind is None:
        return {}
    return {request_id: LLMCallError(kind, f"Batch {job.batch_id} was {batch.status} before this request ran") for request_id in job.requests}


def _reasoning_summaries(response: Any) -> list[str]:
    return [summary.text for item in response.output if item.type == "reasoning" for summary in item.summary]


def _refusal(response: Any) -> str | None:
    # a declined request comes back as a refusal content item instead of output text
    refusals = [content.refusal for item in response.output if item.type == "message" for content in item.content if content.type == "refusal"]
    return "\n".join(refusals) if refusals else None


def responses_total_tokens(response: Any) -> int:
    # what the rate gate charges against a tokens_per_minute quota
    usage = response.usage
    if usage is None:
        return 0
    return (usage.input_tokens or 0) + (usage.output_tokens or 0)


def responses_tokens(usage: Any) -> LLMTokens:
    if usage is None:
        return LLMTokens(input_tokens=0, output_tokens=0, cached_input_tokens=0, reasoning_tokens=0)
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return LLMTokens(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cached_input_tokens=getattr(input_details, "cached_tokens", 0) or 0,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) or 0,
    )


class OpenAILLMClient(LLMClient[OpenAILLMModel, Response]):
    provider_key = Provider.OPENAI
    _retry_errors = _RETRY_ERRORS
    supports_batch = True
    max_batch_requests = _MAX_BATCH_REQUESTS
    max_batch_bytes = _MAX_BATCH_BYTES

    def __init__(self, client: AsyncOpenAI, concurrency: int | None = None, prompt_cache_key: str = DEFAULT_PROMPT_CACHE_KEY) -> None:
        super().__init__(openai_models(), openai_embedding_models(), concurrency)
        self._client = client
        self._prompt_cache_key = prompt_cache_key

    @classmethod
    def build(cls, config: LLMConfig) -> OpenAILLMClient | None:
        if config.openai is None:
            return None
        client = AsyncOpenAI(
            api_key=config.openai.api_key,
            organization=config.openai.organization,
            project=config.openai.project,
            timeout=config.openai.timeout,
        )
        return cls(client, config.default_concurrency, config.prompt_cache_key)

    def _should_retry_on_error(self, error: Exception) -> bool:
        if isinstance(error, APIConnectionError):
            return True  # never reached the API, or timed out on the way
        return isinstance(error, APIStatusError) and retryable_status(error.status_code)

    def error_kind(self, error: Exception) -> FailureKind:
        if isinstance(error, APITimeoutError):
            return FailureKind.TIMEOUT  # before APIConnectionError, which it subclasses
        if isinstance(error, APIConnectionError):
            return FailureKind.CONNECTION
        if isinstance(error, APIStatusError):
            return status_kind(error.status_code)
        # e.g. APIResponseValidationError: the API answered, with something the SDK could not read
        return FailureKind.PROVIDER if isinstance(error, APIError) else FailureKind.OTHER

    async def aclose(self) -> None:
        await self._client.close()

    async def served_models(self) -> set[str]:
        return {model.id async for model in self._client.models.list()}

    def _request(self, model: OpenAILLMModel, system: LLMMessage, messages: list[LLMMessage], options: LLMCallOptions, output_schema: type[BaseModel] | None) -> dict[str, Any]:
        # the request body, apart from the call, so a batch line carries exactly what generate() sends
        params = model.extract_params(options)
        kwargs: dict[str, Any] = dict(
            model=model.model_string,
            instructions=system.content,
            input=[message.to_dict() for message in messages],
            store=False,
            prompt_cache_key=self._prompt_cache_key,
            **params,
        )
        if output_schema is not None:
            # the strict schema responses.parse() would send; the answer is checked by LLMClient._answer
            # rather than inside the SDK, so one that does not fit keeps its raw text and tokens
            kwargs["text"] = {"format": type_to_text_format_param(output_schema)}
        return kwargs

    async def _send(self, kwargs: dict[str, Any]) -> Response:
        return await self._client.responses.create(**kwargs)

    def _tokens_of(self, response: Response) -> int:
        return responses_total_tokens(response)

    def _parse(self, model: OpenAILLMModel, response: Response, output_schema: type[BaseModel] | None) -> LLMResponse:
        reasoning = _reasoning_summaries(response)
        tokens = responses_tokens(response.usage)

        refusal = _refusal(response)
        if refusal is not None:
            raise self._refused(model, refusal, refusal, reasoning, tokens)

        return self._answer(model, response.output_text, reasoning, tokens, output_schema)

    async def _submit_batch(self, model: OpenAILLMModel, requests: list[tuple[str, dict[str, Any]]]) -> str:
        # the model is in every line's body (from _request), so the batch itself needs none
        content = b"".join(_input_line(request_id, body) for request_id, body in requests)
        # the upload is retried like any call: a second copy of a file costs nothing. Creating the batch is not, since a create that timed out may have made a batch that runs, and bills, anyway
        upload = await self._batch_api(lambda: self._client.files.create(file=("batch.jsonl", content), purpose="batch"))
        batch = await self._client.batches.create(completion_window="24h", endpoint=_BATCH_ENDPOINT, input_file_id=upload.id)
        return batch.id

    def _batch_size(self, request_id: str, body: dict[str, Any]) -> int:
        return len(_input_line(request_id, body))

    async def _batch_ended(self, batch_id: str) -> bool:
        batch = await self._client.batches.retrieve(batch_id)
        return batch.status in _BATCH_ENDED

    async def _batch_results(self, job: BatchJob) -> dict[str, Response | LLMCallError]:
        assert job.batch_id is not None  # BatchRun collects only submitted jobs
        batch = await self._client.batches.retrieve(job.batch_id)
        if batch.status == "failed":
            # the input file was rejected as a whole, so no request ran
            errors = "; ".join(f"line {error.line}: {error.message}" for error in (batch.errors.data or [] if batch.errors else []))
            return {request_id: LLMCallError(FailureKind.PROVIDER, f"Batch {job.batch_id} failed validation: {errors}") for request_id in job.requests}

        results: dict[str, Response | LLMCallError] = {}
        for file_id in (batch.output_file_id, batch.error_file_id):
            if file_id is not None:
                content = await self._client.files.content(file_id)
                for text in content.text.splitlines():
                    if text.strip():
                        line = json.loads(text)
                        results[line["custom_id"]] = _batch_line(line)
        return {**_unrun(batch, job), **results}

    async def _cancel_batch(self, batch_id: str) -> None:
        await self._client.batches.cancel(batch_id)

    async def _embed_batch(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        response = await self._dispatch_embedding(
            model,
            texts,
            lambda: (
                self._client.embeddings.create(model=model.model_string, input=texts, dimensions=model.output_dimensionality)
                if model.output_dimensionality is not None
                else self._client.embeddings.create(model=model.model_string, input=texts)
            ),
            _RETRY_ERRORS,
            tokens_of=lambda result: result.usage.total_tokens if result.usage is not None else 0,
        )

        # the API may return items out of order, so sort by the index it echoes back
        items = sorted(response.data, key=lambda item: item.index)
        total_tokens = response.usage.total_tokens if response.usage is not None else 0
        return EmbeddingBatch([item.embedding for item in items], EmbeddingTokens(total_tokens))
