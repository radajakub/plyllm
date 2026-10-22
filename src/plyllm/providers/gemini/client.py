from __future__ import annotations

import json
from typing import Any

import httpx
from google import genai
from google.genai.errors import APIError
from google.genai.types import (
    BatchJob,
    Candidate,
    Content,
    ContentUnion,
    EmbedContentConfig,
    EmbedContentResponse,
    FinishReason,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HttpOptions,
    InlinedRequest,
    Part,
)
from pydantic import BaseModel

from plyllm.batch import BatchJob as StoredBatchJob
from plyllm.client import LLMClient
from plyllm.errors import FailureKind, LLMCallError, retryable_status, status_kind
from plyllm.llm_config import LLMConfig
from plyllm.message import LLMMessage
from plyllm.model import EmbeddingModel
from plyllm.options import LLMCallOptions
from plyllm.providers.gemini.models import GeminiLLMModel, gemini_embedding_models, gemini_models
from plyllm.providers.providers import Provider
from plyllm.rate_limit import estimate_prompt_tokens
from plyllm.response import EmbeddingBatch, LLMResponse
from plyllm.tokens import EmbeddingTokens, LLMTokens

# genai turns HTTP responses into APIError but lets transport failures through untouched. The client is
# pinned to httpx (see build), so httpx's types are the only transport failures it can raise.
_RETRY_ERRORS = (APIError, httpx.TransportError)
_ROLE_MAP = {"user": "user", "assistant": "model", "system": "user"}
# finish reasons that mean the answer was withheld on policy grounds, not cut short or failed
_BLOCKED_FINISH = frozenset({FinishReason.SAFETY, FinishReason.BLOCKLIST, FinishReason.PROHIBITED_CONTENT, FinishReason.SPII, FinishReason.RECITATION})
# inline requests "keep the total request size under 20MB" (https://ai.google.dev/gemini-api/docs/batch-api). _batch_size
# estimates a request from its JSON, while the SDK sends camelCase keys inside its own wrapper, so the limit keeps
# a 10% margin. The docs give no request count for inline batches; the size limit binds first
_MAX_BATCH_BYTES = 18 * 1000 * 1000
_MAX_BATCH_REQUESTS = 100_000
# what the wrapper and the camelCase keys add to each request beyond _inline_size's JSON: 40-70 bytes measured
# on the SDK's own request; a fixed 100 keeps small requests, where it is ~15%, from slipping under the margin
_REQUEST_OVERHEAD = 100
# inline requests have no id field, so each carries its id in the metadata the response echoes back
_ID_KEY = "plyllm_id"
# a job that ended without running every request; the docs say an expired job has no results at all
_UNRUN_KINDS = {"JOB_STATE_EXPIRED": FailureKind.EXPIRED, "JOB_STATE_CANCELLED": FailureKind.CANCELLED, "JOB_STATE_FAILED": FailureKind.PROVIDER}
# requests one answer may take (see _send). A merge answer of ~40k output tokens took about 4 (2026-10-21); an answer
# still unfinished after the last one comes back cut off, and fails to parse under a schema
_MAX_HOPS = 16


def _total_tokens(response: GenerateContentResponse) -> int:
    # what the rate gate charges against a tokens_per_minute quota
    usage = response.usage_metadata
    if usage is None:
        return 0
    return (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)


def _inline_size(request_id: str, body: dict[str, Any]) -> int:
    # the config holds the schema as a pydantic class, which does not dump to JSON, so it is counted as its JSON schema
    config: GenerateContentConfig = body["config"]
    schema = config.response_schema
    payload = {
        "contents": [content.model_dump(mode="json", exclude_none=True) for content in body["contents"]],
        "config": config.model_dump(mode="json", exclude_none=True, exclude={"response_schema"}),
        "schema": schema.model_json_schema() if isinstance(schema, type) and issubclass(schema, BaseModel) else None,
        "metadata": {_ID_KEY: request_id},
    }
    return len(json.dumps(payload, ensure_ascii=False).encode()) + _REQUEST_OVERHEAD


def _unrun(job: BatchJob, stored: StoredBatchJob) -> dict[str, LLMCallError]:
    state = job.state.name if job.state is not None else None
    kind = _UNRUN_KINDS.get(state or "")
    if kind is None:
        return {}
    reason = f": {job.error.message}" if job.error is not None and job.error.message else ""
    return {request_id: LLMCallError(kind, f"Batch {stored.batch_id} ended as {state} before this request ran{reason}") for request_id in stored.requests}


def _continuation(response: GenerateContentResponse) -> bytes | None:
    # the token to resume an unfinished answer from, by the SDK's own rule (should_continue_generation)
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or not candidate.continuation_token or candidate.finish_reason not in (None, FinishReason.CONTINUATION):
        return None
    return candidate.continuation_token


def _usage_sum(hops: list[GenerateContentResponse]) -> GenerateContentResponseUsageMetadata:
    # each hop is a separate request that bills the whole prompt again, so the hops' counts add up
    usages = [hop.usage_metadata for hop in hops if hop.usage_metadata is not None]
    return GenerateContentResponseUsageMetadata(
        prompt_token_count=sum(usage.prompt_token_count or 0 for usage in usages),
        candidates_token_count=sum(usage.candidates_token_count or 0 for usage in usages),
        thoughts_token_count=sum(usage.thoughts_token_count or 0 for usage in usages),
        cached_content_token_count=sum(usage.cached_content_token_count or 0 for usage in usages),
    )


def _joined(hops: list[GenerateContentResponse]) -> GenerateContentResponse:
    if len(hops) == 1:
        return hops[0]
    last = hops[-1]
    parts = [part for hop in hops if hop.candidates and hop.candidates[0].content for part in hop.candidates[0].content.parts or []]
    candidate = (last.candidates[0] if last.candidates else Candidate()).model_copy(update={"content": Content(role="model", parts=parts)})
    return last.model_copy(update={"candidates": [candidate], "usage_metadata": _usage_sum(hops)})


def _prompt_blocked(response: GenerateContentResponse) -> str | None:
    # a filter stopped the prompt before the model saw it. The same request has been seen blocked once and
    # answered another time, so it is kept apart from a refusal for callers that want to send it again
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason is not None:
        return f"prompt blocked ({feedback.block_reason.value})"
    return None


def _blocked(response: GenerateContentResponse) -> str | None:
    # Gemini has no refusal message: a declined request is an answer stopped by a filter
    finish = response.candidates[0].finish_reason if response.candidates else None
    if finish is not None and finish in _BLOCKED_FINISH:
        return f"answer blocked ({finish.value})"
    return None


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


class GeminiLLMClient(LLMClient[GeminiLLMModel, GenerateContentResponse]):
    provider_key = Provider.GEMINI
    _retry_errors = _RETRY_ERRORS
    supports_batch = True
    max_batch_requests = _MAX_BATCH_REQUESTS
    max_batch_bytes = _MAX_BATCH_BYTES

    def __init__(self, client: genai.client.AsyncClient, concurrency: int | None = None, http_client: httpx.AsyncClient | None = None) -> None:
        super().__init__(gemini_models(), gemini_embedding_models(), concurrency)
        self._client = client
        # genai leaves a client it was given open, so the one build() passes in is closed here
        self._http_client = http_client

    @classmethod
    def build(cls, config: LLMConfig) -> GeminiLLMClient | None:
        if config.gemini is None:
            return None
        # genai switches to aiohttp whenever it is installed (any other package may pull it in); passing an httpx
        # client pins the transport, so its failures do not depend on what else happens to be installed
        # (README, "Retries and HTTP transport", says why httpx rather than aiohttp).
        # The timeout still applies per request; HttpOptions takes milliseconds.
        http_client = httpx.AsyncClient()
        http_options = HttpOptions(timeout=int(config.gemini.timeout * 1000), httpx_async_client=http_client)
        client = genai.Client(api_key=config.gemini.api_key, http_options=http_options).aio
        return cls(client, config.default_concurrency, http_client)

    def _should_retry_on_error(self, error: Exception) -> bool:
        if isinstance(error, httpx.TransportError):
            return True  # never reached the API, or timed out on the way
        if not isinstance(error, APIError):
            return False
        code = getattr(error, "code", None)
        # a missing code means the request never reached the API, which is worth another attempt
        return code is None or retryable_status(code)

    def error_kind(self, error: Exception) -> FailureKind:
        if isinstance(error, httpx.TimeoutException):
            return FailureKind.TIMEOUT  # before TransportError, which it subclasses
        if isinstance(error, httpx.TransportError):
            return FailureKind.CONNECTION
        if not isinstance(error, APIError):
            return FailureKind.OTHER
        code = getattr(error, "code", None)
        return FailureKind.CONNECTION if code is None else status_kind(code)

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._http_client is not None:
            await self._http_client.aclose()

    async def served_models(self) -> set[str]:
        return {model.name.removeprefix("models/") async for model in await self._client.models.list() if model.name}

    def _request(self, model: GeminiLLMModel, system: LLMMessage, messages: list[LLMMessage], options: LLMCallOptions, output_schema: type[BaseModel] | None) -> dict[str, Any]:
        # the request, apart from the call, so a batch request carries exactly what generate() sends
        params = model.extract_params(options)
        params.system_instruction = system.content
        if output_schema is not None:
            params.response_mime_type = "application/json"
            params.response_schema = output_schema

        contents: list[ContentUnion] = [Content(role=_ROLE_MAP[message.role], parts=[Part(text=message.content)]) for message in messages]
        return dict(model=model.model_string, contents=contents, config=params)

    async def _send(self, kwargs: dict[str, Any]) -> GenerateContentResponse:
        # Gemini stops a long answer with finish reason CONTINUATION and a token to resume it from. Since 2.30 the
        # SDK follows those hops itself, then re-reads the joined answer against the schema, which crashes on a
        # pydantic class (AttributeError __pydantic_generic_metadata__, google-genai 2.31). plyllm follows the hops
        # instead. The flag is the SDK's own and never sent, so a batch request is unchanged
        config: GenerateContentConfig = kwargs["config"].model_copy(update={"automatic_continuation": False})
        hops = [await self._client.models.generate_content(**{**kwargs, "config": config})]
        while (token := _continuation(hops[-1])) is not None and len(hops) < _MAX_HOPS:
            hops.append(await self._client.models.generate_content(**{**kwargs, "config": config.model_copy(update={"continuation_token": token})}))
        return _joined(hops)

    def _tokens_of(self, response: GenerateContentResponse) -> int:
        return _total_tokens(response)

    def _parse(self, model: GeminiLLMModel, response: GenerateContentResponse, output_schema: type[BaseModel] | None) -> LLMResponse:
        usage = response.usage_metadata
        reasoning_tokens = (usage.thoughts_token_count or 0) if usage is not None else 0
        candidate_tokens = (usage.candidates_token_count or 0) if usage is not None else 0
        input_tokens = (usage.prompt_token_count or 0) if usage is not None else 0
        cached_tokens = (usage.cached_content_token_count or 0) if usage is not None else 0

        tokens = LLMTokens(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=reasoning_tokens + candidate_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        reasoning = _extract_reasoning(response)
        output_text = response.text or ""

        prompt_blocked = _prompt_blocked(response)
        if prompt_blocked is not None:
            answer = LLMResponse.from_parts(self.provider_key, model.model_string, output_text, reasoning, tokens, None)
            raise LLMCallError(FailureKind.PROMPT_BLOCKED, f"Provider blocked the prompt to model {model.model_string}: {prompt_blocked}", answer)

        blocked = _blocked(response)
        if blocked is not None:
            raise self._refused(model, blocked, output_text, reasoning, tokens)

        return self._answer(model, output_text, reasoning, tokens, output_schema)

    def _batch_size(self, request_id: str, body: dict[str, Any]) -> int:
        return _inline_size(request_id, body)

    async def _submit_batch(self, model: GeminiLLMModel, requests: list[tuple[str, dict[str, Any]]]) -> str:
        # inline requests carry no model; the job names it once
        src = [InlinedRequest(contents=body["contents"], config=body["config"], metadata={_ID_KEY: request_id}) for request_id, body in requests]
        job = await self._client.batches.create(model=model.model_string, src=src)
        if job.name is None:
            raise ValueError("Gemini created a batch without a name")
        return job.name

    async def _batch_ended(self, batch_id: str) -> bool:
        job = await self._client.batches.get(name=batch_id)
        return job.done

    async def _batch_results(self, job: StoredBatchJob) -> dict[str, GenerateContentResponse | LLMCallError]:
        assert job.batch_id is not None  # BatchRun collects only submitted jobs
        remote = await self._client.batches.get(name=job.batch_id)
        responses = remote.dest.inlined_responses if remote.dest is not None else None
        ids = list(job.requests)
        results: dict[str, GenerateContentResponse | LLMCallError] = {}
        for index, item in enumerate(responses or []):
            # the echoed id, else the position: inline responses come back in input order
            request_id = (item.metadata or {}).get(_ID_KEY) or (ids[index] if index < len(ids) else None)
            if request_id is None:
                continue
            if item.response is not None:
                results[request_id] = item.response
            else:
                error = item.error
                results[request_id] = LLMCallError(FailureKind.PROVIDER, f"code {error.code if error else None}: {error.message if error else 'no response and no error'}")
        return {**_unrun(remote, job), **results}

    async def _cancel_batch(self, batch_id: str) -> None:
        await self._client.batches.cancel(name=batch_id)

    async def _embed_batch(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        # one Content per text: passing bare parts would return a single aggregated vector instead
        contents: list[ContentUnion] = [Content(parts=[Part.from_text(text=text)]) for text in texts]
        config = EmbedContentConfig(output_dimensionality=model.output_dimensionality)

        response = await self._dispatch_embedding(
            model,
            texts,
            lambda: self._client.models.embed_content(model=model.model_string, contents=contents, config=config),
            _RETRY_ERRORS,
            tokens_of=lambda result: _embedding_tokens(result, texts),
        )

        return EmbeddingBatch(_embedding_vectors(response), EmbeddingTokens(_embedding_tokens(response, texts)))
