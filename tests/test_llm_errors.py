"""Failure classification, and the budget that stops two retry layers multiplying.

Six provider SDKs reach this package with no exception base in common. Callers used to either import
several SDKs or match on strings -- the judge runner matched exception class names, and
`TracedMove.output_failure` still matches substrings of a message. The clients already import these types
for `_should_retry_on_error`, so they classify, and the kind travels out on the exception.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import anthropic
import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, AuthenticationError, BadRequestError, RateLimitError
from pydantic import BaseModel, ValidationError

from plybench.llm import ClaudeProviderConfig, GeminiProviderConfig, GrokProviderConfig, LLMCallOptions, LLMConfig, LLMMessage, OpenAIProviderConfig
from plybench.llm.client import LLMClient
from plybench.llm.concurrency import safe_call
from plybench.llm.errors import FailureKind, LLMCallError, LLMRateLimited, LLMTimedOut, as_call_error, retryable_status
from plybench.llm.llm_config import DEFAULT_TIMEOUT
from plybench.llm.providers.claude.client import ClaudeLLMClient
from plybench.llm.providers.gemini.client import GeminiLLMClient
from plybench.llm.providers.grok.client import GrokLLMClient
from plybench.llm.providers.metacentrum.client import MetacentrumLLMClient
from plybench.llm.providers.openai.client import OpenAILLMClient

REQUEST = httpx.Request("POST", "https://example.invalid/v1/responses")


# each provider on the openai SDK keeps its own copy of the retry rule and error_kind, so they can be fixed
# one at a time; every case below runs against all three so the copies cannot drift apart by accident
_OPENAI_SDK_CLIENTS = [OpenAILLMClient, GrokLLMClient, MetacentrumLLMClient]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (APITimeoutError(request=REQUEST), FailureKind.TIMEOUT),
        (RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None), FailureKind.RATE_LIMIT),
        (APIConnectionError(request=REQUEST), FailureKind.CONNECTION),
        (APIStatusError("boom", response=httpx.Response(500, request=REQUEST), body=None), FailureKind.PROVIDER),
        (ValueError("not from the sdk at all"), FailureKind.OTHER),
    ],
)
@pytest.mark.parametrize("client_class", _OPENAI_SDK_CLIENTS)
def test_the_client_names_its_own_sdks_failures(client_class: type[LLMClient], error: Exception, expected: FailureKind):
    # APITimeoutError subclasses APIConnectionError, so order inside error_kind is load-bearing
    assert client_class.__new__(client_class).error_kind(error) is expected


def test_a_wrapped_error_is_catchable_without_importing_anyones_sdk():
    """The point of wrapping. Catching a rate limit generically used to mean naming four SDKs' types at
    the call site; a kind that travels on the exception works for every provider at once."""
    original = RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)

    wrapped = as_call_error(original, FailureKind.RATE_LIMIT)

    assert isinstance(wrapped, LLMRateLimited) and isinstance(wrapped, LLMCallError)
    assert wrapped.kind is FailureKind.RATE_LIMIT
    assert wrapped.__cause__ is original, "the SDK's own object stays reachable for anything that wants it"
    assert isinstance(as_call_error(APITimeoutError(request=REQUEST), FailureKind.TIMEOUT), LLMTimedOut)
    assert as_call_error(ValueError("x"), FailureKind.OTHER).kind is FailureKind.OTHER


def test_a_schema_mismatch_is_unparseable_whatever_the_provider_guessed():
    """The SDKs validate structured output inside the call, so the pydantic error is raised below
    `error_kind`, which only knows transport types and falls through to OTHER. Observed live: 132 qwen
    calls lost to a constrained-decoding loop, all filed as `other`, hiding the one distinction the kinds
    exist to make -- a transport fault is retried, a schema fault needs a different prompt."""

    class Answer(BaseModel):
        labels: list[str]

    with pytest.raises(ValidationError) as caught:
        Answer.model_validate({"labels": "not a list"})
    failure = caught.value

    wrapped = as_call_error(failure, FailureKind.OTHER)
    assert wrapped.kind is FailureKind.UNPARSEABLE
    assert wrapped.__cause__ is failure


def test_an_already_wrapped_error_is_not_wrapped_twice():
    wrapped = as_call_error(ValueError("x"), FailureKind.PROVIDER)
    assert as_call_error(wrapped, FailureKind.OTHER) is wrapped


def test_a_client_can_lower_its_own_retry_count():
    """Our retries wrap the SDK's, so the two multiply: ten attempts around a 600s timeout that itself
    retries is hours on one stalled call. A provider that stalls gets fewer attempts rather than a
    special mechanism."""
    attempts = 0

    async def always_fails():
        nonlocal attempts
        attempts += 1
        raise TimeoutError("never comes back")

    with pytest.raises(TimeoutError):
        asyncio.run(safe_call(always_fails, retry_errors=(TimeoutError,), retries=3, delay_base=0.01, delay_max=0.01))

    assert attempts == 3


# --- one retry rule for every provider ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [(408, True), (409, True), (429, True), (500, True), (503, True), (529, True), (400, False), (401, False), (403, False), (404, False), (422, False), (None, False)],
)
def test_only_throttling_conflicts_timeouts_and_server_faults_are_retryable(status, expected):
    # 529 is Anthropic's overload signal; a fixed list of 5xx codes used to miss it
    assert retryable_status(status) is expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (BadRequestError("bad", response=httpx.Response(400, request=REQUEST), body=None), False),
        (AuthenticationError("no", response=httpx.Response(401, request=REQUEST), body=None), False),
        (RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None), True),
        (APIStatusError("boom", response=httpx.Response(503, request=REQUEST), body=None), True),
        (APIConnectionError(request=REQUEST), True),
        (APITimeoutError(request=REQUEST), True),
        (ValueError("not from the sdk at all"), False),
    ],
)
@pytest.mark.parametrize("client_class", _OPENAI_SDK_CLIENTS)
def test_openai_sdk_providers_retry_by_status(client_class: type[LLMClient], error: Exception, expected: bool):
    # every 4xx used to be retried ten times, so a malformed request took minutes to fail
    assert client_class.__new__(client_class)._should_retry_on_error(error) is expected


def _claude() -> ClaudeLLMClient:
    return ClaudeLLMClient.__new__(ClaudeLLMClient)  # only the retry rule and error_kind are under test


@pytest.mark.parametrize(
    ("error", "retried", "kind"),
    [
        (anthropic.APIStatusError("overloaded", response=httpx.Response(529, request=REQUEST), body=None), True, FailureKind.PROVIDER),
        (anthropic.InternalServerError("boom", response=httpx.Response(500, request=REQUEST), body=None), True, FailureKind.PROVIDER),
        (anthropic.BadRequestError("bad", response=httpx.Response(400, request=REQUEST), body=None), False, FailureKind.PROVIDER),
        (anthropic.AuthenticationError("no", response=httpx.Response(401, request=REQUEST), body=None), False, FailureKind.PROVIDER),
        (anthropic.RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None), True, FailureKind.RATE_LIMIT),
        (anthropic.APITimeoutError(request=REQUEST), True, FailureKind.TIMEOUT),
        (anthropic.APIConnectionError(request=REQUEST), True, FailureKind.CONNECTION),
        (ValueError("not from the sdk at all"), False, FailureKind.OTHER),
    ],
)
def test_claude_retries_by_status(error: Exception, retried: bool, kind: FailureKind):
    # 529 is Anthropic's overload signal, the failure most worth retrying
    client = _claude()
    assert client._should_retry_on_error(error) is retried
    assert client.error_kind(error) is kind


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (httpx.ConnectError("refused"), FailureKind.CONNECTION),
        (httpx.RemoteProtocolError("server disconnected"), FailureKind.CONNECTION),
        (httpx.ReadTimeout("slow"), FailureKind.TIMEOUT),
        (httpx.ConnectTimeout("slow"), FailureKind.TIMEOUT),
    ],
)
def test_gemini_retries_the_transport_errors_genai_lets_through(error: Exception, kind: FailureKind):
    # genai wraps HTTP responses in APIError but not transport failures, which used to fail at once as `other`
    client = GeminiLLMClient.__new__(GeminiLLMClient)
    assert client._should_retry_on_error(error) is True
    assert client.error_kind(error) is kind


def test_gemini_is_pinned_to_httpx_even_when_aiohttp_is_installed():
    """With aiohttp installed genai uses it, and its failures are aiohttp's types, which the retry rule
    does not know. Pinning the transport keeps the rule complete whatever else is installed."""
    client = GeminiLLMClient.build(LLMConfig(gemini=GeminiProviderConfig(api_key="test", timeout=30.0)))

    assert client is not None
    api_client = client._client._api_client
    assert not api_client._use_aiohttp()
    assert api_client._http_options.timeout == 30_000


# --- timeouts -----------------------------------------------------------------------------------------


def test_remote_providers_default_to_a_finite_timeout(monkeypatch):
    for key in ("OPENAI", "GROK", "CLAUDE", "GEMINI", "MISTRAL"):
        monkeypatch.setenv(f"{key}_API_KEY", "test")
        monkeypatch.delenv(f"{key}_TIMEOUT", raising=False)
    monkeypatch.setenv("CLAUDE_TIMEOUT", "900")

    config = LLMConfig.from_env()

    assert config.openai is not None and config.openai.timeout == DEFAULT_TIMEOUT
    assert config.grok is not None and config.grok.timeout == DEFAULT_TIMEOUT
    assert config.gemini is not None and config.gemini.timeout == DEFAULT_TIMEOUT
    assert config.mistral is not None and config.mistral.timeout == DEFAULT_TIMEOUT
    assert config.claude is not None and config.claude.timeout == 900.0


def test_clients_hand_the_configured_timeout_to_their_sdk():
    """`timeout=None` let a stalled connection hang a run forever, with no error for the retry layer to act on."""
    config = LLMConfig(
        openai=OpenAIProviderConfig(api_key="test", timeout=123.0), grok=GrokProviderConfig(api_key="test"), claude=ClaudeProviderConfig(api_key="test", timeout=45.0)
    )

    openai_client = OpenAILLMClient.build(config)
    grok_client = GrokLLMClient.build(config)
    claude_client = ClaudeLLMClient.build(config)

    assert openai_client is not None and openai_client._client.timeout == 123.0
    assert grok_client is not None and grok_client._client.timeout == DEFAULT_TIMEOUT
    assert claude_client is not None and claude_client._client.timeout == 45.0


# --- the rule as a call sees it ----------------------------------------------------------------------


class _FailingResponses:
    def __init__(self, errors: list[Exception]) -> None:
        self._errors = errors
        self.attempts = 0

    async def create(self, **kwargs):
        self.attempts += 1
        raise self._errors[min(self.attempts, len(self._errors)) - 1]


def _openai_client(errors: list[Exception]) -> tuple[OpenAILLMClient, _FailingResponses]:
    responses = _FailingResponses(errors)
    sdk = SimpleNamespace(responses=responses)
    client = OpenAILLMClient(cast(AsyncOpenAI, sdk), concurrency=1)
    client._retries = 3  # the budget itself is not under test; a small one keeps the test fast
    return client, responses


def _generate(client: OpenAILLMClient):
    return asyncio.run(client.generate("gpt-5.4", LLMMessage.system("rules"), [LLMMessage.user("move")], LLMCallOptions()))


def test_a_rejected_request_is_attempted_once(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    client, responses = _openai_client([BadRequestError("bad", response=httpx.Response(400, request=REQUEST), body=None)])

    with pytest.raises(LLMCallError) as caught:
        _generate(client)

    assert responses.attempts == 1
    assert caught.value.kind is FailureKind.PROVIDER


def test_a_throttled_request_uses_the_whole_retry_budget(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    client, responses = _openai_client([RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)])

    with pytest.raises(LLMRateLimited):
        _generate(client)

    assert responses.attempts == 3


async def _no_sleep(_delay: float) -> None:
    return None
