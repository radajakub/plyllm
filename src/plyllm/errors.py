from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from plybench.llm.response import LLMResponse
from plybench.utils.enums import ExtendedEnum


class FailureKind(ExtendedEnum):
    TIMEOUT = "timeout"  # the request never came back inside the client's ceiling
    RATE_LIMIT = "rate_limit"
    CONNECTION = "connection"  # never reached the API
    PROVIDER = "provider_error"  # the API answered with an error
    UNPARSEABLE = "unparseable"  # an answer arrived and did not match the schema
    REFUSAL = "refusal"  # the model declined to answer, or the provider blocked the answer
    NO_OUTPUT = "no_structured_output"  # the call succeeded and carried no parsed object
    STALE_CACHE = "stale_cache"  # a cached answer no longer matches the schema it was stored under
    OTHER = "other"


class LLMCallError(Exception):
    def __init__(self, kind: FailureKind, message: str, response: LLMResponse | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        # the answer that did arrive when the call itself succeeded (UNPARSEABLE, REFUSAL), so its text and
        # its cost are not lost with the exception; None for transport failures
        self.response = response


class LLMRateLimited(LLMCallError):
    def __init__(self, message: str) -> None:
        super().__init__(FailureKind.RATE_LIMIT, message)


class LLMTimedOut(LLMCallError):
    def __init__(self, message: str) -> None:
        super().__init__(FailureKind.TIMEOUT, message)


# one retry rule for every provider: throttling, conflicts, request timeouts and any server-side fault
# (Anthropic signals overload with 529). Other 4xx statuses are the request's fault and fail at once,
# instead of burning the whole retry budget on a call that cannot succeed
_RETRYABLE_STATUSES = frozenset({408, 409, 429})


def retryable_status(status: int | None) -> bool:
    return status is not None and (status in _RETRYABLE_STATUSES or status >= 500)


def status_kind(status: int | None) -> FailureKind:
    if status == 429:
        return FailureKind.RATE_LIMIT
    if status == 408:
        return FailureKind.TIMEOUT
    return FailureKind.PROVIDER


_SUBCLASSES: dict[FailureKind, Callable[[str], LLMCallError]] = {
    FailureKind.RATE_LIMIT: LLMRateLimited,
    FailureKind.TIMEOUT: LLMTimedOut,
}


def as_call_error(error: Exception, kind: FailureKind) -> LLMCallError:
    if isinstance(error, LLMCallError):
        return error
    if isinstance(error, ValidationError):
        # a provider's `error_kind` only knows its SDK's transport types; a schema mismatch raised inside
        # the call is still the answer not fitting, whatever that provider guessed
        kind = FailureKind.UNPARSEABLE
    message = f"{type(error).__name__}: {error}"
    subclass = _SUBCLASSES.get(kind)
    wrapped = subclass(message) if subclass is not None else LLMCallError(kind, message)
    wrapped.__cause__ = error
    return wrapped
