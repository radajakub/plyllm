"""Checks against the real APIs of every provider configured in .env, at two levels:

- `online` (free, `make test-models`): every active model in the registries is still served.
- `live` (paid, a few cents, `make test-live`, which asks first): a plain and a structured call on the
  cheapest model of each code path.

Both are skipped unless pytest runs with their flag. The unit tests check our logic against hand-built
responses; these check it against the real APIs and the installed SDKs, so run them after changing a
provider or upgrading an SDK. Each provider is skipped when its key is missing. An account without credit
fails the paid calls with a `rate_limit` or `provider_error`.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Coroutine
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from plybench.llm import (
    LLM,
    ClaudeProviderConfig,
    GeminiProviderConfig,
    GrokProviderConfig,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMTokens,
    MetacentrumProviderConfig,
    MistralProviderConfig,
    ModelConfig,
    OpenAIProviderConfig,
    Provider,
)

# an account without credit answers 429 (insufficient_quota), which the normal budget would retry for minutes
_LIVE_RETRIES = 2

_SYSTEM = LLMMessage.system("Answer the question.")
_QUESTION = LLMMessage.user("What is 2 + 2? Reply with the number only.")


class _Answer(BaseModel):
    answer: int


# the cheapest model on each code path a provider has; options use the player-config syntax
_MAX_TOKENS = 2048
_OPTIONS = f"thinking_enabled=True,max_tokens={_MAX_TOKENS}"
_EFFORT = "low"
# models that do not accept _EFFORT
_EFFORT_OVERRIDES = {"mistral:mistral-small-4": "high"}
_CASES = [
    "openai:gpt-6-luna",  # reasoning summaries
    "grok:grok-4.3",
    "claude:claude-haiku-4.5",  # numeric thinking budget
    "claude:claude-haiku-5.5",  # effort + adaptive thinking, the path every newer Claude model takes
    "gemini:gemini-3.5-flash-lite",  # thinking_level
    "gemini:gemini-2.5-flash-lite",  # thinking_budget
    "mistral:mistral-small-4",
    "metacentrum:gpt-oss-120b",  # schema enforced by the endpoint
    "metacentrum:gemma-4",  # schema asked for in the prompt
]
# the most one paid call may cost with its output capped at _MAX_TOKENS; keeps expensive models out of _CASES
_MAX_CALL_COST = 0.03
_PROMPT_ALLOWANCE = 1_000  # generous for a one-line question


T = TypeVar("T")


@pytest.fixture
def llm() -> LLM:
    # built per test: the SDKs' connection pools are bound to the event loop that first used them
    llm = LLM(LLMConfig.from_env())
    for client in llm._provider_map.values():
        client._retries = _LIVE_RETRIES
    return llm


def _run(llm: LLM, call: Coroutine[Any, Any, T]) -> T:
    # closes the connection pools on the loop that used them, so nothing is left for the garbage collector
    # to close after the loop is gone ("Event loop is closed")
    async def body() -> T:
        try:
            return await call
        finally:
            await llm.aclose()

    return asyncio.run(body())


def _require(llm: LLM, provider: Provider) -> None:
    if provider not in llm.available_providers:
        pytest.skip(f"{provider.value} is not configured in .env")


# --- offline: runs with the normal suite ---------------------------------------------------------------


@pytest.mark.parametrize("case", _CASES)
def test_no_paid_case_can_cost_more_than_a_few_cents(case: str):
    # catches an expensive model added to _CASES before anyone pays for it; building clients sends nothing
    offline = LLM(
        LLMConfig(
            openai=OpenAIProviderConfig(api_key="offline"),
            grok=GrokProviderConfig(api_key="offline"),
            claude=ClaudeProviderConfig(api_key="offline"),
            gemini=GeminiProviderConfig(api_key="offline"),
            mistral=MistralProviderConfig(api_key="offline"),
            metacentrum=MetacentrumProviderConfig(api_key="offline", base_url="https://offline.invalid/v1"),
        )
    )
    config = ModelConfig.from_string(case)

    worst = offline.calculate_cost(config, LLMTokens(input_tokens=_PROMPT_ALLOWANCE, output_tokens=_MAX_TOKENS))

    assert worst <= _MAX_CALL_COST, f"{case} can cost ${worst:.3f} per call; pick a cheaper model on the same code path"


# --- online, free --------------------------------------------------------------------------------------


@pytest.mark.online
@pytest.mark.parametrize("provider", [provider for provider in Provider if provider is not Provider.HUGGINGFACE])
def test_every_active_model_is_still_served(llm: LLM, provider: Provider):
    """A model the provider stopped serving fails a run with a 400 that reads like an outage. It is marked
    `retired=True` in its models.py instead of deleted, so recorded results still load and cost."""
    _require(llm, provider)
    models = [*llm.get_available_models(provider), *llm.get_available_embedding_models(provider)]
    active = {model.model_string for model in models if not model.retired}
    retired = {model.model_string for model in models if model.retired}

    served = _run(llm, llm.served_models(provider))

    revived = retired & served
    if revived:
        warnings.warn(f"{provider.value} serves models marked retired again: {sorted(revived)}", stacklevel=1)
    missing = active - served
    assert not missing, f"{provider.value} no longer serves {sorted(missing)}: mark them retired=True in its models.py"


# --- live, paid ----------------------------------------------------------------------------------------


def _config(case: str) -> ModelConfig:
    return ModelConfig.from_string(f"{case}:{_OPTIONS},reasoning_effort={_EFFORT_OVERRIDES.get(case, _EFFORT)}")


def _check(llm: LLM, config: ModelConfig, response: LLMResponse, live_costs: list[float]) -> None:
    assert response.model_string == llm.resolve_model(config.provider, config.model_name).model_string
    assert response.tokens.input_tokens > 0 and response.tokens.output_tokens > 0, response.tokens
    live_costs.append(llm.calculate_cost(config, response.tokens))


@pytest.mark.live
@pytest.mark.parametrize("case", _CASES)
def test_a_plain_answer(llm: LLM, live_costs: list[float], case: str):
    config = _config(case)
    _require(llm, config.provider)

    response = _run(llm, llm.generate(config, _SYSTEM, [_QUESTION]))

    assert "4" in response.output_text, response.output_text
    _check(llm, config, response, live_costs)


@pytest.mark.live
@pytest.mark.parametrize("case", _CASES)
def test_a_structured_answer(llm: LLM, live_costs: list[float], case: str):
    config = _config(case)
    _require(llm, config.provider)

    response = _run(llm, llm.generate(config, _SYSTEM, [_QUESTION], output_schema=_Answer))

    assert response.resolve_structured_output(_Answer).answer == 4, response.output_text
    _check(llm, config, response, live_costs)
