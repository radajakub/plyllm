"""What `thinking_enabled`, `reasoning_effort` and `temperature` turn into on each provider. An option a model
cannot honour raises instead of being dropped: before, thinking off sent nothing to OpenAI, Grok and
Gemini 2.5 Flash, so those models went on reasoning at their default, and a temperature was silently lost."""

from __future__ import annotations

import pytest

from plybench.llm import LLMCallOptions
from plybench.llm.model import LLMModel
from plybench.llm.providers.claude.models import claude_models
from plybench.llm.providers.gemini.models import gemini_models
from plybench.llm.providers.grok.models import grok_models
from plybench.llm.providers.metacentrum.models import metacentrum_models
from plybench.llm.providers.mistral.models import mistral_models
from plybench.llm.providers.openai.models import openai_models

_ALL: list[LLMModel] = [*openai_models(), *gemini_models(), *grok_models(), *claude_models(), *mistral_models(), *metacentrum_models()]


def _model(name: str) -> LLMModel:
    return next(model for model in _ALL if model.model_name == name)


def test_openai_thinking_off_sends_effort_none_and_may_take_a_temperature():
    params = _model("gpt-5.5").extract_params(LLMCallOptions(thinking_enabled=False, temperature=0.2))

    assert params == {"reasoning": {"effort": "none"}, "temperature": 0.2}


def test_openai_thinking_on_always_asks_for_the_summary():
    assert _model("gpt-6-sol").extract_params(LLMCallOptions(thinking_enabled=True)) == {"reasoning": {"summary": "detailed"}}
    assert _model("gpt-6-sol").extract_params(LLMCallOptions(thinking_enabled=True, reasoning_effort="high")) == {"reasoning": {"summary": "detailed", "effort": "high"}}


@pytest.mark.parametrize("name", ["gpt-6-astra", "gpt-6.1-sol", "gpt-5.5-pro", "gpt-5.4-pro", "gpt-5-mini", "gpt-5-nano"])
def test_openai_models_without_effort_none_cannot_turn_thinking_off(name: str):
    with pytest.raises(ValueError, match="requires thinking"):
        _model(name).extract_params(LLMCallOptions(thinking_enabled=False))


def test_openai_temperature_is_refused_while_reasoning():
    with pytest.raises(ValueError, match="only with thinking disabled"):
        _model("gpt-5.5").extract_params(LLMCallOptions(thinking_enabled=True, reasoning_effort="low", temperature=0.2))
    with pytest.raises(ValueError, match="does not accept temperature"):
        _model("gpt-6.1-sol").extract_params(LLMCallOptions(thinking_enabled=True, temperature=0.2))


@pytest.mark.parametrize("name", ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gemma-4"])
def test_models_whose_default_effort_is_none_need_an_effort_to_reason(name: str):
    with pytest.raises(ValueError, match="does not reason without a reasoning_effort"):
        _model(name).extract_params(LLMCallOptions(thinking_enabled=True))


@pytest.mark.parametrize("name", ["gpt-5.5", "grok-4.3", "gemini-2.5-flash", "mistral-small-4", "claude-haiku-4.5", "kimi-k3"])
def test_an_effort_with_thinking_off_is_refused_where_it_would_be_ignored_or_turn_reasoning_on(name: str):
    with pytest.raises(ValueError, match="only with thinking enabled"):
        _model(name).extract_params(LLMCallOptions(thinking_enabled=False, reasoning_effort="high"))


def test_grok_4_3_turns_reasoning_off_with_effort_none():
    assert _model("grok-4.3").extract_params(LLMCallOptions(thinking_enabled=False)) == {"reasoning": {"effort": "none"}}


@pytest.mark.parametrize("name", ["grok-4.5", "grok-4.20-reasoning"])
def test_grok_models_that_always_reason_refuse_thinking_off(name: str):
    with pytest.raises(ValueError, match="requires thinking"):
        _model(name).extract_params(LLMCallOptions(thinking_enabled=False))


def test_grok_refuses_a_temperature():
    with pytest.raises(ValueError, match="does not accept temperature"):
        _model("grok-4.3").extract_params(LLMCallOptions(thinking_enabled=False, temperature=0.2))


@pytest.mark.parametrize("name", ["gemini-2.5-flash", "gemini-2.5-flash-lite"])
def test_gemini_2_5_flash_thinking_off_sends_a_zero_budget(name: str):
    config = _model(name).extract_params(LLMCallOptions(thinking_enabled=False))

    assert config.thinking_config is not None and config.thinking_config.thinking_budget == 0


def test_every_model_that_needs_an_effort_accepts_one():
    # otherwise thinking on could never be satisfied
    assert all(model.supported_reasoning for model in _ALL if model.needs_effort_to_think)
