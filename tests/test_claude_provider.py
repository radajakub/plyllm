import asyncio

import pytest
from anthropic.types import OutputTokensDetails, Usage

from plybench.llm import LLM, ClaudeProviderConfig, LLMCallOptions, LLMConfig, LLMTokens, Provider
from plybench.llm.providers.claude.client import message_tokens
from plybench.llm.providers.claude.models import ClaudeLLMModel, claude_models


def _model(model_name: str) -> ClaudeLLMModel:
    return next(model for model in claude_models() if model.model_name == model_name)


def test_from_env_builds_claude_config_from_api_key(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-test")

    config = LLMConfig.from_env()

    assert config.claude is not None
    assert config.claude.api_key == "sk-ant-test"


def test_router_wires_claude_when_configured():
    llm = LLM(LLMConfig(claude=ClaudeProviderConfig(api_key="sk-ant-test")))

    assert llm.available_providers == [Provider.CLAUDE]
    assert llm.resolve_model(Provider.CLAUDE, "claude-opus-5").model_string == "claude-opus-5"


def test_effort_model_uses_adaptive_thinking_and_output_config():
    params = _model("claude-opus-5").extract_params(LLMCallOptions(thinking_enabled=True, reasoning_effort="xhigh", max_tokens=8000))

    assert params == {
        "max_tokens": 8000,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "xhigh"},
    }


def test_effort_model_falls_back_to_default_max_tokens():
    model = _model("claude-sonnet-5")
    params = model.extract_params(LLMCallOptions(thinking_enabled=True))

    assert params["max_tokens"] == model.max_output_tokens
    # without an explicit effort the API default applies
    assert "output_config" not in params


def test_legacy_model_translates_effort_into_thinking_budget():
    params = _model("claude-haiku-4.5").extract_params(LLMCallOptions(thinking_enabled=True, reasoning_effort="medium", max_tokens=12000))

    assert params["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert "output_config" not in params


def test_thinking_budget_stays_below_max_tokens():
    params = _model("claude-haiku-4.5").extract_params(LLMCallOptions(thinking_enabled=True, reasoning_effort="high", max_tokens=3000))

    assert params["thinking"]["budget_tokens"] < params["max_tokens"]


def test_temperature_only_forwarded_where_the_api_accepts_it():
    options = LLMCallOptions(thinking_enabled=False, temperature=0.3)

    assert "temperature" not in _model("claude-opus-5").extract_params(options)
    assert _model("claude-sonnet-4.6").extract_params(options)["temperature"] == 0.3
    # sampling is rejected together with extended thinking
    assert "temperature" not in _model("claude-haiku-4.5").extract_params(LLMCallOptions(thinking_enabled=True, temperature=0.3))


def test_disabling_thinking_rejected_above_high_effort():
    with pytest.raises(ValueError, match="cannot disable thinking"):
        _model("claude-opus-5").extract_params(LLMCallOptions(thinking_enabled=False, reasoning_effort="max"))


def test_unsupported_effort_rejected():
    with pytest.raises(ValueError, match="does not support reasoning_effort"):
        _model("claude-sonnet-4.6").extract_params(LLMCallOptions(thinking_enabled=True, reasoning_effort="xhigh"))


def test_thinking_only_model_requires_thinking():
    with pytest.raises(ValueError, match="requires thinking"):
        _model("claude-fable-5").extract_params(LLMCallOptions(thinking_enabled=False))


def test_message_tokens_folds_cache_counters_into_input_tokens():
    usage = Usage(input_tokens=100, output_tokens=500, cache_creation_input_tokens=40, cache_read_input_tokens=60)

    tokens = message_tokens(usage, OutputTokensDetails(thinking_tokens=300))

    assert tokens.input_tokens == 200
    assert tokens.cached_input_tokens == 60
    assert tokens.output_tokens == 500
    assert tokens.reasoning_tokens == 300


def test_message_tokens_without_output_details():
    tokens = message_tokens(Usage(input_tokens=10, output_tokens=20), None)

    assert tokens == LLMTokens(input_tokens=10, cached_input_tokens=0, output_tokens=20, reasoning_tokens=0)


def test_cost_charges_cached_tokens_at_the_cache_read_rate():
    model = _model("claude-sonnet-5")
    uncached = message_tokens(Usage(input_tokens=1_000_000, output_tokens=0), None)
    cached = message_tokens(Usage(input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000), None)

    assert model.cost(uncached) == pytest.approx(model.input_cost)
    assert model.cost(cached) == pytest.approx(model.cached_input_cost)


def test_embed_not_supported():
    llm = LLM(LLMConfig(claude=ClaudeProviderConfig(api_key="sk-ant-test")))

    with pytest.raises(NotImplementedError):
        asyncio.run(llm.embed(Provider.CLAUDE, "claude-opus-5", ["hello"]))
