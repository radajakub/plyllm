"""Token prices: the cached-input rate, the cache-write rate, and the Batch API rate that `cost(..., batch=True)` applies."""

from __future__ import annotations

import pytest

from plybench.llm import LLMTokens
from plybench.llm.model import LLMModel
from plybench.llm.providers.claude.models import ClaudeLLMModel, claude_models
from plybench.llm.providers.gemini.models import gemini_models
from plybench.llm.providers.grok.models import grok_models
from plybench.llm.providers.metacentrum.models import metacentrum_models
from plybench.llm.providers.mistral.models import mistral_models
from plybench.llm.providers.openai.models import openai_models

_ALL: list[LLMModel] = [*openai_models(), *gemini_models(), *grok_models(), *claude_models(), *mistral_models(), *metacentrum_models()]
# a million of each, so a cost reads directly as the sum of the per-1M prices
_TOKENS = LLMTokens(input_tokens=2_000_000, cached_input_tokens=1_000_000, output_tokens=1_000_000)


def _model(name: str) -> LLMModel:
    return next(model for model in _ALL if model.model_name == name)


def test_gemini_cached_tokens_are_no_longer_free():
    # 1M uncached at 0.30 + 1M cached at 0.03 + 1M output at 2.50; cached tokens used to cost 0
    assert _model("gemini-2.5-flash").cost(_TOKENS) == pytest.approx(0.30 + 0.03 + 2.50)


def test_a_flat_batch_discount_halves_every_token_type():
    # gpt-5.5: 5.00 / 0.50 / 30.00 standard
    assert _model("gpt-5.5").cost(_TOKENS, batch=True) == pytest.approx((5.0 + 0.5 + 30.0) / 2)


def test_a_listed_batch_cache_price_overrides_the_ratio():
    # gemini-2.5-flash batch: 0.15 input, 0.03 cached (same as standard), 1.25 output
    assert _model("gemini-2.5-flash").cost(_TOKENS, batch=True) == pytest.approx(0.15 + 0.03 + 1.25)


def test_grok_batch_discount_is_per_model():
    # grok-4.3: 20% off 1.25 / 0.20 / 2.50
    assert _model("grok-4.3").cost(_TOKENS, batch=True) == pytest.approx(0.8 * (1.25 + 0.2 + 2.5))
    with pytest.raises(ValueError, match="has no batch price"):
        _model("grok-4.5").cost(_TOKENS, batch=True)


def test_the_standard_price_is_unchanged_by_the_batch_fields():
    for model in _ALL:
        uncached = (_TOKENS.input_tokens - _TOKENS.cached_input_tokens) / 1_000_000 * model.input_cost
        assert model.cost(_TOKENS) == pytest.approx(uncached + model.cached_input_cost + model.output_cost), model.model_name


def test_claude_charges_cache_writes_at_a_premium_over_plain_input():
    # claude-haiku-4.5: 1.00 input, 1.25 5-minute cache write, 0.10 cache read, 5.00 output
    tokens = LLMTokens(input_tokens=3_000_000, cached_input_tokens=1_000_000, cache_write_tokens=1_000_000, output_tokens=1_000_000)

    assert _model("claude-haiku-4.5").cost(tokens) == pytest.approx(1.0 + 0.1 + 1.25 + 5.0)
    # the batch discount stacks with the cache multipliers
    assert _model("claude-haiku-4.5").cost(tokens, batch=True) == pytest.approx((1.0 + 0.1 + 1.25 + 5.0) / 2)


def test_every_claude_model_writes_its_cache_at_1_25x_input():
    for model in claude_models():
        assert model.cache_write_cost == pytest.approx(1.25 * model.input_cost), model.model_name


def test_other_providers_charge_cache_writes_as_plain_input():
    tokens = LLMTokens(input_tokens=1_000_000, cache_write_tokens=1_000_000)

    for model in _ALL:
        if isinstance(model, ClaudeLLMModel):
            continue
        assert model.cost(tokens) == pytest.approx(model.input_cost), model.model_name
