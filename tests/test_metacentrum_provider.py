"""The metacentrum client's structured-output routing.

Some hosted models support JSON schema and still cannot have one enforced: under grammar-constrained
decoding gemma-4 loops forever, because the grammar forbids the EOS token and its repetition bias has
nowhere else to go (vllm-project/vllm#40080). `weak_structured_output` marks those, and this is what the
flag does -- until now it was stored on the model and read nowhere at all.
"""

from __future__ import annotations

import json
import warnings

import pytest
from pydantic import BaseModel, Field

from plybench.llm import LLMCallOptions
from plybench.llm.providers.metacentrum.client import _json_body, _schema_instructions, silence_proxy_serializer_warnings
from plybench.llm.providers.metacentrum.models import metacentrum_models


class Label(BaseModel):
    code_id: str = Field(description="id of the code, or empty when none covers it")


class Answer(BaseModel):
    labels: list[Label]
    notes: str


def _model(name: str):
    return next(model for model in metacentrum_models() if model.model_name == name)


def test_the_models_that_cannot_have_a_schema_enforced_are_marked():
    # the earlier claim here was that only gemma-4 was affected. A live annotation wave disproved it:
    # qwen-3.8-27b lost 132 of 361 calls to the same `{"labels": []` + whitespace loop, so the pathology
    # is this endpoint's guided decoding, not one model's quirk
    assert _model("gemma-4").weak_structured_output
    assert _model("qwen-3.8-27b").weak_structured_output
    assert not _model("gpt-oss-120b").weak_structured_output, "the flag is per model, not a blanket switch -- unaffected models keep enforcement"
    assert _model("gemma-4").can_use_json_schema, "the model does support structured output -- enforcing it is what breaks"


def test_the_schema_is_asked_for_in_the_prompt_with_its_field_descriptions():
    """vLLM's own guide: under guided decoding "the model does not see the schema or its field
    descriptions". Asking in the prompt is what puts them back, so the descriptions must survive."""
    instructions = _schema_instructions("You label reasoning mistakes.", Answer)

    assert instructions.startswith("You label reasoning mistakes.")
    assert "id of the code, or empty when none covers it" in instructions
    assert json.loads(instructions.split("No prose, no code fence.\n")[1])["$defs"]


def test_the_prompt_asks_for_an_instance_not_the_schema():
    # gemma-4 answered {"properties": {"action": ...}} when only told to "match" the schema
    assert "do not repeat the schema itself" in _schema_instructions("Play.", Answer)


def test_a_fenced_answer_is_unwrapped_before_anything_reads_it_as_json():
    assert _json_body('```json\n{"labels": [], "notes": ""}\n```') == '{"labels": [], "notes": ""}'
    assert _json_body('```\n{"labels": []}\n```') == '{"labels": []}'
    assert _json_body('  {"labels": []}  ') == '{"labels": []}'
    assert Answer.model_validate_json(_json_body('```json\n{"labels": [], "notes": "x"}\n```')).notes == "x"


def test_the_registry_only_names_models_the_endpoint_serves():
    """`/v1/models` on 2026-09-28 had no deepseek-v4-flash; it serves deepseek-v4.1-flash. A stale name in
    the active block fails with a 400 that reads like an outage rather than a typo."""
    served = {model.model_string for model in metacentrum_models()}
    assert "deepseek-v4.1-flash" in served
    assert _model("gemma-4").model_string == "gemma4"


def test_the_proxys_unmodellable_responses_do_not_bury_the_log():
    """`responses.parse` re-serialises the payload, and e-INFRA returns one the SDK cannot model, so
    pydantic prints ~28 lines per enforced-schema call. An induction log once held 8000 of them. Only
    that message is filtered -- a serializer warning about one of our own models still has to surface."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_proxy_serializer_warnings()
        warnings.warn("Pydantic serializer warnings:\n  PydanticSerializationUnexpectedValue(...)", UserWarning, stacklevel=1)
        warnings.warn("a schema of ours really is wrong", UserWarning, stacklevel=1)

    assert [str(warning.message) for warning in caught] == ["a schema of ours really is wrong"]


def test_the_extra_body_comes_from_the_model_entry_and_follows_the_thinking_switch():
    qwen_3_8, qwen_3_5 = _model("qwen-3.8-27b"), _model("qwen-3.5")

    assert qwen_3_8.extract_extra_body(LLMCallOptions(thinking_enabled=False)) == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    assert qwen_3_8.extract_extra_body(LLMCallOptions(thinking_enabled=True)) == {}
    assert qwen_3_5.extract_extra_body(LLMCallOptions(thinking_enabled=True)) == {"chat_template_kwargs": {"thinking": True}}
    assert qwen_3_5.extract_extra_body(LLMCallOptions(thinking_enabled=False)) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_thinking_off_uses_the_switch_each_template_reads():
    # the Qwen templates ignore "thinking" and read "enable_thinking"; kimi's reads "thinking" (probe, 2026-10-09)
    for name in ("qwen-3.5", "qwen-3.8-27b", "qwen-3.8-flash-next"):
        assert _model(name).extract_extra_body(LLMCallOptions())["chat_template_kwargs"] == {"enable_thinking": False}
    assert _model("kimi-k3").extract_extra_body(LLMCallOptions()) == {"chat_template_kwargs": {"thinking": False}}


def test_models_that_reason_whatever_is_sent_refuse_thinking_off():
    for name in ("gpt-oss-120b", "deepseek-v4.1-flash"):
        with pytest.raises(ValueError, match="requires thinking"):
            _model(name).extract_params(LLMCallOptions(thinking_enabled=False))


def test_gemma_needs_an_effort_to_reason():
    with pytest.raises(ValueError, match="does not reason without a reasoning_effort"):
        _model("gemma-4").extract_params(LLMCallOptions(thinking_enabled=True))
    assert _model("gemma-4").extract_params(LLMCallOptions(thinking_enabled=True, reasoning_effort="low"))["reasoning"] == {"effort": "low"}


def test_temperature_is_sent_on_metacentrum():
    # it used to be dropped on every model, which only new_api (set on all of them) decided
    assert _model("qwen-3.5").extract_params(LLMCallOptions(thinking_enabled=True, temperature=0.6))["temperature"] == 0.6


def test_editing_a_request_body_does_not_change_the_registry_entry():
    model = _model("qwen-3.8-27b")

    model.extract_extra_body(LLMCallOptions())["chat_template_kwargs"]["enable_thinking"] = True

    assert model.extract_extra_body(LLMCallOptions()) == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
