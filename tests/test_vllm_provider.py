"""The vllm provider against a fake SDK: the Chat Completions body it sends (sampling, template switches, schema)
and how it reads the answers a vLLM server returns."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI
from openai.lib._parsing._completions import type_to_response_format_param
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from plyllm import LLM, FailureKind, LLMCallError, LLMCallOptions, LLMConfig, LLMMessage, Provider, VllmProviderConfig
from plyllm.providers.openweight.models import OpenWeightLLMModel
from plyllm.providers.vllm.client import VllmLLMClient
from plyllm.providers.vllm.models import vllm_models


class _Answer(BaseModel):
    answer: int


class _FakeCompletions:
    def __init__(self, response: ChatCompletion) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> ChatCompletion:
        self.kwargs = kwargs
        return self.response


def _completion(content: str | None, refusal: str | None = None, usage: dict[str, Any] | None = None, fields: dict[str, str] | None = None) -> ChatCompletion:
    # what vLLM returns; fields the SDK does not model (reasoning, reasoning_content) are kept as extras
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen-3.8-27b-fp8",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content, "refusal": refusal, **(fields or {})}}],
            "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
    )


def _client(response: ChatCompletion) -> tuple[VllmLLMClient, _FakeCompletions]:
    completions = _FakeCompletions(response)
    return VllmLLMClient(cast(AsyncOpenAI, SimpleNamespace(chat=SimpleNamespace(completions=completions)))), completions


def _generate(client: VllmLLMClient, model: str, options: LLMCallOptions, output_schema: type[BaseModel] | None = None):
    return asyncio.run(client.generate(model, LLMMessage.system("Answer."), [LLMMessage.user("2+2?")], options, output_schema))


def _model(name: str) -> OpenWeightLLMModel:
    return next(model for model in vllm_models() if model.model_name == name)


def test_from_env_reads_the_variables_plyserve_sets(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "")
    monkeypatch.setenv("VLLM_API_KEY", "")
    assert LLMConfig.from_env().vllm is None

    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:18000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "run-key")
    assert LLMConfig.from_env().vllm == VllmProviderConfig(api_key="run-key", base_url="http://127.0.0.1:18000/v1")


def test_the_router_builds_the_vllm_client():
    llm = LLM(LLMConfig(vllm=VllmProviderConfig(api_key="test", base_url="http://127.0.0.1:1/v1")))

    assert isinstance(llm._provider_map[Provider.VLLM], VllmLLMClient)


def test_the_registry_uses_plyserves_served_names():
    # the contract with plyserve's models.toml: the name sent is the --served-model-name the server runs under
    assert {model.model_name for model in vllm_models()} == {"qwen-3.8-27b-fp8", "gemma-4-31b-fp8"}
    assert all(model.model_string == model.model_name and model.input_cost == model.output_cost == 0 for model in vllm_models())


def test_a_model_without_sampling_for_a_mode_it_runs_in_is_refused():
    with pytest.raises(ValueError, match="needs sampling_thinking"):
        OpenWeightLLMModel("m", sampling_no_thinking={"temperature": 0.7}, thinking=True)
    with pytest.raises(ValueError, match="needs sampling_no_thinking"):
        OpenWeightLLMModel("m", sampling_thinking={"temperature": 1.0}, thinking=True)
    OpenWeightLLMModel("m", sampling_thinking={"temperature": 1.0}, thinking=True, thinking_only=True)


def test_thinking_off_sends_the_instruct_sampling_and_switches_the_template_off():
    client, completions = _client(_completion("4"))

    _generate(client, "qwen-3.8-27b-fp8", LLMCallOptions(max_tokens=64))

    assert completions.kwargs == {
        "model": "qwen-3.8-27b-fp8",
        "messages": [{"role": "system", "content": "Answer."}, {"role": "user", "content": "2+2?"}],
        "max_tokens": 64,
        "extra_body": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def test_thinking_on_sends_the_thinking_sampling_and_the_effort_as_a_template_variable():
    client, completions = _client(_completion("4"))

    _generate(client, "qwen-3.8-27b-fp8", LLMCallOptions(thinking_enabled=True, reasoning_effort="low"))

    body = completions.kwargs["extra_body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}
    assert (body["temperature"], body["top_p"], body["presence_penalty"]) == (1.0, 0.95, 0.0)
    assert "max_tokens" not in completions.kwargs


def test_gemma_has_to_be_told_to_think_and_takes_no_effort():
    # its template does not think unless enable_thinking is set
    assert _model("gemma-4-31b-fp8").extract_params(LLMCallOptions(thinking_enabled=True))["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    with pytest.raises(ValueError, match="does not accept a reasoning_effort"):
        _model("gemma-4-31b-fp8").extract_params(LLMCallOptions(thinking_enabled=True, reasoning_effort="low"))


def test_a_temperature_overrides_only_the_temperature():
    body = _model("gemma-4-31b-fp8").extract_params(LLMCallOptions(temperature=0.0))["extra_body"]

    assert body["temperature"] == 0.0 and body["top_k"] == 64 and body["top_p"] == 0.95


def test_editing_a_request_body_does_not_change_the_registry_entry():
    model = _model("qwen-3.8-27b-fp8")

    body = model.extract_params(LLMCallOptions())["extra_body"]
    body["temperature"] = 2.0
    body["chat_template_kwargs"]["enable_thinking"] = True

    assert model.extract_params(LLMCallOptions())["extra_body"]["temperature"] == 0.7
    assert model.extract_params(LLMCallOptions())["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_reasoning_field_is_read_under_either_name():
    for field in ("reasoning", "reasoning_content"):
        client, _ = _client(_completion("4", fields={field: "two and two"}))

        response = _generate(client, "qwen-3.8-27b-fp8", LLMCallOptions(thinking_enabled=True))

        assert (response.output_text, response.reasoning) == ("4", ["two and two"]), field


def test_both_reasoning_fields_with_the_same_text_give_one_trace():
    client, _ = _client(_completion("4", fields={"reasoning": "two and two", "reasoning_content": "two and two"}))

    assert _generate(client, "qwen-3.8-27b-fp8", LLMCallOptions(thinking_enabled=True)).reasoning == ["two and two"]


def test_an_inline_think_block_is_split_off_when_the_server_has_no_reasoning_parser():
    client, _ = _client(_completion("<think>\ntwo and two\n</think>\n\n4"))

    response = _generate(client, "qwen-3.8-27b-fp8", LLMCallOptions(thinking_enabled=True))

    assert (response.output_text, response.reasoning) == ("4", ["two and two"])


def test_tokens_are_read_from_the_usage_and_its_details():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 64},
        "completion_tokens_details": {"reasoning_tokens": 15},
    }
    client, _ = _client(_completion("4", usage=usage))

    tokens = _generate(client, "qwen-3.8-27b-fp8", LLMCallOptions()).tokens

    assert (tokens.input_tokens, tokens.output_tokens, tokens.cached_input_tokens, tokens.reasoning_tokens) == (100, 20, 64, 15)
    assert client._tokens_of(_completion("4", usage=usage)) == 120


def test_the_schema_is_asked_for_in_the_prompt_and_a_fenced_answer_is_unwrapped():
    client, completions = _client(_completion('```json\n{"answer": 4}\n```'))

    response = _generate(client, "qwen-3.8-27b-fp8", LLMCallOptions(), _Answer)

    assert "response_format" not in completions.kwargs
    system = completions.kwargs["messages"][0]["content"]
    assert system.startswith("Answer.") and "do not repeat the schema itself" in system
    assert response.resolve_structured_output(_Answer).answer == 4


def test_a_model_marked_for_enforcement_sends_the_strict_schema():
    # also pins the private `type_to_response_format_param` import against SDK upgrades
    client, completions = _client(_completion('{"answer": 4}'))
    client.resolve_model("gemma-4-31b-fp8").weak_structured_output = False

    response = _generate(client, "gemma-4-31b-fp8", LLMCallOptions(), _Answer)

    assert completions.kwargs["response_format"] == type_to_response_format_param(_Answer)
    assert completions.kwargs["response_format"]["json_schema"]["strict"] is True
    assert completions.kwargs["messages"][0]["content"] == "Answer."
    assert response.resolve_structured_output(_Answer).answer == 4


def test_an_answer_off_the_schema_keeps_its_text():
    client, _ = _client(_completion("four"))

    with pytest.raises(LLMCallError) as caught:
        _generate(client, "qwen-3.8-27b-fp8", LLMCallOptions(), _Answer)

    assert caught.value.kind is FailureKind.UNPARSEABLE and caught.value.response is not None and caught.value.response.output_text == "four"


def test_a_refusal_is_raised_with_the_answer_attached():
    client, _ = _client(_completion(None, refusal="I cannot help with that."))

    with pytest.raises(LLMCallError) as caught:
        _generate(client, "qwen-3.8-27b-fp8", LLMCallOptions())

    assert caught.value.kind is FailureKind.REFUSAL and caught.value.response is not None
