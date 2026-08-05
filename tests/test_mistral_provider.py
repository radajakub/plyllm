import asyncio

import httpx
import pytest
from mistralai.client.errors import MistralError
from mistralai.client.models import AssistantMessage, ChatCompletionChoice, ChatCompletionResponse, TextChunk, ThinkChunk, UsageInfo
from pydantic import BaseModel, ValidationError

from plybench.llm import LLM, LLMCallOptions, LLMConfig, LLMMessage, MistralProviderConfig, ModelLimits, Provider
from plybench.llm.providers.mistral.client import MistralLLMClient, _is_retryable, completion_tokens, split_content
from plybench.llm.providers.mistral.models import MistralLLMModel, mistral_models
from plybench.llm.rate_limit import ModelGate, NoLimits


class _Move(BaseModel):
    move: int


def _model(model_name: str) -> MistralLLMModel:
    return next(model for model in mistral_models() if model.model_name == model_name)


def _http_error(status_code: int) -> MistralError:
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    return MistralError("boom", httpx.Response(status_code=status_code, request=request))


def test_from_env_builds_mistral_config_from_api_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "mk-test")

    config = LLMConfig.from_env()

    assert config.mistral is not None
    assert config.mistral.api_key == "mk-test"


def test_router_wires_mistral_when_configured():
    llm = LLM(LLMConfig(mistral=MistralProviderConfig(api_key="mk-test")))

    assert llm.available_providers == [Provider.MISTRAL]
    assert llm.resolve_model(Provider.MISTRAL, "mistral-small-4").model_string == "mistral-small-2603"


def test_thinking_enabled_sends_the_requested_effort():
    params = _model("mistral-small-4").extract_params(LLMCallOptions(thinking_enabled=True, reasoning_effort="high", max_tokens=4000))

    assert params == {"reasoning_effort": "high", "max_tokens": 4000}


def test_thinking_disabled_sends_effort_none():
    params = _model("mistral-medium-3.5").extract_params(LLMCallOptions())

    assert params["reasoning_effort"] == "none"


def test_thinking_without_explicit_effort_defaults_to_high():
    params = _model("mistral-medium-3.5").extract_params(LLMCallOptions(thinking_enabled=True))

    assert params["reasoning_effort"] == "high"


def test_temperature_is_forwarded_when_set():
    params = _model("mistral-small-4").extract_params(LLMCallOptions(temperature=0.3))

    assert params["temperature"] == 0.3


def test_unsupported_reasoning_effort_is_rejected():
    with pytest.raises(ValueError):
        _model("mistral-small-4").extract_params(LLMCallOptions(thinking_enabled=True, reasoning_effort="max"))


def test_models_ship_without_quotas():
    # quotas are account-specific, so an install must not assume any (see scripts/_shared.py)
    assert all(model.limits is None for model in mistral_models())


def test_unlimited_models_get_pass_through_gates():
    llm = LLM(LLMConfig(mistral=MistralProviderConfig(api_key="mk-test")))
    client = llm._provider_map[Provider.MISTRAL]

    assert isinstance(client.gate(client.resolve_model("mistral-small-4")), NoLimits)


def test_set_model_limits_installs_a_gate():
    llm = LLM(LLMConfig(mistral=MistralProviderConfig(api_key="mk-test")))
    client = llm._provider_map[Provider.MISTRAL]

    llm.set_model_limits(Provider.MISTRAL, "mistral-small-4", ModelLimits(tpm=100_000))

    assert isinstance(client.gate(client.resolve_model("mistral-small-4")), ModelGate)
    assert client.resolve_model("mistral-small-4").limits == ModelLimits(tpm=100_000)


def test_set_model_limits_can_clear_a_quota():
    llm = LLM(LLMConfig(mistral=MistralProviderConfig(api_key="mk-test")))
    client = llm._provider_map[Provider.MISTRAL]

    llm.set_model_limits(Provider.MISTRAL, "mistral-small-4", ModelLimits(tpm=100_000))
    llm.set_model_limits(Provider.MISTRAL, "mistral-small-4", None)

    assert isinstance(client.gate(client.resolve_model("mistral-small-4")), NoLimits)


def test_split_content_separates_thinking_from_output():
    message = AssistantMessage(
        content=[
            ThinkChunk(thinking=[TextChunk(text="considering the centre")]),
            TextChunk(text="4"),
        ]
    )

    reasoning, output = split_content(message)

    assert reasoning == ["considering the centre"]
    assert output == "4"


def test_split_content_handles_plain_string_content():
    assert split_content(AssistantMessage(content="4")) == ([], "4")


def test_split_content_handles_a_missing_message():
    assert split_content(None) == ([], "")


def test_completion_tokens_reads_cached_and_reasoning_extras():
    usage = UsageInfo.model_validate(
        {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
            "prompt_tokens_details": {"cached_tokens": 60},
            "completion_tokens_details": {"reasoning_tokens": 25},
        }
    )

    tokens = completion_tokens(usage)

    assert tokens.input_tokens == 100
    assert tokens.cached_input_tokens == 60
    assert tokens.output_tokens == 40
    assert tokens.reasoning_tokens == 25


def test_completion_tokens_without_extras_defaults_to_zero():
    tokens = completion_tokens(UsageInfo.model_validate({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}))

    assert tokens.cached_input_tokens == 0
    assert tokens.reasoning_tokens == 0


def test_only_throttling_and_server_errors_are_retried():
    assert _is_retryable(_http_error(429))
    assert _is_retryable(_http_error(503))
    assert not _is_retryable(_http_error(400))
    assert not _is_retryable(_http_error(401))
    assert not _is_retryable(_http_error(422))
    # transport failures carry no status and are always worth another attempt
    assert _is_retryable(RuntimeError("connection reset"))


class _FakeChat:
    def __init__(self, response: ChatCompletionResponse) -> None:
        self._response = response
        self.kwargs: dict = {}

    async def complete_async(self, **kwargs):
        self.kwargs = kwargs
        return self._response


class _FakeSDK:
    def __init__(self, response: ChatCompletionResponse) -> None:
        self.chat = _FakeChat(response)


def _response(message: AssistantMessage) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="cmpl-1",
        object="chat.completion",
        model="mistral-small-2603",
        created=0,
        usage=UsageInfo.model_validate({"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}),
        choices=[ChatCompletionChoice(index=0, finish_reason="stop", message=message)],
    )


def _client(message: AssistantMessage) -> tuple[MistralLLMClient, _FakeSDK]:
    sdk = _FakeSDK(_response(message))
    return MistralLLMClient(sdk, concurrency=4), sdk


def test_generate_puts_the_system_prompt_at_the_head_of_the_messages():
    client, sdk = _client(AssistantMessage(content="4"))

    asyncio.run(client.generate("mistral-small-4", LLMMessage.system("rules"), [LLMMessage.user("your move")], LLMCallOptions()))

    assert sdk.chat.kwargs["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "your move"},
    ]
    assert sdk.chat.kwargs["model"] == "mistral-small-2603"
    assert sdk.chat.kwargs["prompt_cache_key"] == "PlyBench"


def test_generate_surfaces_the_thinking_trace_as_reasoning():
    message = AssistantMessage(content=[ThinkChunk(thinking=[TextChunk(text="centre is strongest")]), TextChunk(text="4")])
    client, _ = _client(message)

    response = asyncio.run(client.generate("mistral-small-4", LLMMessage.system("rules"), [LLMMessage.user("go")], LLMCallOptions(thinking_enabled=True, reasoning_effort="high")))

    assert response.reasoning == ["centre is strongest"]
    assert response.output_text == "4"
    assert response.tokens.input_tokens == 100


def test_generate_validates_structured_output_from_chunked_content():
    # the SDK's own parse helper would raise on this chunk list, which is why we validate ourselves
    message = AssistantMessage(content=[ThinkChunk(thinking=[TextChunk(text="thinking")]), TextChunk(text='{"move": 4}')])
    client, sdk = _client(message)

    response = asyncio.run(client.generate("mistral-small-4", LLMMessage.system("rules"), [LLMMessage.user("go")], LLMCallOptions(thinking_enabled=True), output_schema=_Move))

    assert sdk.chat.kwargs["response_format"]["type"] == "json_schema"
    assert response.resolve_structured_output(_Move) == _Move(move=4)


def test_generate_rejects_output_that_violates_the_schema():
    client, _ = _client(AssistantMessage(content=[TextChunk(text='{"move": "centre"}')]))

    with pytest.raises(ValidationError):
        asyncio.run(client.generate("mistral-small-4", LLMMessage.system("rules"), [LLMMessage.user("go")], LLMCallOptions(), output_schema=_Move))


def test_embeddings_are_not_supported():
    llm = LLM(LLMConfig(mistral=MistralProviderConfig(api_key="mk-test")))

    with pytest.raises(NotImplementedError):
        asyncio.run(llm.embed(Provider.MISTRAL, "mistral-small-4", ["hello"]))
