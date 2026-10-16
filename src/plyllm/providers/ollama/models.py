from __future__ import annotations

from typing import Any

from plyllm.options import LLMCallOptions, ReasoningEffort
from plyllm.providers.openweight.models import OpenWeightLLMModel, Sampling, SchemaDelivery

# what Ollama's /v1/chat/completions takes of the sampling (docs at v0.32.15). top_k, min_p and repeat_penalty are not
# among them: they come from the model's Modelfile, so a registry entry names the Modelfile it was checked against
_SAMPLING_FIELDS = frozenset({"temperature", "top_p", "presence_penalty", "frequency_penalty"})


class OllamaLLMModel(OpenWeightLLMModel):
    # a model pulled into a local Ollama, called through its OpenAI-compatible endpoint. Its tag holds colons, which
    # a model config string cannot, so the alias is separate from the tag
    def __init__(
        self,
        model_name: str,
        tag: str,
        sampling_thinking: Sampling | None = None,
        sampling_no_thinking: Sampling | None = None,
        thinking: bool = False,
        thinking_only: bool = False,
        schema_delivery: SchemaDelivery = SchemaDelivery.ENFORCED,
        supported_reasoning: frozenset[ReasoningEffort] | None = None,
        retired: bool = False,
    ) -> None:
        for sampling in (sampling_thinking, sampling_no_thinking):
            ignored = set(sampling or {}) - _SAMPLING_FIELDS
            if ignored:
                raise ValueError(f"Model {model_name}: Ollama's /v1 endpoint ignores {sorted(ignored)}; they come from the Modelfile")
        super().__init__(
            model_name,
            sampling_thinking,
            sampling_no_thinking,
            thinking=thinking,
            thinking_only=thinking_only,
            schema_delivery=schema_delivery,
            supported_reasoning=supported_reasoning,
            retired=retired,
            # a model with effort levels always gets the caller's, never one chosen here
            needs_effort_to_think=supported_reasoning is not None,
            served_name=tag,
        )

    def _thinking_fields(self, options: LLMCallOptions, thinking: bool) -> dict[str, Any]:
        # Ollama switches thinking through reasoning_effort, with "none" for off. Any level turns it on, so a model
        # without levels gets "medium", which only means on
        if not thinking:
            return {"reasoning_effort": "none"}
        return {"reasoning_effort": options.reasoning_effort or "medium"}


# Gemma 4 card: temperature 1.0, top_p 0.95, top_k 64 in every mode. Penalties are not on the card and are sent at
# their neutral values
_GEMMA4: Sampling = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0, "frequency_penalty": 0.0}
# Qwen3.8 card, the fields /v1 takes: one set for thinking, another for instruct mode, as on vLLM. top_k 20 (both modes)
# comes from the Modelfile plyserve writes from its models.toml
_QWEN3_8_THINKING: Sampling = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0, "frequency_penalty": 0.0}
_QWEN3_8_NO_THINKING: Sampling = {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5, "frequency_penalty": 0.0}


def ollama_models() -> list[OllamaLLMModel]:
    # Tags as `ollama list` shows them. A tag is mutable: a later pull may give other weights under the same name, so
    # each entry notes the digest it was checked against. Probed 2026-10-12.
    # The schema is asked for in the prompt, as for the same family on vLLM
    return [
        # digest c6eb396dbd59 (8.0B, Q4_K_M), checked 2026-10-12; its Modelfile sets top_k 64, as the card does.
        # Probe on Ollama 0.32.15: effort "none" gives no reasoning, the reasoning comes in its own field and the
        # schema in the prompt gives plain JSON. With thinking on it gave no reasoning for "2 + 2" but some with
        # the schema; whether "medium" always turns thinking on is not yet checked
        OllamaLLMModel("gemma-4-e4b", "gemma4:e4b", sampling_thinking=_GEMMA4, sampling_no_thinking=_GEMMA4, thinking=True, schema_delivery=SchemaDelivery.PROMPT),
        # Served by plyserve (backend "ollama"), not pulled: plyserve creates the tag from a pinned GGUF under its served
        # name and puts a key-checking proxy in front, so a client sets OLLAMA_BASE_URL and OLLAMA_API_KEY to the
        # endpoint and key plyserve gives. The same model as vllm's qwen-3.8-27b-fp8 in another 8-bit format, to
        # compare the engines, so the same schema delivery. Not yet probed on Ollama: whether reasoning_effort levels
        # reach the template (until then thinking is only on or off) and how the enforced schema works with thinking
        OllamaLLMModel(
            "qwen-3.8-27b-q8",
            "qwen-3.8-27b-q8",
            sampling_thinking=_QWEN3_8_THINKING,
            sampling_no_thinking=_QWEN3_8_NO_THINKING,
            thinking=True,
            schema_delivery=SchemaDelivery.PROMPT_AND_ENFORCED,
        ),
    ]
