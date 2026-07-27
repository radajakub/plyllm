from __future__ import annotations

from typing import Any

from omniplay.llm.model import LLMModel
from omniplay.llm.options import LLMCallOptions, ReasoningEffort

_DEFAULT_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high"})


class MetacentrumLLMModel(LLMModel):
    def __init__(
        self,
        model_name: str,
        model_string: str,
        thinking: bool = False,
        new_api: bool = False,
        weak_structured_output: bool = False,
        supported_reasoning: frozenset[ReasoningEffort] | None = None,
    ) -> None:
        super().__init__(
            model_name,
            model_string,
            input_cost=0,
            output_cost=0,
            thinking=thinking,
            weak_structured_output=weak_structured_output,
            supported_reasoning=supported_reasoning,
        )
        self.new_api = new_api

    def extract_params(self, options: LLMCallOptions) -> dict[str, Any]:
        self.validate(options)

        params: dict[str, Any] = {}

        if options.reasoning_effort is not None:
            params["reasoning"] = {"effort": options.reasoning_effort}
        elif not (options.thinking_enabled or self.new_api):
            if options.temperature is not None:
                params["temperature"] = options.temperature

        if options.max_tokens:
            params["max_output_tokens"] = options.max_tokens

        return params

    def extract_extra_body(self, options: LLMCallOptions) -> dict[str, Any]:
        match self.model_name:
            case "qwen-3.5-122b" | "qwen-3.5":
                if not (options.thinking_enabled and self.thinking):
                    return {}
                return {"chat_template_kwargs": {"thinking": True}}
            case _:
                return {}


def metacentrum_models() -> list[MetacentrumLLMModel]:
    return [
        MetacentrumLLMModel("gpt-oss-120b", "gpt-oss-120b", thinking=True, new_api=True, supported_reasoning=_DEFAULT_REASONING),
        MetacentrumLLMModel("deepseek-v3.2-thinking", "deepseek-v3.2-thinking", thinking=True, new_api=True),
        MetacentrumLLMModel("qwen-3.5-122b", "qwen3.5-122b", thinking=True, new_api=True),
        MetacentrumLLMModel("glm-5.2", "glm-5.2", thinking=True, new_api=True, weak_structured_output=True),
        MetacentrumLLMModel("kimi-k2.5", "kimi-k2.5", thinking=True, new_api=True),
        MetacentrumLLMModel("qwen-3.5", "qwen3.5", thinking=True, new_api=True),
        MetacentrumLLMModel("mistral-small-4", "mistral-small-4", thinking=True, new_api=True),
    ]
