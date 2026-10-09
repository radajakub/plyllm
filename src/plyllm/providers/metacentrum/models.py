from __future__ import annotations

import copy
from typing import Any

from plybench.llm.model import LLMModel
from plybench.llm.options import LLMCallOptions, ReasoningEffort

_DEFAULT_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high"})
_KIMI_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "high", "max"})
_QWEN3_8_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "xhigh"})
# The Qwen templates read enable_thinking: on 2026-10-09 qwen-3.8-27b still reasoned with {"thinking": false}
# and stopped with {"enable_thinking": false}, as did qwen-3.5 and qwen-3.8-flash-next. The "thinking": true sent
# with thinking on has no effect (they reason by default); it stays so requests match the recorded runs
_QWEN3_5_THINKING: dict[str, Any] = {"chat_template_kwargs": {"thinking": True}}
_QWEN_NO_THINKING: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
_QWEN3_8_NO_THINKING: dict[str, Any] = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
# kimi-k3 stopped reasoning with this on 2026-10-09
_KIMI_NO_THINKING: dict[str, Any] = {"chat_template_kwargs": {"thinking": False}}


class MetacentrumLLMModel(LLMModel):
    def __init__(
        self,
        model_name: str,
        model_string: str,
        thinking: bool = False,
        thinking_only: bool = False,
        needs_effort_to_think: bool = False,
        weak_structured_output: bool = False,
        supported_reasoning: frozenset[ReasoningEffort] | None = None,
        retired: bool = False,
        extra_body_thinking: dict[str, Any] | None = None,
        extra_body_no_thinking: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model_name,
            model_string,
            input_cost=0,
            output_cost=0,
            thinking=thinking,
            thinking_only=thinking_only,
            weak_structured_output=weak_structured_output,
            supported_reasoning=supported_reasoning,
            retired=retired,
            needs_effort_to_think=needs_effort_to_think,
        )
        # vLLM fields outside the Responses API (chat template switches, sampling), sent with or without thinking
        self.extra_body_thinking = extra_body_thinking or {}
        self.extra_body_no_thinking = extra_body_no_thinking or {}

    def extract_params(self, options: LLMCallOptions) -> dict[str, Any]:
        self.validate(options)

        params: dict[str, Any] = {}

        if options.reasoning_effort is not None:
            params["reasoning"] = {"effort": options.reasoning_effort}

        if options.temperature is not None:
            params["temperature"] = options.temperature

        if options.max_tokens:
            params["max_output_tokens"] = options.max_tokens

        return params

    def extract_extra_body(self, options: LLMCallOptions) -> dict[str, Any]:
        # a copy, so a caller that edits the request cannot change the registry entry
        body = self.extra_body_thinking if options.thinking_enabled and self.thinking else self.extra_body_no_thinking
        return copy.deepcopy(body)


def metacentrum_models() -> list[MetacentrumLLMModel]:
    # e-INFRA replaces hosted models without notice; the entries below are the ones it currently
    # serves. Retired models are kept so existing configs and recorded results still resolve.
    # Thinking flags follow a probe on 2026-10-09 (reasoning tokens with thinking on / off): gpt-oss-120b and
    # deepseek-v4.1-flash reason whatever is sent, gemma-4 reasons only when given an effort, and
    # mistral-medium-3.5 returned no reasoning with thinking on
    return [
        MetacentrumLLMModel("gpt-oss-120b", "gpt-oss-120b", thinking=True, thinking_only=True, supported_reasoning=_DEFAULT_REASONING),
        MetacentrumLLMModel("deepseek-v4.1-flash", "deepseek-v4.1-flash", thinking=True, thinking_only=True),
        MetacentrumLLMModel("glm-5.3", "glm-5.3", thinking=True, weak_structured_output=True),
        MetacentrumLLMModel("kimi-k3", "kimi-k3", thinking=True, supported_reasoning=_KIMI_REASONING, extra_body_no_thinking=_KIMI_NO_THINKING),
        MetacentrumLLMModel("qwen-3.5", "qwen3.5", thinking=True, extra_body_thinking=_QWEN3_5_THINKING, extra_body_no_thinking=_QWEN_NO_THINKING),
        MetacentrumLLMModel(
            "qwen-3.8-27b",
            "qwen3.8-27b",
            thinking=True,
            weak_structured_output=True,
            supported_reasoning=_QWEN3_8_REASONING,
            extra_body_no_thinking=_QWEN3_8_NO_THINKING,
        ),
        MetacentrumLLMModel("qwen-3.8-flash-next", "qwen3.8-flash-next", thinking=True, extra_body_no_thinking=_QWEN_NO_THINKING),
        MetacentrumLLMModel("mistral-medium-3.5", "mistral-medium-3.5"),
        MetacentrumLLMModel("gemma-4", "gemma4", thinking=True, needs_effort_to_think=True, weak_structured_output=True, supported_reasoning=_DEFAULT_REASONING),
        # no longer served by e-INFRA as of 2026-09-24; kept so recorded results still load
        MetacentrumLLMModel("deepseek-v4-flash", "deepseek-v4-flash", thinking=True, retired=True),
        MetacentrumLLMModel("deepseek-v3.2-thinking", "deepseek-v3.2-thinking", thinking=True, retired=True),
        MetacentrumLLMModel("qwen-3.5-122b", "qwen3.5-122b", thinking=True, retired=True, extra_body_thinking=_QWEN3_5_THINKING),
        MetacentrumLLMModel("glm-5.2", "glm-5.2", thinking=True, weak_structured_output=True, retired=True),
        MetacentrumLLMModel("mistral-small-4", "mistral-small-4", thinking=True, retired=True),
    ]
