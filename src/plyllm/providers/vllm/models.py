from __future__ import annotations

from plyllm.options import ReasoningEffort
from plyllm.providers.openweight.models import OpenWeightLLMModel, Sampling

# Sampling from the model cards at the revisions plyserve's models.toml pins (read 2026-10-11). Parameters a card does
# not name are sent at their neutral values, so nothing is left to the server.
# Qwen3.8 card: one set for thinking, another for instruct (non-thinking) mode
_QWEN3_8_THINKING: Sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
_QWEN3_8_NO_THINKING: Sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
# Gemma 4 card: "standardized sampling configuration across all use cases"
_GEMMA4: Sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}

# the Qwen3.8 template thinks unless told otherwise, and reads reasoning_effort (xhigh by default)
_QWEN3_8_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "xhigh"})
# the Gemma 4 template does not think unless told to, and has no effort
_THINKING = {"enable_thinking": True}
_NO_THINKING = {"enable_thinking": False}


def vllm_models() -> list[OpenWeightLLMModel]:
    # Served names exactly as in plyserve's models.toml: the server is started under that name, so a model added
    # there is added here with the same string. Flags are from the chat templates, not from a probe of our server.
    # Gemma asks for the schema in the prompt: enforced, it made both models loop on Metacentrum's vLLM
    # (vllm-project/vllm#40080). Qwen has it enforced, on trial on our vLLM: in the prompt it lost 1-4% of answers as
    # unparseable on real traces (2026-10-13)
    return [
        OpenWeightLLMModel(
            "qwen-3.8-27b-fp8",
            sampling_thinking=_QWEN3_8_THINKING,
            sampling_no_thinking=_QWEN3_8_NO_THINKING,
            template_thinking=_THINKING,
            template_no_thinking=_NO_THINKING,
            thinking=True,
            weak_structured_output=False,
            supported_reasoning=_QWEN3_8_REASONING,
        ),
        OpenWeightLLMModel(
            "gemma-4-31b-fp8",
            sampling_thinking=_GEMMA4,
            sampling_no_thinking=_GEMMA4,
            template_thinking=_THINKING,
            template_no_thinking=_NO_THINKING,
            thinking=True,
            weak_structured_output=True,
        ),
    ]
