from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast, get_args

from plyllm.options import LLMCallOptions, ReasoningEffort
from plyllm.providers.providers import Provider
from plyllm.utils import extract_params, to_bool


def _reasoning_effort(value: str | None) -> ReasoningEffort | None:
    if value is None:
        return None
    if value not in get_args(ReasoningEffort):
        raise ValueError(f"Unknown reasoning_effort {value!r} -- expected one of {', '.join(get_args(ReasoningEffort))}")
    return cast("ReasoningEffort", value)


def parse_options(options_string: str) -> LLMCallOptions:
    params = extract_params(options_string)
    return LLMCallOptions(
        reasoning_effort=_reasoning_effort(params.get("reasoning_effort")),
        thinking_enabled=to_bool(params.get("thinking_enabled", False)),
        max_tokens=int(params["max_tokens"]) if "max_tokens" in params else None,
        temperature=float(params["temperature"]) if "temperature" in params else None,
    )


def options_to_string(options: LLMCallOptions) -> str:
    parts: list[str] = []
    if options.thinking_enabled:
        parts.append("thinking_enabled=True")
    if options.reasoning_effort is not None:
        parts.append(f"reasoning_effort={options.reasoning_effort}")
    if options.temperature is not None:
        parts.append(f"temperature={options.temperature}")
    if options.max_tokens is not None:
        parts.append(f"max_tokens={options.max_tokens}")
    return ",".join(parts)


@dataclass(frozen=True)
class ModelConfig:
    provider: Provider
    model_name: str
    options: LLMCallOptions = field(default_factory=LLMCallOptions)

    @classmethod
    def from_string(cls, text: str) -> ModelConfig:
        """`<provider>:<model>[:<options>]`, e.g. `openai:gpt-5.5:thinking_enabled=True,reasoning_effort=high`.
        The options are comma-separated `key=value` pairs (see `parse_options`); without them the model's
        defaults apply."""
        provider_key, _, rest = text.partition(":")
        model_name, _, options = rest.partition(":")
        provider = Provider.from_value(provider_key)
        if not isinstance(provider, Provider) or not model_name:
            raise ValueError(f"expected <provider>:<model>[:<options>], got {text!r} (providers: {', '.join(Provider.values())})")
        return cls(provider, model_name, parse_options(options))

    def to_string(self) -> str:
        options = options_to_string(self.options)
        head = f"{self.provider.value}:{self.model_name}"
        return f"{head}:{options}" if options else head

    @property
    def slug(self) -> str:
        return self.to_string().replace(":", "_").replace("=", "_").replace(",", "_").replace(".", "-")


@dataclass(frozen=True)
class EmbeddingModelConfig:
    provider: Provider
    model_name: str
