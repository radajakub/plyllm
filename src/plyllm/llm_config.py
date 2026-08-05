from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import dotenv_values

# max in-flight requests per provider; this is the ceiling that actually protects against rate limits
DEFAULT_CONCURRENCY = 10


@dataclass(frozen=True)
class OpenAIProviderConfig:
    api_key: str
    organization: str | None = None
    project: str | None = None


@dataclass(frozen=True)
class GeminiProviderConfig:
    api_key: str


@dataclass(frozen=True)
class GrokProviderConfig:
    api_key: str
    # xAI ships an OpenAI-compatible endpoint; overridable via GROK_BASE_URL
    base_url: str = "https://api.x.ai/v1"


@dataclass(frozen=True)
class ClaudeProviderConfig:
    api_key: str


@dataclass(frozen=True)
class MistralProviderConfig:
    api_key: str


@dataclass(frozen=True)
class MetacentrumProviderConfig:
    api_key: str
    base_url: str


@dataclass(frozen=True)
class HuggingFaceProviderConfig:
    # supported model aliases (see huggingface_models()) to download/verify for this environment
    models: tuple[str, ...]
    # HF_TOKEN, only needed for gated/private models
    token: str | None = None
    # explicit device override (e.g. "cuda", "cpu"); auto-detected when None
    device: str | None = None


@dataclass(frozen=True)
class LLMConfig:
    openai: OpenAIProviderConfig | None = None
    gemini: GeminiProviderConfig | None = None
    grok: GrokProviderConfig | None = None
    claude: ClaudeProviderConfig | None = None
    mistral: MistralProviderConfig | None = None
    metacentrum: MetacentrumProviderConfig | None = None
    huggingface: HuggingFaceProviderConfig | None = None
    default_concurrency: int = DEFAULT_CONCURRENCY

    @classmethod
    def from_env(cls, default_concurrency: int | None = None, huggingface_models: list[str] | None = None) -> LLMConfig:
        # merge process env with a local .env file (env takes precedence)
        values: dict[str, str | None] = {**dotenv_values(), **os.environ}

        def get(key: str) -> str | None:
            value = values.get(key)
            return value or None

        openai = None
        if (openai_key := get("OPENAI_API_KEY")) is not None:
            openai = OpenAIProviderConfig(
                api_key=openai_key,
                organization=get("OPENAI_ORGANIZATION"),
                project=get("OPENAI_PROJECT"),
            )

        gemini = None
        if (gemini_key := get("GEMINI_API_KEY")) is not None:
            gemini = GeminiProviderConfig(api_key=gemini_key)

        grok = None
        if (grok_key := get("GROK_API_KEY")) is not None:
            grok_url = get("GROK_BASE_URL")
            grok = GrokProviderConfig(api_key=grok_key, base_url=grok_url) if grok_url else GrokProviderConfig(api_key=grok_key)

        claude = None
        if (claude_key := get("CLAUDE_API_KEY")) is not None:
            claude = ClaudeProviderConfig(api_key=claude_key)

        mistral = None
        if (mistral_key := get("MISTRAL_API_KEY")) is not None:
            mistral = MistralProviderConfig(api_key=mistral_key)

        metacentrum = None
        meta_key = get("METACENTRUM_API_KEY") or get("OS_API_KEY")
        meta_url = get("METACENTRUM_BASE_URL") or get("OS_BASE_URL")
        if meta_key is not None and meta_url is not None:
            metacentrum = MetacentrumProviderConfig(api_key=meta_key, base_url=meta_url)

        huggingface = None
        if huggingface_models:
            huggingface = HuggingFaceProviderConfig(models=tuple(huggingface_models), token=get("HF_TOKEN"))

        return cls(
            openai=openai,
            gemini=gemini,
            grok=grok,
            claude=claude,
            mistral=mistral,
            metacentrum=metacentrum,
            huggingface=huggingface,
            default_concurrency=default_concurrency if default_concurrency is not None else DEFAULT_CONCURRENCY,
        )
