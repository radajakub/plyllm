from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import dotenv_values

# max in-flight requests per provider, unless the provider sets its own; this is the ceiling that actually
# protects against rate limits
DEFAULT_CONCURRENCY = 10
# seconds one request may take before the SDK gives up and our retry layer takes over; long enough for
# a slow thinking trace, short enough that a stalled connection does not hang a run forever
DEFAULT_TIMEOUT = 600.0
# sent with every request to the providers that take one (OpenAI, Grok, Mistral): requests that share a key
# and a prompt prefix are routed to the same cache, so a shared system prompt is read from it
DEFAULT_PROMPT_CACHE_KEY = "PlyLLM"


@dataclass(frozen=True)
class OpenAIProviderConfig:
    api_key: str
    organization: str | None = None
    project: str | None = None
    timeout: float = DEFAULT_TIMEOUT


@dataclass(frozen=True)
class GeminiProviderConfig:
    api_key: str
    timeout: float = DEFAULT_TIMEOUT


@dataclass(frozen=True)
class GrokProviderConfig:
    api_key: str
    # xAI ships an OpenAI-compatible endpoint; overridable via GROK_BASE_URL
    base_url: str = "https://api.x.ai/v1"
    timeout: float = DEFAULT_TIMEOUT


@dataclass(frozen=True)
class ClaudeProviderConfig:
    api_key: str
    timeout: float = DEFAULT_TIMEOUT


@dataclass(frozen=True)
class MistralProviderConfig:
    api_key: str
    timeout: float = DEFAULT_TIMEOUT


@dataclass(frozen=True)
class MetacentrumProviderConfig:
    api_key: str
    base_url: str
    timeout: float = 1200.0
    max_retries: int = 0
    retries: int = 3


@dataclass(frozen=True)
class VllmProviderConfig:
    # a vLLM server we run, e.g. through plyserve, which sets VLLM_BASE_URL and VLLM_API_KEY for its client
    api_key: str
    base_url: str
    # long: requests queue on one GPU, and a thinking trace can take minutes once there
    timeout: float = 1200.0
    # no SDK retries inside ours, so a stopped server fails a call in minutes rather than hours
    max_retries: int = 0
    retries: int = 3


@dataclass(frozen=True)
class HuggingFaceProviderConfig:
    # supported model aliases (see huggingface_embedding_models()) to download/verify for this environment
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
    vllm: VllmProviderConfig | None = None
    huggingface: HuggingFaceProviderConfig | None = None
    # None lets each provider use its own default (LLMClient.default_concurrency)
    default_concurrency: int | None = None
    prompt_cache_key: str = DEFAULT_PROMPT_CACHE_KEY

    @classmethod
    def from_env(cls, default_concurrency: int | None = None, huggingface_models: list[str] | None = None, prompt_cache_key: str | None = None) -> LLMConfig:
        # merge process env with a local .env file (env takes precedence)
        values: dict[str, str | None] = {**dotenv_values(), **os.environ}

        def get(key: str) -> str | None:
            value = values.get(key)
            return value or None

        def timeout(prefix: str) -> float:
            return float(get(f"{prefix}_TIMEOUT") or DEFAULT_TIMEOUT)

        openai = None
        if (openai_key := get("OPENAI_API_KEY")) is not None:
            openai = OpenAIProviderConfig(
                api_key=openai_key,
                organization=get("OPENAI_ORGANIZATION"),
                project=get("OPENAI_PROJECT"),
                timeout=timeout("OPENAI"),
            )

        gemini = None
        if (gemini_key := get("GEMINI_API_KEY")) is not None:
            gemini = GeminiProviderConfig(api_key=gemini_key, timeout=timeout("GEMINI"))

        grok = None
        if (grok_key := get("GROK_API_KEY")) is not None:
            grok_url = get("GROK_BASE_URL")
            grok_timeout = timeout("GROK")
            grok = GrokProviderConfig(api_key=grok_key, base_url=grok_url, timeout=grok_timeout) if grok_url else GrokProviderConfig(api_key=grok_key, timeout=grok_timeout)

        claude = None
        if (claude_key := get("CLAUDE_API_KEY")) is not None:
            claude = ClaudeProviderConfig(api_key=claude_key, timeout=timeout("CLAUDE"))

        mistral = None
        if (mistral_key := get("MISTRAL_API_KEY")) is not None:
            mistral = MistralProviderConfig(api_key=mistral_key, timeout=timeout("MISTRAL"))

        metacentrum = None
        meta_key = get("METACENTRUM_API_KEY")
        meta_url = get("METACENTRUM_BASE_URL")
        if meta_key is not None and meta_url is not None:
            metacentrum = MetacentrumProviderConfig(
                api_key=meta_key,
                base_url=meta_url,
                timeout=float(get("METACENTRUM_TIMEOUT") or 1200.0),
                max_retries=int(get("METACENTRUM_MAX_RETRIES") or 0),
                retries=int(get("METACENTRUM_RETRIES") or 3),
            )

        vllm = None
        vllm_key = get("VLLM_API_KEY")
        vllm_url = get("VLLM_BASE_URL")
        if vllm_key is not None and vllm_url is not None:
            vllm = VllmProviderConfig(
                api_key=vllm_key,
                base_url=vllm_url,
                timeout=float(get("VLLM_TIMEOUT") or 1200.0),
                max_retries=int(get("VLLM_MAX_RETRIES") or 0),
                retries=int(get("VLLM_RETRIES") or 3),
            )

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
            vllm=vllm,
            huggingface=huggingface,
            default_concurrency=default_concurrency,
            # an argument wins over PROMPT_CACHE_KEY in the environment
            prompt_cache_key=prompt_cache_key or get("PROMPT_CACHE_KEY") or DEFAULT_PROMPT_CACHE_KEY,
        )
