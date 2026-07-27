from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import dotenv_values


@dataclass(frozen=True)
class OpenAIProviderConfig:
    api_key: str
    organization: str | None = None
    project: str | None = None


@dataclass(frozen=True)
class GeminiProviderConfig:
    api_key: str


@dataclass(frozen=True)
class MetacentrumProviderConfig:
    api_key: str
    base_url: str


@dataclass(frozen=True)
class LLMConfig:
    openai: OpenAIProviderConfig | None = None
    gemini: GeminiProviderConfig | None = None
    metacentrum: MetacentrumProviderConfig | None = None
    default_concurrency: int = 10

    @classmethod
    def from_env(cls, default_concurrency: int = 10) -> LLMConfig:
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

        metacentrum = None
        meta_key = get("METACENTRUM_API_KEY") or get("OS_API_KEY")
        meta_url = get("METACENTRUM_BASE_URL") or get("OS_BASE_URL")
        if meta_key is not None and meta_url is not None:
            metacentrum = MetacentrumProviderConfig(api_key=meta_key, base_url=meta_url)

        return cls(
            openai=openai,
            gemini=gemini,
            metacentrum=metacentrum,
            default_concurrency=default_concurrency,
        )
