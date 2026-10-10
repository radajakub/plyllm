# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Repository setup: uv project, Makefile, CI (lint, typecheck, tests on Python 3.12 and 3.13, build
  check), and a release script that publishes to PyPI through a GitHub Release.
- The LLM layer of plybench (`plybench.llm`, plybench 2.1.2 plus the unreleased changes up to
  plybench@e27fafc), moved here with its git history: one interface to OpenAI, Gemini, Grok, Claude,
  Mistral, Metacentrum and local HuggingFace embedding models. Import from `plyllm` instead of
  `plybench.llm`; the API is otherwise unchanged. The small helpers it used from plybench
  (`ExtendedEnum`, `MILLION`, `to_bool`, `extract_params`) are copied into `plyllm.utils`.
- `make test-models` (free check that every active model is still served) and `make test-live` (paid
  calls, asks first); `make release` runs `test-models`.
- The prompt cache key sent to OpenAI, Grok and Mistral is configurable: `LLMConfig(prompt_cache_key=...)`,
  `LLMConfig.from_env(prompt_cache_key=...)` or `PROMPT_CACHE_KEY` in the environment. It defaults to
  `PlyLLM`; in plybench it was the fixed string `PlyBench`.
- Provider SDK majors are capped at the versions the live tests ran on (`openai<3`, `anthropic<1`,
  `mistralai<3`, `google-genai<3`).

### Changed

- Metacentrum is configured only through `METACENTRUM_API_KEY` and `METACENTRUM_BASE_URL`. The old
  `OS_API_KEY` / `OS_BASE_URL` fallback is gone; with only those set, Metacentrum is left out.
