# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- OpenAI Decisions API: `llm.decide(DecisionModelConfig(Provider.OPENAI, "gpt-6-luna"), text, questions)`
  answers `Predicate`, `Choice` and `Score` questions with probabilities instead of generated text, as a
  `DecisionResponse`. A refused question comes back as a `RefusedAnswer` beside the other answers. Text input
  only. `calculate_decision_cost` bills input tokens only. See the README, "Decisions API".
- Batch API for OpenAI, Claude, Gemini and Mistral: `llm.batch(model, requests, store)` submits new requests,
  collects ended jobs and reports the rest as pending, so a script can be rerun until every answer is in.
  `JsonBatchStore` keeps the running jobs in one JSON file. Requests carry exactly the body `generate` sends
  and their answers are parsed the same way. See the README, "Batch API".
- `LLMResponse.batch` marks an answer from a batch, to be priced with `calculate_cost(..., batch=True)`.
- `FailureKind.EXPIRED` (`batch_expired`) and `FailureKind.CANCELLED` (`batch_cancelled`) for batch
  requests that never ran.

### Changed

- `LLMClient` takes the SDK response type as a second type parameter, and `generate` is now one method in
  the base class built from per-provider `_request`, `_send`, `_tokens_of` and `_parse`. A subclass outside
  this package must implement these instead of `generate`.

- Provider SDKs raised to new majors: `openai>=3.28.0,<4`, `anthropic>=1.13.0,<2`, `mistralai>=3.2.0,<4`.
  All three now use httpx2 instead of httpx.
- Claude no longer accepts `temperature` on any model; a call with one raises `ValueError` before any
  request is sent. anthropic 1.0 removed the argument from `messages.create`. Sonnet 4.6 and Haiku 4.5
  accepted it before.

### Fixed

- Mistral: timeouts and connection failures are retried and classified again. mistralai 3 raises httpx2
  errors, which the client did not catch, so they failed at once as `other`.

## [1.0.0] - 2026-10-10

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
