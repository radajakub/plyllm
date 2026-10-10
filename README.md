# plyllm

Package that unifies communication with multiple LLM providers and models

## Installation

```bash
pip install "plyllm[all]"
```

Each provider SDK is an optional extra, so install only the ones you use: `openai`, `gemini`, `grok`,
`anthropic`, `mistral`, `metacentrum` (an OpenAI-compatible endpoint), `huggingface` (local embedding
models; pulls in torch).

## Quickstart

```python
import asyncio

from plyllm import LLM, LLMConfig, LLMMessage, ModelConfig


async def main() -> None:
    llm = LLM(LLMConfig.from_env())  # reads provider keys from the environment (see Configuration)
    try:
        model = ModelConfig.from_string("openai:gpt-5.5:thinking_enabled=True,reasoning_effort=low")
        response = await llm.generate(model, LLMMessage.system("Answer with one number."), [LLMMessage.user("What is 2+2?")])
        print(response.output_text, llm.calculate_cost(model, response.tokens))
    finally:
        await llm.aclose()


asyncio.run(main())
```

Pass a pydantic model as `output_schema` to `generate` for structured output; `response.resolve_structured_output()`
returns the parsed object.

### Model config strings

A model is addressed by `<provider>:<model>[:<options>]`, e.g. `claude:claude-opus-5.5:thinking_enabled=True,reasoning_effort=high`.
The options are `thinking_enabled`, `reasoning_effort` (`low`, `medium`, `high`, `xhigh`, `max`), `max_tokens` and
`temperature`. An option a model cannot honour raises `ValueError` before any request is sent.

## Configuration

Providers are configured through environment variables (a `.env` file is loaded automatically).
`LLMConfig.from_env()` leaves out every provider whose key is absent. See [`.env.example`](.env.example):

```bash
OPENAI_API_KEY=...
OPENAI_ORGANIZATION=...
OPENAI_PROJECT=...

GEMINI_API_KEY=...
GEMINI_PROJECT=...

GROK_API_KEY=...
GROK_BASE_URL=...   # optional, defaults to https://api.x.ai/v1

CLAUDE_API_KEY=...

MISTRAL_API_KEY=...

METACENTRUM_BASE_URL=...
METACENTRUM_API_KEY=...

HF_TOKEN=...   # only for gated/private HuggingFace models

PROMPT_CACHE_KEY=...   # optional, defaults to PlyLLM
```

OpenAI, Grok and Mistral take a prompt cache key with every request: requests that share the key and a
prompt prefix are routed to the same cache, so a repeated system prompt is billed at the cached rate. The
key is `PlyLLM` unless set through `PROMPT_CACHE_KEY`, `LLMConfig.from_env(prompt_cache_key=...)` (which
wins over the environment) or `LLMConfig(prompt_cache_key=...)`.

### Rate limits

Two independent layers protect against provider throttling:

- **Provider concurrency** (`ProviderSemaphore`) caps in-flight requests per provider. This is the
  only mechanism most providers need — set it with `LLMConfig(default_concurrency=...)` or
  `llm.set_concurrency(provider, n)`.
- **Per-model quotas** (`ModelLimits`) additionally pace a single model by in-flight share, requests
  per second and tokens per minute.

Both layers apply to **every** provider — the gate is enforced in the shared dispatch path each client
routes its API call through, so nothing is provider-specific.

**No model ships with a quota**, because published allowances are account-specific — they depend on
your tier and differ per model. Apply your own when you know them:

```python
from plyllm import ModelLimits, Provider

llm.set_model_limits(Provider.MISTRAL, "mistral-small-4", ModelLimits(max_concurrent=8, rps=1.67, tpm=100_000))
llm.set_model_limits(Provider.MISTRAL, "mistral-small-4", None)  # back to unlimited
```

Every field is optional, so you can pace on tokens alone and leave requests unbounded. Unset limits
mean a model is governed only by the provider semaphore.

Token pacing reserves an estimate before each call (prompt length plus `max_tokens`, or the model's
default output guess) and reconciles it against reported usage once the response lands, so an
inaccurate estimate costs a little throughput rather than correctness. Because quotas are account-wide
while each process paces only itself, scale them down with `ModelLimits.scaled(factor)` (a factor below
`1.0`) when several runs share an account.
`safe_call`'s backoff remains the backstop for any 429 that slips through.

When per-model shares exceed the provider cap they queue behind it, so set the provider concurrency at
or above their sum (24 for the two models above) to keep one model from holding slots the other's quota
could use.

### Retries and HTTP transport

A failed call is retried with exponential backoff only when another attempt can help: a connection
failure, a timeout, or HTTP 408, 409, 429 or 5xx. Any other error (400, 401, 404, ...) fails at once. A call
that still fails raises `LLMCallError`, whose `kind` says why (`timeout`, `rate_limit`, `connection`, ...).
This rule only works if every provider raises errors of a known type. That is why all providers talk HTTP
through **httpx**:

- OpenAI, Grok, Metacentrum, Claude and Mistral use httpx through their SDKs.
- `google-genai` switches its async calls to **aiohttp** whenever aiohttp is importable. Then a network
  failure surfaces as an aiohttp exception, which the retry rule does not know. The call would fail at once
  as `other` instead of being retried. The Gemini client therefore passes genai its own `httpx.AsyncClient`,
  which is genai's only switch that forces httpx. Gemini then behaves the same whatever else is installed
  (any other package can install aiohttp).

Supporting aiohttp's errors instead was rejected. It would need aiohttp as a required dependency of the
`gemini` extra, a second set of error types in the Gemini retry rule, and per-event-loop session cleanup.
And it gains nothing measurable here: each call waits seconds to minutes for the model, so transport
overhead does not matter. genai does not close a client it was given, so `LLM.aclose()` closes this one
itself.

### HuggingFace (local models)

The HuggingFace provider runs models locally instead of calling a remote API — it exposes
`embed()` (generation is not supported yet). `huggingface_models` selects which of the supported models
(see `providers/huggingface/models.py`) this environment uses; `llm.bootstrap()` downloads/verifies them
into the local HF cache:

```python
llm = LLM(LLMConfig.from_env(huggingface_models=["sup-simcse-bert"]))
llm.bootstrap()
resp = await llm.embed(EmbeddingModelConfig(Provider.HUGGINGFACE, "sup-simcse-bert"), ["hello"], EmbeddingTask.SEARCH_QUERY)
```

Requesting a supported model that was not part of `huggingface_models` raises an error telling you to add
it to the bootstrap list. Needs the `huggingface` extra installed.

## Development

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev   # create .venv with the dev tools and the API SDKs
make test             # run the offline tests
make test-models      # free: check every active model is still served (needs the keys in .env)
make test-live        # paid: a few real calls per provider (asks first)
make lint             # ruff, prettier and pyright, as in CI
make format           # fix lint and formatting
make build            # build the wheel and sdist into dist/ and check their metadata
```

## Releasing

```bash
make release-dry-run VERSION=1.0.0   # print every step, change nothing
make release VERSION=1.0.0           # lint, test, check served models, bump, tag, push, create the GitHub Release
```

The GitHub Release starts `.github/workflows/publish.yml`, which uploads to PyPI through trusted
publishing. Add release notes under `## [Unreleased]` in `CHANGELOG.md` first; the release moves them
under the new version.

## License

MIT
