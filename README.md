# plyllm

Package that unifies communication with multiple LLM providers and models

## Installation

```bash
pip install "plyllm[all]"
```

Each provider SDK is an optional extra, so install only the ones you use: `openai`, `gemini`, `grok`,
`anthropic`, `mistral`, `metacentrum` (an OpenAI-compatible endpoint), `vllm` (a vLLM server we run), `ollama` (a local Ollama), `huggingface` (local embedding
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

VLLM_BASE_URL=...
VLLM_API_KEY=...

OLLAMA_BASE_URL=...   # e.g. http://127.0.0.1:11434/v1

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
This rule only works if every provider raises errors of a known type. That is why each provider talks HTTP
through one fixed library:

- OpenAI, Grok, Metacentrum, Claude and Mistral use **httpx2** through their SDKs (since `openai` 3,
  `anthropic` 1 and `mistralai` 3). The OpenAI-based and Claude clients catch the SDKs' own error types;
  the Mistral client catches httpx2's transport errors, which do not subclass httpx's.
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

### Batch API

OpenAI, Claude, Gemini and Mistral answer batch jobs at a lower price (half, for all four), in minutes to a day
instead of seconds. `llm.batch` sends independent requests that way. A batch can outlive the process that sent
it, so a run does what it can now and reports the rest: it submits the new requests, collects the jobs that have
ended and leaves the others pending. Running the same script again continues where it stopped.

```python
from plyllm import BatchRequest, JsonBatchStore

store = JsonBatchStore("batches.json")  # the jobs still at the provider; never the answers
requests = [BatchRequest(key, system, [LLMMessage.user(prompt)]) for key, prompt in uncached]

async with llm.batch(model, requests, store, output_schema=Verdict) as outcome:
    for key, result in outcome.results.items():  # LLMResponse, or LLMCallError for a failed request
        cache[key] = result  # save inside the block
    print(len(outcome.pending), "still running")
```

- **Ids are the caller's keys, per model config** (provider, model and options), so one store can run the same
  ids for several models. A request whose id is already in a running job of its model config is not sent
  again; the same id there with another prompt or schema raises `ValueError`. Claude allows only 1-64
  letters, digits, `-` and `_`.
- **Save the results inside the block.** Leaving it without an exception removes the collected jobs from the
  store, after which the same ids are new requests again. An exception keeps them, and the next run
  collects the same answers again, for free. `outcome.results` holds every request of a collected job, also
  ones this call did not pass.
- **Each request ends on its own:** an `LLMResponse` with `batch=True` (price it with
  `calculate_cost(..., batch=response.batch)`), or an `LLMCallError` whose `kind` is `refusal`,
  `unparseable`, `rate_limit`, `provider_error`, `batch_expired` or `batch_cancelled`.
- `wait=True` checks every `poll_interval` seconds (default 60) until every job has ended, instead of
  returning with requests pending. `llm.cancel_batches(provider, store)` stops a provider's running jobs;
  the next run collects them.
- **One store, any number of runs.** `JsonBatchStore` is one JSON file: written atomically and locked with
  `flock` (POSIX only), so two runs never submit the same id twice. Any object with `lock`, `load` and `save`
  (`BatchStore`) can replace it.
- Calls larger than a provider's limit are split into several batches, by request count and by size.
  OpenAI's uploaded input file is deleted once its results are collected; the output files stay.
- **If a submission times out**, the provider may have created the batch anyway. The store then keeps a
  record without a `batch_id`, and the next run that asks for those ids stops instead of paying twice:
  check the provider's batch list, then remove the record or fill in its `batch_id`.

A batch request carries exactly the body `generate` sends, and its answer is read by the same code, so the
two transports measure the same thing. Grok is not supported: its batch API differs from OpenAI's and the
result format for Responses requests is not documented.

### Decisions API

OpenAI's Decisions API answers questions with a fixed set of answers about one text, instead of generating
text: `Predicate` gives the probability that a statement is true, `Choice` picks one of the given options and
`Score` rates against ordered levels. `Choice` and `Score` also return a probability for each option or level.
Only `openai:gpt-6-luna` serves it, and it bills input tokens only.

```python
from plyllm import Choice, ChoiceOption, DecisionModelConfig, Predicate, Provider

model = DecisionModelConfig(Provider.OPENAI, "gpt-6-luna")
questions = [
    Predicate("damaged", "Is the item damaged?"),
    Choice("department", "Which department should handle this?", [ChoiceOption("billing"), ChoiceOption("shipping", "Delivery problems")]),
]
response = await llm.decide(model, "The parcel arrived torn.", questions)
print(response.answer("department"), llm.calculate_decision_cost(model, response.tokens))
```

- Every question needs a unique `name`. Answers come back in question order, and `response.answer(name)`
  finds one by name.
- The model can refuse a single question. That answer is a `RefusedAnswer`, and the other answers are kept.
- Input is text only. The API also takes images, which this package does not send.
- A choice question needs 2 to 255 options with distinct values. A value is a string or a boolean, and
  `"true"` and `True` are different options.
- `response.tokens` records every token count the API reports, including output and reasoning tokens.
  Only `input_tokens` is billed.
- Decisions are not available through the Batch API.

### vLLM (our own server)

The `vllm` provider calls a vLLM server we run, for example one started by
[plyserve](https://github.com/radajakub/plyserve), through Chat Completions. plyserve sets `VLLM_BASE_URL` and
`VLLM_API_KEY` for the command it runs, so `LLMConfig.from_env()` picks the server up:

```python
model = ModelConfig.from_string("vllm:qwen-3.8-27b-fp8:thinking_enabled=False")
assert model.model_name in await llm.served_models(Provider.VLLM)  # one server serves one model
```

- A model's name is the served name in plyserve's `models.toml`. Add a model to both, with the same name.
- Every request sends the model's full sampling set (from its model card), so the server's defaults from
  `generation_config.json` never apply. `temperature` overrides only the temperature.
- Thinking and `reasoning_effort` are chat template variables (`chat_template_kwargs`).
- A schema is asked for in the prompt, not enforced by the server (`weak_structured_output`).
- Calls cost nothing. Concurrency defaults to 32.

### Ollama (a local server)

The `ollama` provider calls a local Ollama through its OpenAI-compatible endpoint, with the same client code as
`vllm`. It is for trying prompts and the plumbing on a laptop; GGUF weights on Ollama and FP8 weights on vLLM are
different judges, so do not mix their results.

```python
model = ModelConfig.from_string("ollama:gemma-4-e4b:thinking_enabled=True")  # the tag gemma4:e4b
```

- A model has an alias without colons (`gemma-4-e4b`); the Ollama tag (`gemma4:e4b`) is what is sent.
- Thinking is switched with `reasoning_effort` (`none` for off).
- Ollama's `/v1` takes only `temperature`, `top_p` and the penalties. `top_k`, `min_p` and `repeat_penalty`
  come from the model's Modelfile, and the context size from the server (`OLLAMA_CONTEXT_LENGTH`, by default
  set from the memory). **A longer prompt is cut silently**, so set the context size before a long prompt.
- Concurrency defaults to 1, as Ollama serves one request per model unless `OLLAMA_NUM_PARALLEL` is raised.
- Ollama does not report reasoning or cached tokens: `reasoning_tokens` and `cached_input_tokens` are always 0,
  and the reasoning is likely counted in `output_tokens`. Do not measure reasoning length on Ollama.

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
