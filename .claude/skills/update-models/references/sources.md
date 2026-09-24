# Provider documentation sources

Where the model data in `src/plybench/llm/providers/*/models.py` comes from. Verified 2026-09-24.

Fetch the pricing page and the per-model pages in parallel; the pricing pages carry every model's
rates in one table, and the per-model pages are the only place that lists a model's exact set of
reasoning/effort levels. A general "reasoning" guide and a per-model page sometimes disagree about
which levels a model accepts — prefer the intersection, because an unsupported level is a runtime
400 while a missing one only blocks a config.

## Anthropic (`claude`)

- Pricing (input / output / cache read + write, batch, all models): <https://platform.claude.com/docs/en/about-claude/pricing>
- Model lineup, IDs, context, max output, thinking mode, default effort: <https://platform.claude.com/docs/en/about-claude/models/overview>
- Effort levels per model, and per-model recommendations: <https://platform.claude.com/docs/en/build-with-claude/effort>
- One model's full spec: `https://platform.claude.com/docs/en/models/<slug>/overview` (e.g. `opus-5-5`, `fable-5-1`)
- Deprecations / retirements: <https://platform.claude.com/docs/en/about-claude/model-deprecations>

`docs.claude.com/en/docs/...` 302-redirects to `platform.claude.com/docs/en/...`; fetch the
`platform.claude.com` URL directly to save a round trip.

Repository notes: `cached_input_cost` is the cache-read rate, normally 0.1x input but 0.025x on
Fable 5.1 and 0.05x on Opus 5.5. "Adaptive (always on)" in the docs maps to `thinking_only=True`.
Models whose docs say thinking cannot be disabled at any effort are `thinking_only`; Opus 5 can only
disable it at `high` or below, which `_THINKING_ONLY_EFFORTS` already encodes.

## OpenAI (`openai`)

- Pricing, all models, input / cached input / output: <https://developers.openai.com/api/docs/pricing>
- Current model list: <https://developers.openai.com/api/docs/models>
- Per-model effort values, default and pricing: `https://developers.openai.com/api/docs/models/<model-id>`
- Reasoning guide (general, does not enumerate per model): <https://developers.openai.com/api/docs/guides/reasoning>

`platform.openai.com/docs/...` 301-redirects to `developers.openai.com/api/docs/...`.

Repository notes: the per-model pages list `none` as an effort value; the harness expresses that as
`thinking_enabled=False`, so `none` is never in a `supported_reasoning` set. Effort sets differ
within a generation — GPT-6 and GPT-5.6 accept `max`, GPT-5.5 and GPT-5.4 stop at `xhigh`, and the
Pro variants start at `medium`.

## Google Gemini (`gemini`)

- Pricing, paid tier, per model: <https://ai.google.dev/gemini-api/docs/pricing>
- Model list, stable vs preview: <https://ai.google.dev/gemini-api/docs/models>
- Deprecations and shutdown dates: <https://ai.google.dev/gemini-api/docs/deprecations>
- Embedding task prefixes (`task:` / `title:`), verbatim: <https://ai.google.dev/gemini-api/docs/embeddings>

Repository notes: Flash pricing is promotional and dated — 3.8 / 3.7 / 3.6 Flash are $0.75 / $3.75
through 2026-12-31 and $1.50 / $7.50 after. Record the rate in force, with a comment giving the
change date. Pro models have a >200k-token tier; the repository stores the standard tier only.
The models page lists preview models that the deprecations page has already shut down — always
cross-check both before keeping a `-preview` id.

## xAI Grok (`grok`)

- Model list and pricing table: <https://docs.x.ai/docs/models>
- Per-model page, with the effort list and default: `https://docs.x.ai/docs/models/<model-id>`
- Reasoning guide: <https://docs.x.ai/docs/guides/reasoning>

Repository notes: prices are the standard sub-200k-token tier; requests reaching 200k are billed at
roughly double for the whole request. The reasoning guide and the per-model pages disagree about
`xhigh` on grok-4.5 and about whether grok-4.3 reasons at all — both are kept at low/medium/high.
Reasoning cannot be disabled on any current Grok model.

## Mistral (`mistral`)

- API pricing per model: <https://mistral.ai/pricing/api>
- Model list, dated ids, deprecations: <https://docs.mistral.ai/getting-started/models/models_overview/>
- Reasoning capability: <https://docs.mistral.ai/capabilities/reasoning/>

Repository notes: the reasoning docs only document `high` and `none`. The accepted set comes from
the installed SDK enum instead — read it directly:

```
cat .venv/lib/python3.13/site-packages/mistralai/client/models/reasoningeffort.py
```

It currently reads `none, minimal, low, medium, high, xhigh`; `_MISTRAL_REASONING` is that set minus
`none`. Cached input is documented only as "up to -90% on input tokens", so `cached_input_cost` is
0.1x input. Mistral also sells Large 3, Ministral 3 and hosted GLM models; they are deliberately not
in the repository because their reasoning support is undocumented.

## Metacentrum / e-INFRA (`metacentrum`)

- Served models, API ids, aliases, obsolete models: <https://docs.cerit.io/en/docs/ai-as-a-service/chat-ai>
- API base URL: `https://llm.ai.e-infra.cz/v1/` (set locally through `OS_BASE_URL` / `OS_API_KEY`)

Repository notes: self-hosted and free, so every model carries `input_cost=0` / `output_cost=0`.
The service replaces models without notice and the page has a separate "Obsolete Models" table —
check it, not just the main table. Models it drops are **kept** in `models.py` under a
"no longer served" comment so existing experiment configs and recorded results still resolve
(`resolve_model` raises on an unknown name). Reasoning-effort support is not documented per model,
so leave `supported_reasoning=None` unless a model is known to accept it. The stable aliases
(`qwen3.5`, `kimi`, `glm`, ...) resolve to whatever is current and are safe as `model_string`.

## Hugging Face (`huggingface`)

Local encoder checkpoints run through `transformers`; they are free, have no API pricing and no
reasoning settings. Only verify the repo id still exists, e.g.
<https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased>.

## After editing

```
make format
make test
make typecheck
```

Then check the prices and effort sets round-trip:

```
uv run python -c "from plybench.llm.providers.claude.models import claude_models; print([(m.model_name, m.input_cost, m.output_cost) for m in claude_models()])"
```
