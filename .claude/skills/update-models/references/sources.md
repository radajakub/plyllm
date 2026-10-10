# Provider documentation sources

Where the model data in `src/plybench/llm/providers/*/models.py` comes from. Verified 2026-10-09.

Fetch the pricing page and the per-model pages in parallel; the pricing pages carry every model's
rates in one table, and the per-model pages are the only place that lists a model's exact set of
reasoning/effort levels. A general "reasoning" guide and a per-model page sometimes disagree about
which levels a model accepts — prefer the intersection, because an unsupported level is a runtime
400 while a missing one only blocks a config.

Batch prices (checked 2026-10-10) are a `batch_ratio` on each model, applied to input, cached input and
output: OpenAI 0.5 (every Batch row is half the Standard row; gpt-5.4 cached shows $0.13, rounded), Claude
0.5 (stacks with the cache multipliers), Mistral 0.5 (whether it covers cached input is not stated),
Gemini 0.5 (see its section for cache reads), Grok per model (0.8 on grok-4.3 and grok-4.20; no discount
listed for the others, which get `None`), Metacentrum `None` (free). Batch: OpenAI
<https://developers.openai.com/api/docs/guides/batch>, Anthropic pricing page, Gemini
<https://ai.google.dev/gemini-api/docs/batch-mode>, xAI <https://docs.x.ai/developers/pricing>, Mistral
<https://mistral.ai/pricing>.

## Anthropic (`claude`)

- Pricing (input / output / cache read + write, batch, all models): <https://platform.claude.com/docs/en/about-claude/pricing>
- Model lineup, IDs, context, max output, thinking mode, default effort: <https://platform.claude.com/docs/en/about-claude/models/overview>
- Effort levels per model, and per-model recommendations: <https://platform.claude.com/docs/en/build-with-claude/effort>
- One model's full spec: `https://platform.claude.com/docs/en/models/<slug>/overview` (e.g. `opus-5-5`, `fable-5-1`)
- Deprecations / retirements: <https://platform.claude.com/docs/en/about-claude/model-deprecations>

`docs.claude.com/en/docs/...` 302-redirects to `platform.claude.com/docs/en/...`; fetch the
`platform.claude.com` URL directly to save a round trip.

Repository notes: `cached_input_cost` is the cache-read rate, normally 0.1x input but 0.025x on
Fable 5.1 and 0.05x on Opus 5.5 and Sonnet 5.5. The models endpoint lists some older models only by dated snapshot (`claude-haiku-4-5-20251001`,
`claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929`), not by alias, so use the snapshot there. Haiku 5.5 has two price tiers (prompts up to / over
100k tokens); the repository stores the lower one. "Adaptive (always on)" in the docs maps to `thinking_only=True`.
Models whose docs say thinking cannot be disabled at any effort are `thinking_only`; Opus 5 can only
disable it at `high` or below, which `_THINKING_ONLY_EFFORTS` already encodes. Sonnet 5.5 rejects
`"disabled"` and only offers `thinking: {"type": "between_tools"}` (effort `high` or below), which the
harness does not model, so it is `thinking_only`. The per-model thinking table ("Rejected with 400")
is at <https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting>. Its footnote
names only Opus 5 and Haiku 5.5 as rejecting `"disabled"` at `xhigh`/`max`; the repository applies
that rule to every Claude model (`_THINKING_ONLY_EFFORTS`), which the docs neither confirm nor rule
out for Opus 4.8, Sonnet 5 and Sonnet 4.6.

## OpenAI (`openai`)

- Pricing, all models, input / cached input / output: <https://developers.openai.com/api/docs/pricing>
- Current model list: <https://developers.openai.com/api/docs/models>
- Per-model effort values, default and pricing: `https://developers.openai.com/api/docs/models/<model-id>`
- Reasoning guide (general, does not enumerate per model): <https://developers.openai.com/api/docs/guides/reasoning>

`platform.openai.com/docs/...` 301-redirects to `developers.openai.com/api/docs/...`.

Repository notes: the per-model pages list `none` as an effort value; the harness expresses that as
`thinking_enabled=False` (the provider sends effort `none`), so `none` is never in a `supported_reasoning`
set. A model whose page does not list `none` is `thinking_only` (on 2026-10-09: gpt-6-astra, gpt-6.1-sol,
the Pro models, and gpt-5-mini / gpt-5-nano, whose pages list no efforts at all). A model whose default
effort is `none` (gpt-5.4, -mini, -nano) is `needs_effort_to_think`, or thinking on would not reason.
Temperature is accepted only at effort `none` (GPT-6 migration guide, "When reasoning effort is not none,
remove temperature, top_p"), which `OpenAILLMModel` derives from `thinking_only`. Effort sets differ
within a generation — GPT-6.x and GPT-5.6 accept `max`, GPT-5.5 and GPT-5.4 stop at `xhigh`, and the
Pro variants start at `medium`. Cached input is not always 0.1x: gpt-6.1-sol is 0.05x. gpt-6.1-sol rejects `none` (thinking cannot be
turned off), and the gpt-6-astra page does not list `none`. gpt-5.6-sol is on promotional pricing "at
least through November 21, 2026". The gpt-5-mini / gpt-5-nano pages no longer list effort values.
Shutdown dates: <https://developers.openai.com/api/docs/deprecations>.

## Google Gemini (`gemini`)

- Pricing, paid tier, per model: <https://ai.google.dev/gemini-api/docs/pricing>

`cached_input_cost` is the "Context caching price" row (hourly storage is not modelled). The batch table
lists its own cache-read price; where it is not half of the standard one (2.5 models, 3.1 Pro, 3 Flash:
same as standard; 3.5 Flash-Lite: 0.02), record it as `batch_cached_input_cost`. The batch-mode page says
batch cache hits pay "the standard context caching rates", which contradicts the table for 3.8 / 3.6 Flash
and 3.1 Flash-Lite; the table is used. 3.7 Flash and 3.5 Flash had no pricing block on 2026-10-10, so
their cache price is 0 until the page lists one.
- Model list, stable vs preview: <https://ai.google.dev/gemini-api/docs/models>
- Deprecations and shutdown dates: <https://ai.google.dev/gemini-api/docs/deprecations>
- Embedding task prefixes (`task:` / `title:`), verbatim: <https://ai.google.dev/gemini-api/docs/embeddings>

Repository notes: Flash pricing is promotional and dated — 3.8 / 3.7 / 3.6 Flash are $0.75 / $3.75
through 2026-12-31 and $1.50 / $7.50 after. Record the rate in force, with a comment giving the
change date. Pro models have a >200k-token tier; the repository stores the standard tier only.
The models page lists preview models that the deprecations page has already shut down — always
cross-check both before keeping a `-preview` id (on 2026-10-09 it still listed
`gemini-embedding-2-preview`, which shut down on 2026-08-10). The thinking guide
(<https://ai.google.dev/gemini-api/docs/thinking>) is the per-model table of thinking levels and
defaults; it no longer documents `thinking_budget`, which the 2.5 models still use here. 3.8 and 3.7
Flash reject `minimal`. 3.1 Flash-Lite is not in that table.

## xAI Grok (`grok`)

- Model list and pricing table: <https://docs.x.ai/docs/models>
- Per-model page, with the effort list and default: `https://docs.x.ai/docs/models/<model-id>`
- Reasoning guide: <https://docs.x.ai/docs/guides/reasoning>

Repository notes: prices are the standard sub-200k-token tier; requests reaching 200k are billed at
roughly double for the whole request. The reasoning guide and the per-model pages disagree about
`xhigh` on grok-4.5 (the guide says it is silently treated as `high`) and about whether grok-4.3
reasons at all — both are kept at low/medium/high. The guide documents `reasoning_effort` only for
grok-4.7 / 4.6 / 4.5, and the grok-4.20 page has no effort section, so grok-4.20 takes none.
Reasoning cannot be disabled on grok-4.7 / 4.6 / 4.5, so they are `thinking_only`, as is grok-4.20
(no effort section; it always reasons); grok-4.3 accepts `none` (default `low`), which the provider sends
when thinking is off. No page documents `temperature` for the reasoning models, so every Grok model has
`TemperatureSupport.NEVER`; nor is `reasoning.summary` documented (it is still sent with an effort). Retirements: <https://docs.x.ai/developers/migration/may-15-retirement>.

**Waiting to be re-added: grok-4.7 and grok-4.6.** Removed on 2026-10-09: the docs list them, but the
API returns 404 for this account ("does not exist or your team does not have access to it"). On every
update, check whether they are served now (free, needs `GROK_API_KEY`):

```
uv run python -c "import asyncio; from plybench.llm import LLM, LLMConfig, Provider; print(sorted(asyncio.run(LLM(LLMConfig.from_env()).served_models(Provider.GROK))))"
```

If the list contains them, add them back with the values from their model pages (on 2026-10-09:
$2.00 / $6.00, cached $0.50; effort low/medium/high/xhigh, default high; reasoning cannot be
disabled), run `make test-models`, and delete this paragraph.

## Mistral (`mistral`)

- API pricing per model: <https://docs.mistral.ai/inference/pricing> (`mistral.ai/pricing/api` 301-redirects here)
- One model's page, with its API id: `https://docs.mistral.ai/models/<slug>` (e.g. `mistral-large-4-0`,
  `mistral-medium-3-5-26-04`, `mistral-small-4-0-26-03`; the undated slugs of the last two 404)
- Model list, dated ids, deprecations: <https://docs.mistral.ai/getting-started/models/models_overview/>
- Reasoning capability: <https://docs.mistral.ai/capabilities/reasoning/>

Repository notes: docs slugs are not API ids — the model page shows the id next to the parameter
count (`mistral-large-2512 +1` for Large 3, `mistral-large-4 +1` for Large 4). The reasoning page
lists which models take `reasoning_effort` (Small, Medium 3.5, Large 4, hosted GLM 5.3), but only
documents `high` and `none`, so `_MISTRAL_REASONING` is `{high}` (`none` is what the provider sends
when thinking is off). Do not take the set from the SDK enum (`none, minimal, low, medium, high,
xhigh`): on 2026-10-09 `mistral-small-2603` rejected `low` with "Must be one of (none, high)".
Widen the set for a model only after `make test-live` (or a single paid call, approved first) shows
the API accepts the level. Cached input is documented only as "up to -90% on input tokens", so `cached_input_cost` is
0.1x input. Large 4 is on an undated sale; the repository records the sale price
with the list price in a comment. Large 3 and Ministral 3 are deliberately not in the repository
because their reasoning support is undocumented. Hosted GLM 5.3 (`zai-glm-5-3`) documents only
`low`/`high`/`max` and rejects `none`, which the provider sends when thinking is off; it is left out
until the provider can express that. The Medium 3.5 page lists `mistral-medium-3-5`,
`mistral-medium-3` and `mistral-medium-latest`, not the `mistral-medium-2604` the repository sends;
`make test-models` shows whether the endpoint still serves it. WebFetch has swapped price labels on
the Large 4 page; read the pricing table from raw HTML (`curl`) when in doubt.

## Metacentrum / e-INFRA (`metacentrum`)

- Served models, API ids, aliases, obsolete models: <https://docs.cerit.io/en/docs/ai-as-a-service/chat-ai>
- API base URL: `https://llm.ai.e-infra.cz/v1/` (set locally through `OS_BASE_URL` / `OS_API_KEY`)

Repository notes: self-hosted and free, so every model carries `input_cost=0` / `output_cost=0`.
The service replaces models without notice and the page has a separate "Obsolete Models" table —
check it, not just the main table. The page can lag the live endpoint (on 2026-10-07 it still read
"Effective June 30, 2026" and listed `deepseek-v4-flash`, while `/v1/models` served
`deepseek-v4.1-flash` from 2026-09-28). Prefer `GET https://llm.ai.e-infra.cz/v1/models` (free,
needs `OS_API_KEY`) and ask before calling it. Models it drops are **kept** in `models.py`
and marked `retired=True`, so existing experiment configs and recorded results still resolve and
cost (`resolve_model` raises on an unknown name) while new calls are refused. `make test-models`
(free, needs the keys in .env) compares every provider's active registry models with its served list
and fails on a mismatch. Reasoning-effort support is not documented per model, so leave
`supported_reasoning=None` unless a model is known to accept it; the only documented toggle is
DeepSeek's `chat_template_kwargs: {"thinking": true}` (reasoning is off by default), on
<https://docs.cerit.io/en/docs/ai-as-a-service/ai-api>. That page is out of date: on 2026-10-09
deepseek-v4.1-flash reasoned with and without it, and with `thinking: false`. The thinking flags come
from a free probe instead (one short prompt per model, thinking on and off, reasoning tokens counted; put
a unique id in each prompt, because the proxy caches identical prompts and ignores `extra_body` when it
does). Results on 2026-10-09: the Qwen templates read `enable_thinking` (not `thinking`), kimi-k3 reads
`thinking`, gpt-oss-120b and deepseek-v4.1-flash always reason (`thinking_only`), gemma-4 reasons only with
an effort (`needs_effort_to_think`), mistral-medium-3.5 gave no reasoning with thinking on
(`thinking=False`), and glm-5.3 answered every call with 400 "`tools` must not be an empty array". On
2026-10-10 that happened even for a minimal `/v1/responses` request, while `/v1/chat/completions` answered
normally: the proxy's Responses-to-chat conversion breaks for this model, not the harness's request. The stable aliases (`qwen3.5`, `kimi`,
`glm`, ...) resolve to whatever is current, so they keep configs working but not outputs: `qwen3.5`
points at `qwen3.5-int4` (397B, AWQ int4).

## Hugging Face (`huggingface`)

Local encoder checkpoints run through `transformers`; they are free, have no API pricing and no
reasoning settings. Only verify the repo id still exists, e.g.
<https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased>.

## After editing

```
make format
make test
make typecheck
make test-models   # free: are all active models still served? Needs the keys in .env
```

Then check the prices and effort sets round-trip:

```
uv run python -c "from plybench.llm.providers.claude.models import claude_models; print([(m.model_name, m.input_cost, m.output_cost) for m in claude_models()])"
```
