---
name: update-models
description: Update supported model definitions using current official provider documentation.
---

Read `references/sources.md` before step 2. For every provider it gives the exact official
documentation URLs to fetch, the repository's conventions for that provider (which pricing tier,
the cache-read multiplier, how thinking and effort are encoded), and the places where a provider's
own pages contradict each other. It is a shortcut past the search, not a substitute for the docs:
always re-fetch the pages and verify against them. Update `references/sources.md` in the same change
whenever a URL moves, a provider is added, or a convention changes.

For every model provider supported by this repository:

1. Inspect the repository to find the provider's model definitions and related metadata.
2. Check the provider's official developer documentation for:
   - current API model IDs
   - pricing in USD
   - reasoning/thinking support and available settings
   - deprecated, renamed, or newly available models

3. Update the repository with the latest verified information.
   - Preserve existing data structures, naming conventions, and pricing units.
   - Add new supported models and update outdated entries.
   - Remove or deprecate models only when appropriate for the repository.
   - Do not guess undocumented pricing or capabilities.

4. Update related aliases, tests, fixtures, or documentation when required.
5. Run the relevant tests, formatter, and type checks. Then run `make test-models`: it is free (each
   provider's model-list endpoint, keys from .env) and compares every active registry model with the
   ids the provider serves right now. Read a failure before changing anything:
   - `no longer serves [...]`: first check whether the id is an alias the listing omits (Anthropic
     lists only the dated snapshot of some models, e.g. `claude-haiku-4-5-20251001`). If so, switch
     `model_string` to the listed id the docs say it resolves to. Mark a model `retired=True` only
     when the docs or the listing show it is really gone; never delete it, recorded results need it.
   - An error raised while listing (401, 403, "used all available credits", a timeout) is an
     account or network problem, not a registry one. Report it and leave the registry alone.
   - A warning that a retired model is served again: report it; do not un-retire on your own.

   `make test-models` only checks models already in the registry; it cannot see one that was left
   out. `references/sources.md` lists models that were removed or skipped because the account could
   not use them (e.g. grok-4.7 / grok-4.6), each with a free command to check whether they are served
   now. Run those checks too, and re-add a model as soon as it is served.

6. Summarize:
   - providers checked
   - models added, updated, or removed
   - pricing or reasoning changes
   - validation performed, including the `make test-models` result per provider
   - official documentation used

Use official provider documentation as the source of truth. Research every provider supported by the repository, not only providers such as OpenAI, Gemini, or Anthropic.
