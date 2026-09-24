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
5. Run the relevant tests, formatter, and type checks.
6. Summarize:
   - providers checked
   - models added, updated, or removed
   - pricing or reasoning changes
   - validation performed
   - official documentation used

Use official provider documentation as the source of truth. Research every provider supported by the repository, not only providers such as OpenAI, Gemini, or Anthropic.
