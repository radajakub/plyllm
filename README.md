# plyllm

Package that unifies communication with multiple LLM providers and models

## Installation

```bash
pip install "plyllm[all]"
```

Each provider SDK is an optional extra, so install only the ones you use: `openai`, `gemini`, `grok`,
`anthropic`, `mistral`, `metacentrum` (an OpenAI-compatible endpoint), `huggingface` (local embedding
models; pulls in torch).

## Development

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev   # create .venv with the dev tools and the API SDKs
make test             # run the tests
make lint             # ruff, prettier and pyright, as in CI
make format           # fix lint and formatting
make build            # build the wheel and sdist into dist/ and check their metadata
```

## Releasing

```bash
make release-dry-run VERSION=1.0.0   # print every step, change nothing
make release VERSION=1.0.0           # lint, test, bump, tag, push, create the GitHub Release
```

The GitHub Release starts `.github/workflows/publish.yml`, which uploads to PyPI through trusted
publishing. Add release notes under `## [Unreleased]` in `CHANGELOG.md` first; the release moves them
under the new version.

## License

MIT
