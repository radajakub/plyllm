.PHONY: lint typecheck format check fix test test-models test-live build clean release release-dry-run require-version

# Check lint and formatting (same checks as CI)
lint:
	uv run ruff check .
	uv run ruff format --check .
	npx --yes prettier@3.9.6 --check .
	$(MAKE) typecheck

typecheck:
	uv run pyright

# Alias for lint
check: lint

# Autofix lint issues and reformat
format:
	uv run ruff check --fix .
	uv run ruff format .
	npx --yes prettier@3.9.6 --write .

# Alias for format
fix: format

test:
	uv run --group dev pytest -q

# Free: checks every active model in the registries is still served (needs network and the keys in .env)
test-models:
	uv run --group dev pytest -q -m online --online -rs

# Real, paid API calls to every provider with a key in .env (a few cents per run), plus the free model
# check; asks before running. CONFIRM=yes skips the question, for a future CI job
test-live:
	@if [ "$(CONFIRM)" != "yes" ]; then \
		printf "This makes real, paid API calls to every provider configured in .env (a few cents). Continue? [y/N] "; \
		read answer; \
		case "$$answer" in y|Y|yes) ;; *) echo "Aborted."; exit 1 ;; esac; \
	fi
	uv run --group dev pytest -q -m "live or online" --live --online -rs

# Build wheel and sdist into dist/, then validate the metadata PyPI will see
build: clean
	uv build
	uvx twine check dist/*

clean:
	rm -rf dist build

# Fail before the slow checks run, so a missing VERSION costs nothing
require-version:
	@test -n "$(VERSION)" || (echo "usage: make $(MAKECMDGOALS) VERSION=1.0.0" >&2; exit 1)

# Cut a release: make release VERSION=1.0.0
# Bumps the version, tags it, and publishes a GitHub Release, which triggers the PyPI upload.
# test-models first, so a release never ships a registry with a model the provider has dropped
release: require-version lint test test-models
	uv run python scripts/release.py $(VERSION)

# Print every step of a release without changing anything
release-dry-run: require-version
	uv run python scripts/release.py $(VERSION) --dry-run
