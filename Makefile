.PHONY: lint typecheck format check fix test build clean release release-dry-run require-version

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
release: require-version lint test
	uv run python scripts/release.py $(VERSION)

# Print every step of a release without changing anything
release-dry-run: require-version
	uv run python scripts/release.py $(VERSION) --dry-run
