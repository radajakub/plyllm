# kept so `uv run python scripts/generate.py ...` still works in this checkout; the tool is plyllm.cli.generate, installed as plyllm-generate
from plyllm.cli.generate import main

if __name__ == "__main__":
    raise SystemExit(main())
