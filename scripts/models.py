# kept so `uv run python scripts/models.py ...` still works in this checkout; the tool is plyllm.cli.models, installed as plyllm-models
from plyllm.cli.models import main

if __name__ == "__main__":
    raise SystemExit(main())
