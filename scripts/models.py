#!/usr/bin/env python3
# Lists the models in the registry of every configured provider, with their prices and capabilities.
# --served also asks each provider which models it serves right now (free, needs network).
#
# Usage:
#   uv run python scripts/models.py
#   uv run python scripts/models.py claude openai --served

import argparse
import sys

from common import build_llm, run

from plyllm import LLM, Provider


def provider_argument(text: str) -> Provider:
    provider = Provider.from_value(text)
    if provider is None:
        raise argparse.ArgumentTypeError(f"unknown provider {text!r} (providers: {', '.join(Provider.values())})")
    return provider


async def served(llm: LLM, providers: list[Provider]) -> dict[Provider, set[str] | Exception]:
    result: dict[Provider, set[str] | Exception] = {}
    for provider in providers:
        try:
            result[provider] = await llm.served_models(provider)
        except Exception as error:
            result[provider] = error
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="List the models of the configured providers.")
    parser.add_argument("providers", nargs="*", type=provider_argument, help="providers to list (default: every configured one)")
    parser.add_argument("--served", action="store_true", help="also check which models each provider serves now (free, needs network)")
    args = parser.parse_args()

    llm = build_llm()
    providers: list[Provider] = args.providers or llm.available_providers
    missing = [provider.value for provider in providers if provider not in llm.available_providers]
    if missing:
        print(f"error: not configured (no key in .env): {', '.join(missing)}", file=sys.stderr)
        return 2

    served_by = run(llm, served(llm, providers)) if args.served else {}

    for provider in providers:
        print(f"== {provider.value}")
        status = served_by.get(provider)
        if isinstance(status, Exception):
            print(f"   cannot list served models: {status}")
        for model in llm.get_available_models(provider):
            flags = [
                "retired" if model.retired else "",
                "thinking-only" if model.thinking_only else "thinking" if model.thinking else "",
                "schema" if model.can_use_json_schema else "",
                f"batch x{model.batch_ratio}" if model.batch_ratio is not None else "",
                f"effort {'/'.join(sorted(model.supported_reasoning))}" if model.supported_reasoning else "",
            ]
            if isinstance(status, set):
                flags.append("served" if model.model_string in status else "NOT SERVED")
            price = f"${model.input_cost}/${model.cached_input_cost}/${model.output_cost} per 1M in/cached/out"
            print(f"   {model.model_name:<28} {model.model_string:<40} {price:<42} {', '.join(flag for flag in flags if flag)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
