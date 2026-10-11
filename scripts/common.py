# Shared helpers for the scripts that exercise one plyllm feature each: building the LLM from .env, parsing
# the command-line arguments every script takes, and printing a response.

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from plyllm import LLM, LLMCallError, LLMConfig, LLMResponse, ModelConfig

T = TypeVar("T")


def build_llm(concurrency: int | None = None) -> LLM:
    # every provider with a key in the environment or .env; the others are left out
    return LLM(LLMConfig.from_env(default_concurrency=concurrency))


def run(llm: LLM, call: Coroutine[Any, Any, T]) -> T:
    # closes the connection pools on the loop that used them ("Event loop is closed" otherwise)
    async def body() -> T:
        try:
            return await call
        finally:
            await llm.aclose()

    return asyncio.run(body())


def model_argument(text: str) -> ModelConfig:
    # argparse type for `<provider>:<model>[:<options>]`
    try:
        return ModelConfig.from_string(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def schema_argument(text: str) -> type[BaseModel]:
    # argparse type for `module:Class`, a pydantic model importable from here (scripts/ is on the path), or
    # a bare `Class` defined in the running script
    module_name, _, class_name = text.rpartition(":")
    module_name = module_name or "__main__"
    if not class_name:
        raise argparse.ArgumentTypeError(f"expected Class or module:Class, got {text!r}")
    try:
        schema = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as error:
        raise argparse.ArgumentTypeError(f"cannot import {text!r}: {error}") from error
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise argparse.ArgumentTypeError(f"{text!r} is not a pydantic model")
    return schema


def read_text(value: str) -> str:
    # `-` reads stdin, `@path` reads a file, anything else is the text itself
    if value == "-":
        return sys.stdin.read()
    if value.startswith("@"):
        return Path(value[1:]).read_text()
    return value


def print_response(llm: LLM, config: ModelConfig, response: LLMResponse) -> None:
    for summary in response.reasoning:
        print(f"[reasoning] {summary}\n")
    print(response.output_text)
    tokens = response.tokens
    cost = llm.calculate_cost(config, tokens, batch=response.batch)
    print(
        f"\n[{response.model_string}] input={tokens.input_tokens} (cached={tokens.cached_input_tokens}, cache_write={tokens.cache_write_tokens}) "
        f"output={tokens.output_tokens} (reasoning={tokens.reasoning_tokens}) cost=${cost:.6f}",
        file=sys.stderr,
    )


def print_failure(llm: LLM, config: ModelConfig, error: LLMCallError) -> None:
    print(f"error ({error.kind.value}): {error}", file=sys.stderr)
    # a refusal or an answer off the schema still arrived and was billed
    if error.response is not None:
        print(f"partial answer: {error.response.output_text}", file=sys.stderr)
        print(f"cost=${llm.calculate_cost(config, error.response.tokens, batch=error.response.batch):.6f}", file=sys.stderr)
