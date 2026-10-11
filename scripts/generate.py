#!/usr/bin/env python3
# One generation call: send a prompt to a model, print the answer, its tokens and its cost.
# Makes a real, paid API call.
#
# Usage:
#   uv run python scripts/generate.py claude:claude-haiku-4.5 "What is 2 + 2?"
#   uv run python scripts/generate.py "openai:gpt-6-luna:thinking_enabled=True,reasoning_effort=low" @prompt.txt --system @system.txt
#   uv run python scripts/generate.py gemini:gemini-3.5-flash-lite "What is 2 + 2?" --schema Answer

import argparse
import sys

from common import build_llm, model_argument, print_failure, print_response, read_text, run, schema_argument
from pydantic import BaseModel

from plyllm import LLMCallError, LLMMessage


# example schema for --schema Answer
class Answer(BaseModel):
    answer: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one prompt to one model and print the answer.")
    parser.add_argument("model", type=model_argument, help="<provider>:<model>[:<options>], e.g. openai:gpt-5.5:thinking_enabled=True,reasoning_effort=low")
    parser.add_argument("prompt", help="the user message; '-' reads stdin, '@path' reads a file")
    parser.add_argument("--system", default="You are a helpful assistant.", help="the system message; '@path' reads a file")
    parser.add_argument("--schema", type=schema_argument, help="pydantic model for structured output: a class in this script (e.g. Answer) or module:Class")
    args = parser.parse_args()

    llm = build_llm()
    system = LLMMessage.system(read_text(args.system))
    messages = [LLMMessage.user(read_text(args.prompt))]

    try:
        response = run(llm, llm.generate(args.model, system, messages, output_schema=args.schema))
    except LLMCallError as error:
        print_failure(llm, args.model, error)
        return 1
    except ValueError as error:
        # an unconfigured provider, an unknown model or options the model cannot take; nothing was sent
        print(f"error: {error}", file=sys.stderr)
        return 2

    print_response(llm, args.model, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
