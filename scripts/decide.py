#!/usr/bin/env python3
# Decisions API: ask fixed-answer questions about one or more texts and print every answer with its
# probabilities, the tokens (output and reasoning included, though only input is billed), the cost and the
# latency. Makes real, paid API calls, one per input, sent concurrently.
#
# Questions come from a JSON file, or default to the example below. The file holds a list of questions;
# an option or a level may be a bare string when it needs no description:
#   [
#     {"type": "predicate", "name": "damaged", "instructions": "Is the item damaged?"},
#     {"type": "choice", "name": "department", "instructions": "Which department should handle this?",
#      "options": [{"value": "billing", "description": "Payments and invoices"}, "shipping", true]},
#     {"type": "score", "name": "severity", "instructions": "How severe is the issue?",
#      "levels": ["low", {"label": "high", "description": "Blocks the customer"}]}
#   ]
#
# Usage:
#   uv run python scripts/decide.py openai:gpt-6-luna "The parcel arrived torn and I was charged twice."
#   uv run python scripts/decide.py openai:gpt-6-luna @trace1.txt @trace2.txt --questions questions.json
#   cat trace.txt | uv run python scripts/decide.py openai:gpt-6-luna - --questions questions.json

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from common import build_llm, read_text, run

from plyllm import (
    LLM,
    Choice,
    ChoiceAnswer,
    ChoiceOption,
    DecisionModelConfig,
    DecisionQuestion,
    DecisionResponse,
    LLMCallError,
    Predicate,
    PredicateAnswer,
    Provider,
    RefusedAnswer,
    Score,
    ScoreAnswer,
    ScoreLevel,
)

EXAMPLE_QUESTIONS: list[DecisionQuestion] = [
    Predicate("damaged", "Does the customer report a damaged item?"),
    Choice(
        "department",
        "Which department should handle this message?",
        [ChoiceOption("billing", "Payments, invoices and refunds"), ChoiceOption("shipping", "Delivery and damaged parcels"), ChoiceOption("other")],
    ),
    Score("severity", "How severe is the customer's problem?", [ScoreLevel("low"), ScoreLevel("medium"), ScoreLevel("high", "The customer lost money or cannot use the product")]),
]


def model_argument(text: str) -> DecisionModelConfig:
    # argparse type for `<provider>:<model>`; decisions take no options
    provider, _, model_name = text.partition(":")
    if provider not in Provider.values() or not model_name:
        raise argparse.ArgumentTypeError(f"expected <provider>:<model> with a provider in {Provider.values()}, got {text!r}")
    return DecisionModelConfig(Provider(provider), model_name)


def _option(item: Any) -> ChoiceOption:
    return ChoiceOption(item["value"], item.get("description")) if isinstance(item, dict) else ChoiceOption(item)


def _level(item: Any) -> ScoreLevel:
    return ScoreLevel(item["label"], item.get("description")) if isinstance(item, dict) else ScoreLevel(item)


def load_questions(path: Path) -> list[DecisionQuestion]:
    questions: list[DecisionQuestion] = []
    for item in json.loads(path.read_text()):
        kind, name, instructions = item["type"], item["name"], item["instructions"]
        if kind == "predicate":
            questions.append(Predicate(name, instructions))
        elif kind == "choice":
            questions.append(Choice(name, instructions, [_option(option) for option in item["options"]]))
        elif kind == "score":
            questions.append(Score(name, instructions, [_level(level) for level in item["levels"]]))
        else:
            raise ValueError(f"question {name!r} has type {kind!r}; expected predicate, choice or score")
    return questions


def print_decision(llm: LLM, config: DecisionModelConfig, response: DecisionResponse, seconds: float) -> None:
    for answer in response.answers:
        if isinstance(answer, PredicateAnswer):
            print(f"  {answer.name}: p(true)={answer.probability:.3f}")
        elif isinstance(answer, ChoiceAnswer):
            ranked = sorted(answer.probabilities.items(), key=lambda item: item[1], reverse=True)
            print(f"  {answer.name}: {answer.choice!r} (confidence={answer.confidence:.3f})  " + "  ".join(f"{value!r}={p:.3f}" for value, p in ranked))
        elif isinstance(answer, ScoreAnswer):
            levels = "  ".join(f"{item.value}:{item.label}={item.probability:.3f}" for item in answer.probabilities)
            print(f"  {answer.name}: score={answer.score:.3f} (confidence={answer.confidence:.3f})  {levels}")
        elif isinstance(answer, RefusedAnswer):
            print(f"  {answer.name}: refused")
    tokens = response.tokens
    cost = llm.calculate_decision_cost(config, tokens)
    print(
        f"  [{response.model_string}] input={tokens.input_tokens} (cached={tokens.cached_input_tokens}) "
        f"output={tokens.output_tokens} (reasoning={tokens.reasoning_tokens}) cost=${cost:.6f} time={seconds:.2f}s",
        file=sys.stderr,
    )


async def decide_all(llm: LLM, config: DecisionModelConfig, inputs: list[str], questions: list[DecisionQuestion]) -> list[tuple[DecisionResponse | LLMCallError, float]]:
    async def one(text: str) -> tuple[DecisionResponse | LLMCallError, float]:
        start = time.perf_counter()
        try:
            response: DecisionResponse | LLMCallError = await llm.decide(config, text, questions)
        except LLMCallError as error:
            response = error
        return response, time.perf_counter() - start

    return list(await asyncio.gather(*(one(text) for text in inputs)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask fixed-answer questions about texts through the Decisions API.")
    parser.add_argument("model", type=model_argument, help="<provider>:<model>, e.g. openai:gpt-6-luna")
    parser.add_argument("inputs", nargs="+", help="the texts to decide on, one call each; '-' reads stdin, '@path' reads a file")
    parser.add_argument("--questions", type=Path, help="JSON file with the questions (format at the top of this script); default: a customer-support example")
    args = parser.parse_args()

    try:
        questions = load_questions(args.questions) if args.questions is not None else EXAMPLE_QUESTIONS
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"error: cannot read questions: {error!r}", file=sys.stderr)
        return 2
    inputs = [read_text(value) for value in args.inputs]

    llm = build_llm()
    try:
        results = run(llm, decide_all(llm, args.model, inputs, questions))
    except (ValueError, NotImplementedError) as error:
        # an unconfigured provider, a provider without decisions, an unknown model or invalid questions;
        # nothing was sent
        print(f"error: {error}", file=sys.stderr)
        return 2

    failed = 0
    for value, (result, seconds) in zip(args.inputs, results):
        print(f"{value if len(value) <= 60 else value[:57] + '...'}")
        if isinstance(result, LLMCallError):
            failed += 1
            print(f"  error ({result.kind.value}): {result}  time={seconds:.2f}s", file=sys.stderr)
        else:
            print_decision(llm, args.model, result, seconds)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
