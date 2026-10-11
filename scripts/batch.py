#!/usr/bin/env python3
# Exercises every usage of the batch API against a real provider. `run`, `wait` and `cancel` make real, paid
# calls at the provider's batch price; `status` and `conflict` send nothing.
#
# The requests are fixed arithmetic questions with ids q0, q1, ..., so a later run asks the same questions under
# the same ids and collects the answers of an earlier one. The store is a JSON file under .batches/ (gitignored).
#
# Usage, in the order a batch goes through:
#   uv run python scripts/batch.py run openai:gpt-6-luna --count 3            # submit q0-q2; collect whatever has ended
#   uv run python scripts/batch.py status                                     # what the store tracks; free
#   uv run python scripts/batch.py run openai:gpt-6-luna --count 3 --keep     # collect, but keep the jobs in the store
#   uv run python scripts/batch.py run openai:gpt-6-luna --count 5            # tracks q0-q2, submits only q3-q4
#   uv run python scripts/batch.py conflict openai:gpt-6-luna                 # same ids, other prompts: refused; free
#   uv run python scripts/batch.py cancel openai:gpt-6-luna                   # stop the running jobs; `run` then collects them
#   uv run python scripts/batch.py wait openai:gpt-6-luna --count 2 --schema Answer --poll 30  # submit and wait for the answers

import argparse
import sys
from pathlib import Path

from common import build_llm, model_argument, print_failure, print_response, run, schema_argument
from pydantic import BaseModel

from plyllm import LLM, BatchOutcome, BatchRequest, JsonBatchStore, LLMCallError, LLMMessage, LLMResponse, ModelConfig

DEFAULT_STORE = Path(".batches/batches.json")
SYSTEM = LLMMessage.system("Answer the question.")


# example schema for --schema Answer
class Answer(BaseModel):
    answer: int


class _KeepJobs(Exception):
    # raised inside the batch block: leaving it with an exception keeps the collected jobs in the store
    pass


def question(index: int, changed: bool = False) -> tuple[str, int]:
    a, b = index + 2, index + 3
    if changed:
        b += 1
    return f"What is {a} + {b}? Reply with the number only.", a + b


def requests(count: int) -> list[BatchRequest]:
    return [BatchRequest(f"q{index}", SYSTEM, [LLMMessage.user(question(index)[0])]) for index in range(count)]


def expected(request_id: str) -> int:
    return question(int(request_id.removeprefix("q")))[1]


def correct(response: LLMResponse, request_id: str, schema: type[BaseModel] | None) -> bool:
    if schema is Answer:
        return response.resolve_structured_output(Answer).answer == expected(request_id)
    return str(expected(request_id)) in response.output_text


def report(llm: LLM, config: ModelConfig, outcome: BatchOutcome, schema: type[BaseModel] | None) -> None:
    cost = 0.0
    right = 0
    for request_id in sorted(outcome.results, key=lambda key: int(key.removeprefix("q"))):
        result = outcome.results[request_id]
        print(f"--- {request_id} (expected {expected(request_id)})")
        if isinstance(result, LLMCallError):
            print_failure(llm, config, result)
            if result.response is not None:
                cost += llm.calculate_cost(config, result.response.tokens, batch=result.response.batch)
            continue
        print_response(llm, config, result)
        cost += llm.calculate_cost(config, result.tokens, batch=result.batch)
        right += correct(result, request_id, schema)
    print(f"\ncollected {len(outcome.results)} ({right} correct), pending {len(outcome.pending)}: {sorted(outcome.pending)}")
    print(f"cost of the collected answers: ${cost:.6f}")


async def batch(llm: LLM, config: ModelConfig, store: JsonBatchStore, count: int, schema: type[BaseModel] | None, wait: bool, poll: float, keep: bool) -> None:
    # checks the model, options, schema and ids here, before anything is sent
    batch_run = llm.batch(config, requests(count), store, output_schema=schema, wait=wait, poll_interval=poll)
    try:
        async with batch_run as outcome:
            report(llm, config, outcome, schema)
            if keep:
                raise _KeepJobs
    except _KeepJobs:
        print("kept the collected jobs in the store; the next run collects them again")


async def conflict(llm: LLM, config: ModelConfig, store: JsonBatchStore, schema: type[BaseModel] | None) -> int:
    # the stored ids with other prompts. Only stored ids are asked, so every one conflicts and nothing is sent
    async with store.lock():
        ids = sorted(request_id for job in store.load() if job.model == config.to_string() for request_id in job.requests)
    if not ids:
        print("error: the store has no running jobs for this model; `run` first", file=sys.stderr)
        return 2
    changed = [BatchRequest(request_id, SYSTEM, [LLMMessage.user(question(int(request_id.removeprefix("q")), changed=True)[0])]) for request_id in ids]
    try:
        async with llm.batch(config, changed, store, output_schema=schema):
            pass
    except ValueError as error:
        if "already in a running batch" not in str(error):
            raise
        print(f"refused as expected: {error}")
        return 0
    print("error: the changed requests were not refused", file=sys.stderr)
    return 1


def status(store: JsonBatchStore) -> None:
    jobs = store.load()
    if not jobs:
        print("the store tracks no jobs")
    for job in jobs:
        print(f"{job.model} batch_id={job.batch_id} submitted_at={job.submitted_at} requests={list(job.requests)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise the batch API against a real provider.")
    parser.add_argument("action", choices=["run", "wait", "status", "cancel", "conflict"], help="see the usage at the top of this file")
    parser.add_argument("model", nargs="?", type=model_argument, help="<provider>:<model>[:<options>]; every action but status needs it")
    parser.add_argument("--count", type=int, default=3, help="questions q0..q<count-1> to ask (default 3)")
    parser.add_argument("--schema", type=schema_argument, help="pydantic model for structured output: a class in this script (e.g. Answer) or module:Class")
    parser.add_argument("--poll", type=float, default=60.0, help="seconds between status checks for `wait` (default 60)")
    parser.add_argument("--keep", action="store_true", help="keep the collected jobs in the store (exit the batch block with an exception)")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help=f"the batch store file (default {DEFAULT_STORE})")
    args = parser.parse_args()

    if args.action != "status" and args.model is None:
        parser.error(f"{args.action} needs a model")
    if args.count < 1:
        parser.error("--count must be at least 1")
    args.store.parent.mkdir(parents=True, exist_ok=True)
    store = JsonBatchStore(args.store)

    if args.action == "status":
        status(store)
        return 0

    llm = build_llm()
    config: ModelConfig = args.model
    try:
        if args.action == "conflict":
            return run(llm, conflict(llm, config, store, args.schema))
        if args.action == "cancel":
            running = [job.batch_id for job in store.load() if job.provider is config.provider and job.batch_id is not None]
            run(llm, llm.cancel_batches(config.provider, store))
            print(f"asked {config.provider.value} to cancel {len(running)} job(s) {running}; `run` collects them once they have ended")
        else:
            run(llm, batch(llm, config, store, args.count, args.schema, args.action == "wait", args.poll, args.keep))
    except LLMCallError as error:
        print(f"error ({error.kind.value}): {error}", file=sys.stderr)
        return 1
    except (ValueError, NotImplementedError, RuntimeError) as error:
        # an unconfigured provider, a model or provider without batch support, options the model cannot take,
        # a changed request under a running id, or a submission that never confirmed
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
