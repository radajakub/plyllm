#!/usr/bin/env python3
# Smoke test for a vLLM server we run (e.g. through plyserve): a few judge prompts through plyllm, plus raw probes for
# the registry flags in providers/vllm/models.py. Free (our server), but it needs a running one: VLLM_BASE_URL and
# VLLM_API_KEY come from plyserve (same job), from `plyserve endpoint` (another job) or by hand (through a tunnel).
# The traces are SYNTHETIC tic-tac-toe examples written for this test, not data from an experiment.
#
# Checks, all sent at once (the server batches them):
#   served    the server serves the model under its registry name
#   judge     three traces x {thinking off, thinking on}, the schema asked in the prompt (the registry's setting):
#             does the answer parse, and does the verdict match the expected one
#   effort    one trace at each effort the registry allows: does the reasoning get longer with the effort
#   enforced  the same traces with the schema enforced by the server (response_format), sent directly: finish_reason
#             "length" means the answer ran into max_tokens, the loop seen on Metacentrum (vllm-project/vllm#40080)
#   usage     the raw usage of one answer: does the server report cached and reasoning tokens
#
# Usage:
#   uv run python scripts/smoke_vllm.py vllm:qwen-3.8-27b-fp8 --out smoke-qwen.json

import argparse
import asyncio
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from common import build_llm, model_argument, run
from openai import APIError, AsyncOpenAI
from openai.lib._parsing._completions import type_to_response_format_param
from pydantic import BaseModel, ValidationError

from plyllm import LLM, LLMCallError, LLMCallOptions, LLMConfig, LLMMessage, ModelConfig, Provider
from plyllm.providers.openweight.models import OpenWeightLLMModel

# upper bound on every answer, so a looping one ends; thinking at the highest effort can need several thousand tokens
MAX_TOKENS = 16384

SYSTEM = (
    "You judge the reasoning of a tic-tac-toe player. Cells are numbered 1-9, left to right, top to bottom. "
    "Read the position and the player's reasoning, check every claim against the position, and decide whether the "
    "reasoning is sound or contains a mistake. If it contains one, name the first mistake in one sentence."
)


class Judgement(BaseModel):
    verdict: Literal["sound", "flawed"]
    mistake: str | None


_POSITION = "X: cells 1 and 5. O: cells 3 and 9. Empty: 2, 4, 6, 7, 8. X to move."
# name -> (reasoning, expected verdict). O threatens 3-6-9, so X must take 6
TRACES: dict[str, tuple[str, str]] = {
    "blocks-threat": ("O has 3 and 9, so O wins at 6 next move unless I take it. Taking 6 also gives me 4-5-6 with 4 empty. Move: 6.", "sound"),
    "misses-threat": ("I have 1 and 5. Playing 4 makes two lines at once, 1-4-7 and 4-5-6, so O cannot block both. Move: 4.", "flawed"),
    "misreads-board": ("I hold 1, 5 and 9, so the diagonal 1-5-9 is already mine and any move wins. Move: 2.", "flawed"),
}


def _prompt(reasoning: str) -> str:
    return f"Position: {_POSITION}\n\nPlayer's reasoning:\n{reasoning}"


async def _judge(llm: LLM, config: ModelConfig, check: str, trace: str) -> dict[str, Any]:
    reasoning, expected = TRACES[trace]
    record: dict[str, Any] = {"check": check, "model": config.to_string(), "trace": trace, "expected": expected}
    started = time.monotonic()
    try:
        response = await llm.generate(config, LLMMessage.system(SYSTEM), [LLMMessage.user(_prompt(reasoning))], output_schema=Judgement)
    except LLMCallError as error:
        record.update(ok=False, error=f"{error.kind.value}: {error}", output=error.response.output_text if error.response else None)
    else:
        tokens = response.tokens
        record.update(
            ok=True,
            verdict=Judgement.model_validate_json(response.output_text).verdict,
            output=response.output_text,
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            reasoning_tokens=tokens.reasoning_tokens,
            reasoning_chars=sum(len(part) for part in response.reasoning),
        )
    record["seconds"] = round(time.monotonic() - started, 1)
    return record


async def _enforced(client: AsyncOpenAI, model: OpenWeightLLMModel, options: LLMCallOptions, trace: str) -> dict[str, Any]:
    # what the client would send with weak_structured_output=False: the registry's sampling and template switches, and the strict schema
    reasoning, expected = TRACES[trace]
    record: dict[str, Any] = {"check": "enforced", "thinking": options.thinking_enabled, "trace": trace, "expected": expected}
    kwargs: dict[str, Any] = dict(
        model=model.model_string,
        messages=[LLMMessage.system(SYSTEM).to_dict(), LLMMessage.user(_prompt(reasoning)).to_dict()],
        response_format=type_to_response_format_param(Judgement),
        **model.extract_params(options),
    )
    started = time.monotonic()
    try:
        response = await client.chat.completions.create(**kwargs)
    except APIError as error:
        record.update(ok=False, error=f"{type(error).__name__}: {error}", seconds=round(time.monotonic() - started, 1))
        return record
    choice, content = response.choices[0], response.choices[0].message.content or ""
    try:
        verdict = Judgement.model_validate_json(content).verdict
    except ValidationError:
        verdict = None
    record.update(
        ok=verdict is not None,
        verdict=verdict,
        finish_reason=choice.finish_reason,
        output_tokens=response.usage.completion_tokens if response.usage else None,
        output=content if verdict is not None else f"...{content[-300:]}",
        usage=response.usage.model_dump() if response.usage else None,
        seconds=round(time.monotonic() - started, 1),
    )
    return record


async def _smoke(llm: LLM, config: ModelConfig) -> dict[str, Any]:
    model = llm.resolve_model(Provider.VLLM, config.model_name)
    assert isinstance(model, OpenWeightLLMModel)
    vllm = LLMConfig.from_env().vllm
    assert vllm is not None  # build_llm() would have no vllm provider either
    served = await llm.served_models(Provider.VLLM)

    off = replace(config, options=LLMCallOptions(max_tokens=MAX_TOKENS))
    on = replace(config, options=LLMCallOptions(thinking_enabled=True, max_tokens=MAX_TOKENS))
    calls = [_judge(llm, mode, "judge", trace) for mode in (off, on) for trace in TRACES]
    for effort in sorted(model.supported_reasoning or ()):
        effort_config = replace(config, options=LLMCallOptions(thinking_enabled=True, reasoning_effort=effort, max_tokens=MAX_TOKENS))
        calls.append(_judge(llm, effort_config, "effort", "misses-threat"))
    client = AsyncOpenAI(api_key=vllm.api_key, base_url=vllm.base_url, timeout=vllm.timeout)
    try:
        calls += [_enforced(client, model, mode.options, trace) for mode in (off, on) for trace in TRACES]
        records = await asyncio.gather(*calls)
    finally:
        await client.close()
    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": vllm.base_url,
        "model": config.model_name,
        "served": sorted(served),
        "max_tokens": MAX_TOKENS,
        "records": records,
    }


def _summary(result: dict[str, Any]) -> bool:
    ok = result["model"] in result["served"]
    print(f"served: {result['served']} {'ok' if ok else 'MISSING ' + result['model']}")
    for record in result["records"]:
        mode = record.get("model", f"enforced thinking={record.get('thinking')}")
        verdict = f"{record.get('verdict')} (expected {record['expected']})" if record["ok"] else f"FAILED {record.get('error') or record.get('finish_reason')}"
        extra = f" finish={record['finish_reason']}" if "finish_reason" in record else f" reasoning={record.get('reasoning_chars')} chars"
        print(f"{record['check']:9} {record['trace']:15} {mode:60} {verdict}{extra} out={record.get('output_tokens')} tok {record['seconds']} s")
        ok = ok and (record["ok"] or record["check"] == "enforced")
    usage = next((record["usage"] for record in result["records"] if record.get("usage")), None)
    print(f"usage of one answer: {json.dumps(usage)}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a vLLM server through plyllm.")
    parser.add_argument("model", type=model_argument, help="vllm:<served name>, e.g. vllm:qwen-3.8-27b-fp8; options are ignored")
    parser.add_argument("--out", type=Path, help="write every request and answer here as JSON")
    args = parser.parse_args()
    if args.model.provider is not Provider.VLLM:
        parser.error("the model must be a vllm one")

    llm = build_llm()
    result = run(llm, _smoke(llm, args.model))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if _summary(result) else 1


if __name__ == "__main__":
    sys.exit(main())
