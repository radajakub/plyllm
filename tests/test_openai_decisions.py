import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types import Decision

from plyllm.decision import Choice, ChoiceAnswer, ChoiceOption, Predicate, PredicateAnswer, RefusedAnswer, Score, ScoreAnswer, ScoreLevel, ScoreProbability
from plyllm.providers.claude.client import ClaudeLLMClient
from plyllm.providers.openai.client import OpenAILLMClient
from plyllm.tokens import LLMTokens

_MODEL = "gpt-6-luna"

_USAGE = {
    "input_tokens": 1000,
    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 200},
    "output_tokens": 7,
    "output_tokens_details": {"reasoning_tokens": 5},
    "total_tokens": 1007,
}

_QUESTIONS = [
    Predicate("damaged", "Is the item damaged?"),
    Choice("department", "Which department should handle it?", [ChoiceOption("billing", "Payments and invoices"), ChoiceOption("shipping"), ChoiceOption(True)]),
    Score("severity", "How severe is the issue?", [ScoreLevel("low"), ScoreLevel("high", "Blocks the customer")]),
    Predicate("abusive", "Is the message abusive?"),
]

_ANSWERS = [
    {"type": "predicate", "name": "damaged", "probability": 0.8},
    {
        "type": "choice",
        "name": "department",
        "choice": "billing",
        "confidence": 0.7,
        "probabilities": [{"value": "billing", "probability": 0.7}, {"value": "shipping", "probability": 0.2}, {"value": True, "probability": 0.1}],
    },
    {
        "type": "score",
        "name": "severity",
        "score": 0.25,
        "confidence": 0.75,
        "probabilities": [{"label": "low", "value": 0, "probability": 0.75}, {"label": "high", "value": 1, "probability": 0.25}],
    },
    {"type": "refusal", "name": "abusive"},
]


class _FakeDecisions:
    def __init__(self, answers: list[dict]) -> None:
        self.answers = answers
        self.calls: list[dict] = []

    async def create(self, **kwargs) -> Decision:
        self.calls.append(kwargs)
        return Decision.model_validate({"answers": self.answers, "model": "gpt-6-luna-2026-10-01", "usage": _USAGE})


class _FakeClient:
    def __init__(self, answers: list[dict]) -> None:
        self.decisions = _FakeDecisions(answers)


def _client(answers: list[dict] = _ANSWERS) -> tuple[OpenAILLMClient, _FakeDecisions]:
    fake = _FakeClient(answers)
    # This test double implements the decision methods exercised here.
    return OpenAILLMClient(cast(AsyncOpenAI, fake)), fake.decisions


def test_decide_sends_each_question_type():
    client, fake = _client()

    asyncio.run(client.decide(_MODEL, "The parcel arrived torn.", _QUESTIONS))

    call = fake.calls[0]
    assert call["model"] == _MODEL
    assert call["input"] == "The parcel arrived torn."
    assert call["questions"] == [
        {"type": "predicate", "name": "damaged", "instructions": "Is the item damaged?"},
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which department should handle it?",
            # a description is sent only when set, and a boolean value stays a boolean
            "choices": [{"value": "billing", "description": "Payments and invoices"}, {"value": "shipping"}, {"value": True}],
        },
        {"type": "score", "name": "severity", "instructions": "How severe is the issue?", "levels": [{"label": "low"}, {"label": "high", "description": "Blocks the customer"}]},
        {"type": "predicate", "name": "abusive", "instructions": "Is the message abusive?"},
    ]


def test_decide_reads_every_answer_type_and_keeps_a_refusal_among_them():
    client, _ = _client()

    response = asyncio.run(client.decide(_MODEL, "The parcel arrived torn.", _QUESTIONS))

    assert response.model_string == "gpt-6-luna-2026-10-01"
    assert response.answers == [
        PredicateAnswer("damaged", 0.8),
        ChoiceAnswer("department", "billing", 0.7, {"billing": 0.7, "shipping": 0.2, True: 0.1}),
        ScoreAnswer("severity", 0.25, 0.75, [ScoreProbability("low", 0, 0.75), ScoreProbability("high", 1, 0.25)]),
        RefusedAnswer("abusive"),
    ]
    assert response.answer("severity") == response.answers[2]
    with pytest.raises(KeyError):
        response.answer("missing")


def test_decide_records_all_tokens_but_bills_only_input():
    client, _ = _client()

    response = asyncio.run(client.decide(_MODEL, "text", _QUESTIONS))

    assert response.tokens == LLMTokens(input_tokens=1000, cached_input_tokens=200, output_tokens=7, reasoning_tokens=5)
    assert client.calculate_decision_cost(_MODEL, response.tokens) == pytest.approx(1000 / 1_000_000 * 0.1)


def test_decide_rejects_answers_that_do_not_match_the_questions():
    client, _ = _client(_ANSWERS[:2])

    with pytest.raises(ValueError, match="2 answers for 4 questions"):
        asyncio.run(client.decide(_MODEL, "text", _QUESTIONS))


@pytest.mark.parametrize(
    ("questions", "message"),
    [
        ([], "at least one question"),
        ([Predicate("", "Is it?")], "non-empty name"),
        ([Predicate("a", "Is it?"), Predicate("a", "Is it really?")], "repeated: \\['a'\\]"),
        ([Choice("c", "Which?", [ChoiceOption("only")])], "1 options"),
        ([Choice("c", "Which?", [ChoiceOption(str(index)) for index in range(256)])], "256 options"),
        ([Choice("c", "Which?", [ChoiceOption("x"), ChoiceOption("x", "again")])], "repeats an option"),
    ],
)
def test_decide_rejects_invalid_questions_before_sending(questions, message):
    client, fake = _client()

    with pytest.raises(ValueError, match=message):
        asyncio.run(client.decide(_MODEL, "text", questions))
    assert fake.calls == []


def test_a_string_and_a_boolean_choice_are_distinct():
    client, fake = _client([_ANSWERS[1]])

    asyncio.run(client.decide(_MODEL, "text", [Choice("c", "Which?", [ChoiceOption("True"), ChoiceOption(True)])]))

    assert fake.calls[0]["questions"][0]["choices"] == [{"value": "True"}, {"value": True}]


def test_unknown_decision_model_is_rejected():
    client, _ = _client()

    with pytest.raises(ValueError, match="Decision model"):
        asyncio.run(client.decide("gpt-6-sol", "text", _QUESTIONS))


def test_a_provider_without_decisions_refuses():
    client = ClaudeLLMClient(cast(AsyncAnthropic, SimpleNamespace()))

    with pytest.raises(NotImplementedError, match="decisions are not supported"):
        asyncio.run(client.decide(_MODEL, "text", _QUESTIONS))
