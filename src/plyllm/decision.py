from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# a choice value is typed: the string "true" and the boolean True are different choices
ChoiceValue = str | bool


@dataclass(frozen=True)
class Predicate:
    # the probability that a statement about the input is true
    name: str
    instructions: str


@dataclass(frozen=True)
class ChoiceOption:
    value: ChoiceValue
    description: str | None = None


@dataclass(frozen=True)
class Choice:
    # one of the options, with a probability for each
    name: str
    instructions: str
    options: list[ChoiceOption]


@dataclass(frozen=True)
class ScoreLevel:
    label: str
    description: str | None = None


@dataclass(frozen=True)
class Score:
    # a rating against ordered levels; level values are their indices, from 0
    name: str
    instructions: str
    levels: list[ScoreLevel]


DecisionQuestion = Predicate | Choice | Score


@dataclass(frozen=True)
class PredicateAnswer:
    name: str
    probability: float


@dataclass(frozen=True)
class ChoiceAnswer:
    name: str
    choice: ChoiceValue
    confidence: float
    probabilities: dict[ChoiceValue, float]


@dataclass(frozen=True)
class ScoreProbability:
    label: str
    value: int
    probability: float


@dataclass(frozen=True)
class ScoreAnswer:
    name: str
    # the probability-weighted level value, so it can fall between levels
    score: float
    confidence: float
    probabilities: list[ScoreProbability]


@dataclass(frozen=True)
class RefusedAnswer:
    # the model declined this question; the others in the request can still be answered
    name: str


DecisionAnswer = PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusedAnswer


def check_questions(questions: Sequence[DecisionQuestion]) -> None:
    # names are required and unique, so every answer can be stored and joined by its question's name
    if not questions:
        raise ValueError("A decision needs at least one question")
    names = [question.name for question in questions]
    if any(not name for name in names):
        raise ValueError("Every decision question needs a non-empty name")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Decision question names must be unique; repeated: {duplicates}")
