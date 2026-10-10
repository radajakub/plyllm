from enum import Enum
from typing import Self

MILLION = 1_000_000


class ExtendedEnum(Enum):
    @classmethod
    def from_value(cls, value: str | int) -> Self | None:
        return next((e for e in cls if str(e.value).lower() == str(value).lower()), None)

    @classmethod
    def values(cls) -> list[str]:
        return [item.value for item in cls]

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return self.__str__()


def to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("true", "True", "1", "yes", "Yes", "YES", "y", "Y"):
        return True
    if value in ("false", "False", "0", "no", "No", "NO", "n", "N"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def extract_params(param_string: str) -> dict[str, str]:
    return {k: v for k, v in (param.split("=") for param in param_string.split(",") if param)}
