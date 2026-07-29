from __future__ import annotations

from omniplay.llm.model import LLMModel
from omniplay.llm.options import LLMCallOptions


class HuggingFaceLLMModel(LLMModel):
    def __init__(self, model_name: str, model_string: str, max_length: int = 512) -> None:
        # model_string is the HF repo id; local models are free and carry no cost
        super().__init__(model_name, model_string, input_cost=0.0, output_cost=0.0)
        self.max_length = max_length  # tokenizer truncation length for embedding

    def extract_params(self, options: LLMCallOptions) -> dict[str, object]:
        # generation is not supported; embedding does not consume call options
        return {}


def huggingface_models() -> list[HuggingFaceLLMModel]:
    return [
        HuggingFaceLLMModel("sup-simcse-bert", "princeton-nlp/sup-simcse-bert-base-uncased", max_length=512),
    ]
