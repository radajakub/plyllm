from __future__ import annotations

from plybench.llm.model import EmbeddingModel, EmbeddingTask


class HuggingFaceEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str, model_string: str, max_length: int = 512, batch_size: int = 64) -> None:
        # model_string is the HF repo id; local models are free and carry no cost. The tokenizer
        # truncates to max_length, so oversized inputs are handled rather than rejected.
        super().__init__(
            model_name,
            model_string,
            context_size=max_length,
            input_cost=0.0,
            max_batch_size=batch_size,
            truncates_input=True,
        )

    def format_texts(self, texts: list[str], task: EmbeddingTask) -> list[str]:
        # local encoder models take raw text; the task carries no wire representation
        return texts


def huggingface_embedding_models() -> list[EmbeddingModel]:
    return [
        HuggingFaceEmbeddingModel("sup-simcse-bert", "princeton-nlp/sup-simcse-bert-base-uncased", max_length=512),
    ]
