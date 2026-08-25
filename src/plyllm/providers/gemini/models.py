from __future__ import annotations

from google.genai.types import GenerateContentConfig, ThinkingConfig

from plybench.llm.model import EmbeddingModel, EmbeddingTask, LLMModel
from plybench.llm.options import LLMCallOptions, ReasoningEffort
from plybench.llm.rate_limit import ModelLimits

# Gemini 2.5 models (thinking_budget) and some Gemini 3 Pro variants
_GEMINI_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high"})
# Gemini 3 Flash / 3.5 / 3.6 models that support the full thinking_level set
_GEMINI3_FULL: frozenset[ReasoningEffort] = frozenset({"minimal", "low", "medium", "high"})
# gemini-3-pro-preview
_GEMINI3_PRO: frozenset[ReasoningEffort] = frozenset({"low", "high"})

# the embed endpoint publishes no batch cap, so keep requests small enough to stay well inside the
# per-request token limit and to bound the damage when one batch fails
_EMBEDDING_BATCH_SIZE = 100
# embedding quota is per-endpoint and separate from the generation quota; paced rather than bursted
_EMBEDDING_RPS = 5.0

# older Gemini models take a numeric thinking_budget instead of a level; medium -> -1 (automatic)
_BUDGET_BY_EFFORT: dict[ReasoningEffort, int] = {
    "low": 1024,
    "medium": -1,
    "high": 24576,
}


class GeminiLLMModel(LLMModel):
    def __init__(
        self,
        model_name: str,
        model_string: str,
        input_cost: float,
        output_cost: float,
        thinking: bool = False,
        thinking_only: bool = False,
        uses_thinking_level: bool = False,
        supported_reasoning: frozenset[ReasoningEffort] | None = None,
    ) -> None:
        super().__init__(
            model_name,
            model_string,
            input_cost=input_cost,
            output_cost=output_cost,
            thinking=thinking,
            thinking_only=thinking_only,
            supported_reasoning=supported_reasoning,
        )
        self.uses_thinking_level = uses_thinking_level

    def extract_params(self, options: LLMCallOptions) -> GenerateContentConfig:
        self.validate(options)

        effort: ReasoningEffort = options.reasoning_effort or "high"

        if options.thinking_enabled:
            if self.uses_thinking_level:
                thinking_config = ThinkingConfig(include_thoughts=True, thinking_level=effort)
            else:
                thinking_config = ThinkingConfig(include_thoughts=True, thinking_budget=_BUDGET_BY_EFFORT[effort])
            return GenerateContentConfig(
                temperature=options.temperature,
                max_output_tokens=options.max_tokens,
                response_mime_type="text/plain",
                thinking_config=thinking_config,
            )

        return GenerateContentConfig(
            temperature=options.temperature,
            max_output_tokens=options.max_tokens,
            response_mime_type="text/plain",
        )


# Gemini embedding-2 drops `task_type`; query vs document is a prefix on the input text.
# The strings below are the ones documented at ai.google.dev/gemini-api/docs/embeddings -- a wrong
# prefix degrades retrieval quality silently, so they are verbatim and covered by tests.
_QUERY_TASKS: dict[EmbeddingTask, str] = {
    EmbeddingTask.SEARCH_QUERY: "search result",
    EmbeddingTask.QUESTION_ANSWERING: "question answering",
    EmbeddingTask.FACT_CHECKING: "fact checking",
    EmbeddingTask.CODE_RETRIEVAL: "code retrieval",
    EmbeddingTask.CLASSIFICATION: "classification",
    EmbeddingTask.CLUSTERING: "clustering",
    EmbeddingTask.SEMANTIC_SIMILARITY: "sentence similarity",
}


class GeminiEmbeddingModel(EmbeddingModel):
    def format_texts(self, texts: list[str], task: EmbeddingTask) -> list[str]:
        if task is EmbeddingTask.SEARCH_DOCUMENT:
            # `title: none` is the documented form when a document carries no title
            return [f"title: none | text: {text}" for text in texts]

        prefix = _QUERY_TASKS.get(task)
        if prefix is None:
            raise ValueError(f"No Gemini embedding prefix is defined for task {task}; add it to _QUERY_TASKS")
        return [f"task: {prefix} | query: {text}" for text in texts]


def gemini_models() -> list[GeminiLLMModel]:
    return [
        GeminiLLMModel(
            "gemini-3.7-flash",
            "gemini-3.7-flash",
            input_cost=0.75,
            output_cost=3.75,
            thinking=True,
            thinking_only=True,
            uses_thinking_level=True,
            supported_reasoning=_GEMINI_REASONING,
        ),
        GeminiLLMModel(
            "gemini-3.6-flash", "gemini-3.6-flash", input_cost=1.5, output_cost=7.5, thinking=True, thinking_only=True, uses_thinking_level=True, supported_reasoning=_GEMINI3_FULL
        ),
        GeminiLLMModel(
            "gemini-3.5-flash", "gemini-3.5-flash", input_cost=1.5, output_cost=9.0, thinking=True, thinking_only=True, uses_thinking_level=True, supported_reasoning=_GEMINI3_FULL
        ),
        GeminiLLMModel(
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash-lite",
            input_cost=0.3,
            output_cost=2.5,
            thinking=True,
            thinking_only=True,
            uses_thinking_level=True,
            supported_reasoning=_GEMINI3_FULL,
        ),
        GeminiLLMModel(
            "gemini-3.1-pro",
            "gemini-3.1-pro-preview",
            input_cost=2.0,
            output_cost=12.0,
            thinking=True,
            thinking_only=True,
            uses_thinking_level=True,
            supported_reasoning=_GEMINI_REASONING,
        ),
        GeminiLLMModel(
            "gemini-3.1-flash-lite",
            "gemini-3.1-flash-lite",
            input_cost=0.25,
            output_cost=1.5,
            thinking=True,
            thinking_only=True,
            uses_thinking_level=True,
            supported_reasoning=_GEMINI3_FULL,
        ),
        GeminiLLMModel(
            "gemini-3-flash",
            "gemini-3-flash-preview",
            input_cost=0.5,
            output_cost=3.0,
            thinking=True,
            thinking_only=True,
            uses_thinking_level=True,
            supported_reasoning=_GEMINI3_FULL,
        ),
        GeminiLLMModel(
            "gemini-3-pro", "gemini-3-pro-preview", input_cost=2.0, output_cost=12.0, thinking=True, thinking_only=True, uses_thinking_level=True, supported_reasoning=_GEMINI3_PRO
        ),
        # gemini 2.5 models (thinking_budget, not thinking_level)
        GeminiLLMModel(
            "gemini-2.5-pro",
            "gemini-2.5-pro",
            input_cost=1.25,
            output_cost=10.0,
            thinking=True,
            thinking_only=True,
            uses_thinking_level=False,
            supported_reasoning=_GEMINI_REASONING,
        ),
        GeminiLLMModel(
            "gemini-2.5-flash",
            "gemini-2.5-flash",
            input_cost=0.3,
            output_cost=2.5,
            thinking=True,
            thinking_only=False,
            uses_thinking_level=False,
            supported_reasoning=_GEMINI_REASONING,
        ),
        GeminiLLMModel(
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-lite",
            input_cost=0.1,
            output_cost=0.4,
            thinking=True,
            thinking_only=False,
            uses_thinking_level=False,
            supported_reasoning=_GEMINI_REASONING,
        ),
    ]


def gemini_embedding_models() -> list[EmbeddingModel]:
    return [
        # embedding-2 normalizes every output width, so vectors are comparable without post-processing
        GeminiEmbeddingModel(
            "gemini-embedding-2",
            "gemini-embedding-2",
            context_size=8192,
            input_cost=0.2,
            max_batch_size=_EMBEDDING_BATCH_SIZE,
            limits=ModelLimits(rps=_EMBEDDING_RPS),
        ),
    ]
