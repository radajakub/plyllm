from plyllm.utils import ExtendedEnum


class Provider(str, ExtendedEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    GROK = "grok"
    CLAUDE = "claude"
    MISTRAL = "mistral"
    METACENTRUM = "metacentrum"
    VLLM = "vllm"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
