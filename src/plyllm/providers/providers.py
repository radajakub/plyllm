from plybench.utils.enums import ExtendedEnum


class Provider(str, ExtendedEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    GROK = "grok"
    CLAUDE = "claude"
    MISTRAL = "mistral"
    METACENTRUM = "metacentrum"
    HUGGINGFACE = "huggingface"
